"""Make py-cord 2.8 voice receive work with Discord's DAVE end-to-end encryption.

Discord now refuses voice connections without DAVE (close code 4017), and only py-cord 2.8
speaks it, but its receive path is unfinished (pycord issue #3139):

- The transport-decrypted payload is dropped unless DAVE is active, and on the DAVE path
  the RTP header extension is stripped twice, so opus gets a cut-off frame.
- After opus decoding it tries to DAVE-decrypt the PCM again.
- One undecodable packet raises OpusError in the router thread and stops recording.

apply() swaps in a decrypt that does transport -> strip extension -> DAVE (by the speaker's
user ID), and drops packets it can't decrypt or decode instead of crashing. Only applied
to the py-cord versions it was written against; remove once upstream ships a fix.
"""

import logging
import struct
import time
from collections import Counter

import discord

logger = logging.getLogger(__name__)

PATCHED_VERSIONS = ("2.8.0", "2.8.1")

_applied = False

# Packet outcomes, logged every STATS_INTERVAL seconds while audio is arriving
STATS_INTERVAL = 10.0
_stats: Counter = Counter()
_stats_since = time.monotonic()


def _count(outcome: str) -> None:
    global _stats_since
    _stats[outcome] += 1
    now = time.monotonic()
    if now - _stats_since >= STATS_INTERVAL:
        logger.info(f"[DAVE RECV] {dict(_stats)} in {now - _stats_since:.0f}s")
        _stats.clear()
        _stats_since = now


def apply() -> bool:
    """Patch py-cord's receive path. Returns True if the patch is (now) active."""
    global _applied
    if _applied:
        return True
    if discord.__version__ not in PATCHED_VERSIONS:
        logger.info(f"py-cord {discord.__version__}: DAVE receive patch not applied")
        return False

    from discord import opus
    from discord.voice.receive import reader

    try:
        import davey
    except ImportError:
        davey = None

    def decrypt_rtp(self, packet) -> bytes:
        payload = _transport_decrypt(self, packet)
        state = self.client._connection
        dave = state.dave_session
        if davey is None or dave is None or not dave.ready:
            _count("no_dave")
            return payload
        user_id = state.ssrc_user_map.get(packet.ssrc)
        if not user_id:
            _count("unknown_ssrc")
            return b""  # can't decrypt without knowing the speaker; reader skips empty packets
        try:
            data = dave.decrypt(user_id, davey.MediaType.audio, payload)
            _count("ok")
            return data
        except Exception as e:
            # Unencrypted frames are allowed while a call moves into or out of DAVE
            if dave.can_passthrough(user_id):
                _count("passthrough")
                return payload
            _count(f"dave_fail:{type(e).__name__}:{str(e)[:40]}")
            return b""

    def decode_packet(self, packet):
        try:
            return _original_decode(self, packet)
        except opus.OpusError as e:
            _count("opus_error")
            logger.debug(f"Dropping undecodable voice packet from ssrc {self.ssrc}: {e}")
            return packet, b""

    _original_decode = opus.PacketDecoder._decode_packet
    reader.PacketDecryptor.decrypt_rtp = decrypt_rtp
    opus.PacketDecoder._decode_packet = decode_packet
    # Disables the post-decode "DAVE decrypt the PCM" step; decrypt_rtp already did it
    opus.HAS_DAVEY = False

    _applied = True
    logger.info(f"py-cord {discord.__version__}: DAVE voice receive patch applied")
    return True


def _transport_decrypt(decryptor, packet) -> bytes:
    """Transport-layer decrypt, with the RTP header extension body removed."""
    if decryptor.mode != "aead_xchacha20_poly1305_rtpsize":
        return decryptor._decryptor_rtp(packet)  # other modes already strip it correctly

    # rtpsize: the 4-byte extension header is authenticated but not encrypted; the
    # extension body is encrypted together with the payload.
    from discord.voice.receive.reader import CryptoError

    packet.adjust_rtpsize()
    nonce = packet.nonce + b"\x00" * 20
    try:
        result = decryptor.box.decrypt(bytes(packet.data), bytes(packet.header), nonce)
    except Exception as e:
        raise CryptoError(e) from None
    if packet.extended:
        packet.update_extended_header(result)
        _, length = struct.unpack_from(">2sH", bytes(packet.header), len(packet.header) - 4)
        result = result[length * 4:]
    return result
