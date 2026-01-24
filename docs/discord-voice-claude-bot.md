# Discord Voice Claude Bot

Voice-activated AI assistant that joins Discord voice channels, listens for questions, and responds using Claude Code with full tool access.

## Concept

- Wake word: "Hey Claude"
- Automatic game context from Discord Rich Presence
- Claude Code (headless) processes queries with web search, wiki lookups, etc.
- Speaks responses back via TTS
- Can ask clarifying questions and wait for voice responses

## Architecture

```
Discord Voice Channel
        |
        v
+------------------+
|  Wake Word       |  "Hey Claude" detection
|  (Porcupine/     |
|   Whisper-based) |
+------------------+
        |
        v
+------------------+
|  STT             |  Whisper (local, GPU)
|  Speech-to-Text  |
+------------------+
        |
        v
+------------------+
|  Context         |  Discord Rich Presence
|  Enrichment      |  -> "User is playing Starsector"
+------------------+
        |
        v
+------------------+
|  Claude Code     |  SDK/headless mode
|  (Brain)         |  Tools: WebSearch, WebFetch, etc.
+------------------+
        |
        v
+------------------+
|  TTS             |  Qwen3-TTS (local, GPU)
|  Text-to-Speech  |
+------------------+
        |
        v
Discord Voice Channel (playback)
```

## Hardware Requirements

Target: RTX 3080 Ti (12GB VRAM)

| Component | Model | VRAM |
|-----------|-------|------|
| STT | Whisper medium | ~2GB |
| TTS | Qwen3-TTS-1.7B | ~5-7GB |
| **Total** | | ~7-9GB |

Comfortable fit with headroom.

## Tech Stack

| Component | Library/Service |
|-----------|-----------------|
| Discord bot | discord.py[voice] |
| Wake word | Porcupine or continuous Whisper |
| STT | OpenAI Whisper (local) |
| LLM/Brain | Claude Code SDK |
| TTS | Qwen3-TTS |
| Audio encoding | FFmpeg (Opus for Discord) |

## Installation

```bash
# Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Discord bot
pip install discord.py[voice]

# Whisper (STT)
pip install openai-whisper

# Qwen TTS
pip install qwen-tts
pip install flash-attn  # optional, saves VRAM

# Claude Code SDK
npm install -g @anthropic-ai/claude-code
pip install claude-code-sdk

# Audio processing
pip install pydub numpy
# Also need FFmpeg on PATH
```

## Discord Bot Setup

1. Create application at https://discord.com/developers/applications
2. Bot permissions needed:
   - Connect (voice)
   - Speak (voice)
   - Read Messages
   - View Channels
3. Enable "Message Content Intent" and "Presence Intent" in bot settings
4. Invite bot to server with OAuth2 URL generator

## Code Sketches

### Main Bot Structure

```python
import discord
from discord.ext import commands
import asyncio

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

# Components (initialize these)
whisper_model = None  # Whisper STT
tts_model = None      # Qwen3-TTS
voice_clients = {}    # Track voice connections

@bot.event
async def on_ready():
    print(f"Bot ready: {bot.user}")

@bot.command()
async def join(ctx):
    """Join user's voice channel"""
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        vc = await channel.connect()
        voice_clients[ctx.guild.id] = vc
        # Start listening loop
        asyncio.create_task(listen_loop(ctx))

async def listen_loop(ctx):
    """Continuous listening for wake word + questions"""
    vc = voice_clients[ctx.guild.id]

    while vc.is_connected():
        # Record audio chunk
        audio = await record_audio(vc, duration=3)

        # Transcribe
        text = transcribe(audio)

        # Check for wake word
        if "hey claude" in text.lower():
            # Get the question part
            question = extract_question(text)
            if not question:
                # Listen for follow-up
                audio = await record_audio(vc, duration=10)
                question = transcribe(audio)

            # Get game context
            game = get_current_game(ctx.author)

            # Process with Claude
            response = await process_with_claude(question, game)

            # Speak response
            await speak(vc, response)

bot.run("YOUR_BOT_TOKEN")
```

### Discord Rich Presence (Game Detection)

```python
import discord

def get_current_game(member: discord.Member) -> str | None:
    """Get the game the user is currently playing"""
    for activity in member.activities:
        if isinstance(activity, discord.Game):
            return activity.name
        elif isinstance(activity, discord.Activity):
            if activity.type == discord.ActivityType.playing:
                return activity.name
    return None

# Optional: Map games to their wiki URLs
GAME_WIKIS = {
    "Starsector": "starsector.fandom.com",
    "Arma Reforger": "community.bistudio.com/wiki",
    "Elden Ring": "eldenring.wiki.fextralife.com",
    "Baldur's Gate 3": "bg3.wiki",
    "Factorio": "wiki.factorio.com",
    "Rimworld": "rimworldwiki.com",
}

def get_wiki_for_game(game_name: str) -> str | None:
    """Get wiki URL for a game"""
    for game, wiki in GAME_WIKIS.items():
        if game.lower() in game_name.lower():
            return wiki
    return None
```

### Claude Code Integration

```python
import subprocess
import json

async def process_with_claude(question: str, game: str | None) -> str:
    """Send question to Claude Code headless mode"""

    # Build context-aware prompt
    context = ""
    if game:
        wiki = get_wiki_for_game(game)
        context = f"User is currently playing: {game}\n"
        if wiki:
            context += f"Relevant wiki: {wiki}\n"
        context += "\n"

    prompt = f"{context}User question: {question}"

    # Call Claude Code in headless mode
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True
    )

    response = json.loads(result.stdout)
    return response.get("result", "I couldn't find an answer.")

# Alternative: Using Claude Code SDK for interactive sessions
from claude_code_sdk import query

async def process_with_claude_interactive(question: str, game: str | None, voice_client):
    """Interactive Claude session that can ask clarifying questions"""

    context = f"User is playing: {game}\n\n" if game else ""
    prompt = f"{context}{question}"

    async for event in query(prompt=prompt):
        if event.type == "assistant":
            # Claude's response
            return event.content
        elif event.type == "input_request":
            # Claude needs clarification - speak and listen
            await speak(voice_client, event.content)
            audio = await record_audio(voice_client, duration=10)
            user_response = transcribe(audio)
            # SDK handles feeding this back
            event.respond(user_response)
```

### Whisper STT

```python
import whisper
import numpy as np

# Load model once at startup
whisper_model = whisper.load_model("medium", device="cuda")

def transcribe(audio_data: np.ndarray) -> str:
    """Transcribe audio to text using Whisper"""
    result = whisper_model.transcribe(
        audio_data,
        language="en",
        fp16=True
    )
    return result["text"].strip()
```

### Qwen3-TTS

```python
from qwen_tts import QwenTTS
import torch

# Load model once at startup
tts_model = QwenTTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B",
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

def synthesize_speech(text: str) -> np.ndarray:
    """Convert text to speech audio"""
    audio = tts_model.generate(
        text,
        voice="default",  # or configure voice
    )
    return audio

async def speak(voice_client, text: str):
    """Speak text in Discord voice channel"""
    # Generate audio
    audio = synthesize_speech(text)

    # Save to temp file (discord.py needs file or stream)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write audio to wav file
        write_wav(f.name, audio)

        # Play in Discord
        source = discord.FFmpegPCMAudio(f.name)
        voice_client.play(source)

        # Wait for playback to finish
        while voice_client.is_playing():
            await asyncio.sleep(0.1)
```

### Recording from Discord Voice

```python
import discord
from discord.sinks import WaveSink

async def record_audio(voice_client, duration: float) -> np.ndarray:
    """Record audio from Discord voice channel"""

    sink = WaveSink()
    voice_client.start_recording(sink, finished_callback, ctx)

    await asyncio.sleep(duration)

    voice_client.stop_recording()

    # Get audio data from sink
    # Note: Audio comes as separate streams per user
    # You may need to mix or select primary speaker
    audio_data = sink.get_audio()

    return audio_data
```

## Latency Expectations

| Step | Estimated Time |
|------|----------------|
| Wake word detection | ~0.5s |
| STT (Whisper medium) | ~1-2s |
| Claude Code + tools | ~2-5s |
| TTS generation | ~1-2s |
| **Total** | ~4-9s |

The pause feels natural, like someone thinking before answering.

## Potential Enhancements

1. **Voice cloning** - Clone your own voice or use a character voice
2. **Conversation memory** - Keep context across multiple questions
3. **Push-to-talk mode** - Alternative to wake word
4. **Multiple users** - Track who's speaking and respond appropriately
5. **Slash commands** - `/ask` as text alternative to voice
6. **Streaming TTS** - Start speaking before full response is ready

## Resources

- [discord.py voice docs](https://discordpy.readthedocs.io/en/stable/api.html#voice)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Claude Code SDK](https://docs.anthropic.com/en/docs/claude-code)
- [Porcupine wake word](https://picovoice.ai/platform/porcupine/)
