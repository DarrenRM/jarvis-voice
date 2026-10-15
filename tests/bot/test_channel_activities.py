"""SpeakerManager.channel_activities: what the others in the voice channel are doing
(the game notes go in when anyone there is playing the game)."""

from types import SimpleNamespace

import discord

from src.bot.speaker_manager import SpeakerManager


def member(user_id, *activities, bot=False):
    return SimpleNamespace(id=user_id, bot=bot, activities=list(activities))


def test_lists_others_skipping_bots_and_the_speaker():
    channel = SimpleNamespace(members=[
        member(1, discord.Game("Enshrouded")),          # the speaker
        member(2, discord.Game("Enshrouded")),
        member(3),                                       # shows nothing
        member(4, discord.Game("Jarvis stuff"), bot=True),
    ])
    assert SpeakerManager({}).channel_activities(channel, exclude_id=1) == ["Playing Enshrouded"]


def test_no_channel():
    assert SpeakerManager({}).channel_activities(None) == []
