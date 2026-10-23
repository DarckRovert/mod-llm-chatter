#!/usr/bin/env python3
"""Automated validation for Bot Situational Awareness and Embodied Cognition (Dual-Attention pattern)."""

import importlib
import sys
import types
from pathlib import Path


def _ensure_module(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


for dependency in ('anthropic', 'openai'):
    try:
        importlib.import_module(dependency)
    except ModuleNotFoundError:
        module = _ensure_module(dependency)
        attribute = 'Anthropic' if dependency == 'anthropic' else 'OpenAI'
        setattr(module, attribute, type(attribute, (), {}))

try:
    importlib.import_module('mysql.connector')
except ModuleNotFoundError:
    mysql_module = _ensure_module('mysql')
    connector_module = _ensure_module('mysql.connector')
    setattr(mysql_module, 'connector', connector_module)

TOOLS_DIR = Path(__file__).resolve().parents[1]
MODULE_DIR = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from chatter_proximity import (
    build_proximity_bot_situational_lines,
    _player_say_single_prompt,
    _player_say_conversation_prompt,
    _player_emote_single_prompt,
    _player_emote_conversation_prompt,
    _single_prompt,
    _conversation_prompt,
)


class _DB:
    def cursor(self, *args, **kwargs):
        return self

    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        return None

    def fetchall(self):
        return []


NORMAL_CONFIG = {'LLMChatter.ChatterMode': 'normal'}
ROLEPLAY_CONFIG = {'LLMChatter.ChatterMode': 'roleplay'}

BASE_EXTRA = {
    'player_guid': 101,
    'player_name': 'Khadgar',
    'zone_id': 12,
    'zone_name': 'Elwynn Forest',
    'subzone_name': 'Goldshire',
    'map_id': 0,
    'instance_id': 0,
    'map_name': 'Eastern Kingdoms',
    'is_dungeon': False,
    'is_raid': False,
    'nearby_names': [],
    'max_lines': 2,
}


def test_npc_never_receives_situational_lines():
    npc_speaker = {
        'name': 'Marshal Dughan',
        'is_npc': True,
        'bot_state': {'health_pct': 10, 'in_combat': True},
        'current_activity': 'fishing',
    }
    lines = build_proximity_bot_situational_lines(npc_speaker, BASE_EXTRA)
    assert lines == []


def test_combat_urgency_and_target():
    bot = {
        'name': 'Borok',
        'is_npc': False,
        'bot_guid': 555,
        'class': 'Warrior',
        'race': 'Orc',
        'bot_state': {
            'health_pct': 80,
            'mana_pct': -1,
            'in_combat': True,
            'target': 'Defias Pillager',
        },
        'current_activity': 'combat',
    }
    lines = build_proximity_bot_situational_lines(bot, BASE_EXTRA)
    joined = " ".join(lines)
    assert "SITUATIONAL AWARENESS: You are currently fighting Defias Pillager!" in joined
    assert "urgent, breathless, or telegraphic (2-6 words)" in joined
    assert "Do NOT engage in calm, casual, or leisurely conversation" in joined


def test_critical_health_and_mana_exhaustion():
    bot = {
        'name': 'Kaelis',
        'is_npc': False,
        'bot_guid': 556,
        'class': 'Priest',
        'race': 'Blood Elf',
        'bot_state': {
            'health_pct': 18,
            'mana_pct': 5,
            'in_combat': False,
            'target': '',
        },
        'current_activity': 'standing',
    }
    lines = build_proximity_bot_situational_lines(bot, BASE_EXTRA)
    joined = " ".join(lines)
    assert "critically wounded (18% health)" in joined
    assert "acute physical pain and distress" in joined
    assert "MANA EXHAUSTION: You are completely drained of magical energy (5% mana)" in joined


def test_dual_attention_activities():
    activities = [
        ('fishing', 'fishing with a rod', 'bobber'),
        ('gathering', 'gathering resources', 'tools'),
        ('crafting', 'actively crafting', 'trade skills'),
        ('eating_drinking', 'eating food or drinking', 'bites or sips'),
        ('sitting_resting', 'resting your legs', 'seated, relaxed posture'),
        ('mounted', 'mounted on your steed/mount', 'reins'),
        ('swimming', 'swimming/treading water', 'paddling'),
        ('dead', 'ghost / releasing spirit', 'cannot speak normally'),
    ]
    for act, phrase1, phrase2 in activities:
        bot = {
            'name': 'TestBot',
            'is_npc': False,
            'bot_guid': 700,
            'bot_state': {'health_pct': 100, 'mana_pct': 100, 'in_combat': False},
            'current_activity': act,
        }
        lines = build_proximity_bot_situational_lines(bot, BASE_EXTRA)
        joined = " ".join(lines)
        assert phrase1 in joined, f"Expected {phrase1} in {joined} for activity {act}"
        assert phrase2 in joined, f"Expected {phrase2} in {joined} for activity {act}"


def test_taxi_flight_and_mount_fallback():
    bot = {
        'name': 'Skydancer',
        'is_npc': False,
        'bot_guid': 701,
        'bot_state': {
            'health_pct': 100,
            'mana_pct': 100,
            'in_combat': False,
            'travel_state': {'mode': 'taxi_flight'},
        },
        'current_activity': 'standing',
    }
    lines = build_proximity_bot_situational_lines(bot, BASE_EXTRA)
    joined = " ".join(lines)
    assert "flying on a flight master taxi mount" in joined


def test_social_level_and_guild_relations():
    veteran_bot = {
        'name': 'HighGeneral',
        'is_npc': False,
        'bot_guid': 702,
        'level_diff': 25,
        'same_guild': True,
        'bot_state': {},
        'current_activity': 'standing',
    }
    lines = build_proximity_bot_situational_lines(veteran_bot, BASE_EXTRA)
    joined = " ".join(lines)
    assert "vastly higher level veteran (+25 levels higher than the player)" in joined
    assert "GUILD AFFILIATION: The player is a fellow guildmate" in joined

    novice_bot = {
        'name': 'Newbie',
        'is_npc': False,
        'bot_guid': 703,
        'level_diff': -30,
        'bot_state': {},
        'current_activity': 'standing',
    }
    novice_lines = build_proximity_bot_situational_lines(novice_bot, BASE_EXTRA)
    novice_joined = " ".join(novice_lines)
    assert "legendary high-level hero (30 levels above you)" in novice_joined


def test_player_say_single_prompt_combat_override():
    bot_combat = {
        'name': 'Grimbold',
        'is_npc': False,
        'bot_guid': 800,
        'class': 'Warrior',
        'race': 'Dwarf',
        'bot_state': {
            'health_pct': 22,
            'mana_pct': -1,
            'in_combat': True,
            'target': 'Loch Modan Crocolisk',
        },
        'current_activity': 'combat',
    }
    extra = dict(BASE_EXTRA, player_message="Can you give me directions?")
    prompt = _player_say_single_prompt(
        _DB(), extra, bot_combat, extra['player_message'], [], config=ROLEPLAY_CONFIG
    ).user_prompt

    # Length prompt line tightened to 2-6 words
    assert "Write an urgent, breathless /say reply of 2-6 words." in prompt
    assert "Keep it urgent and authentic to active battle." in prompt
    assert "SITUATIONAL AWARENESS: You are currently fighting Loch Modan Crocolisk!" in prompt
    assert "critically wounded (22% health)" in prompt


def test_player_say_single_prompt_fishing_dual_attention():
    bot_fishing = {
        'name': 'Angler',
        'is_npc': False,
        'bot_guid': 801,
        'class': 'Hunter',
        'race': 'Night Elf',
        'bot_state': {
            'health_pct': 100,
            'mana_pct': 100,
            'in_combat': False,
        },
        'current_activity': 'fishing',
    }
    extra = dict(BASE_EXTRA, player_message="Good day! Nice weather for fishing.")
    prompt = _player_say_single_prompt(
        _DB(), extra, bot_fishing, extra['player_message'], [], config=NORMAL_CONFIG
    ).user_prompt

    assert "CURRENT PHYSICAL ACTIVITY (Dual-Attention): You are currently fishing with a rod" in prompt
    assert "watching the water and bobber" in prompt


def test_player_say_conversation_prompt_bot_situational():
    bot1 = {
        'name': 'MinerBob',
        'is_npc': False,
        'bot_guid': 901,
        'current_activity': 'gathering',
        'bot_state': {'health_pct': 100, 'in_combat': False},
    }
    bot2 = {
        'name': 'HungryDan',
        'is_npc': False,
        'bot_guid': 902,
        'current_activity': 'eating_drinking',
        'bot_state': {'health_pct': 60, 'in_combat': False},
    }
    extra = dict(BASE_EXTRA, max_lines=2)
    prompt = _player_say_conversation_prompt(
        _DB(), extra, [bot1, bot2], "Hey there guys!", [], config=NORMAL_CONFIG
    ).user_prompt

    assert "Situational awareness for MinerBob:" in prompt
    assert "gathering resources (mining/herbalism/skinning)" in prompt
    assert "Situational awareness for HungryDan:" in prompt
    assert "eating food or drinking to regain strength" in prompt


def test_player_emote_prompts_situational():
    bot_mounted = {
        'name': 'Cavalier',
        'is_npc': False,
        'bot_guid': 903,
        'current_activity': 'mounted',
        'bot_state': {'health_pct': 100, 'in_combat': False},
    }
    extra = dict(BASE_EXTRA, addressed_name='Cavalier', player_emote='salute')
    prompt = _player_emote_single_prompt(
        _DB(), extra, bot_mounted, 'salute', config=ROLEPLAY_CONFIG
    ).user_prompt

    assert "CURRENT PHYSICAL ACTIVITY (Dual-Attention): You are mounted on your steed/mount." in prompt


def test_ambient_prompts_situational():
    bot_crafting = {
        'name': 'BlacksmithJor',
        'is_npc': False,
        'bot_guid': 904,
        'current_activity': 'crafting',
        'bot_state': {'health_pct': 100, 'in_combat': False},
    }
    single_p = _single_prompt(
        _DB(), BASE_EXTRA, bot_crafting, 'weapons and armor', config=NORMAL_CONFIG
    ).user_prompt
    assert "CURRENT PHYSICAL ACTIVITY (Dual-Attention): You are actively crafting" in single_p

    conv_p = _conversation_prompt(
        _DB(), BASE_EXTRA, [bot_crafting], config=NORMAL_CONFIG
    ).user_prompt
    assert "Situational awareness for BlacksmithJor:" in conv_p


def test_cpp_code_structure_verification():
    cpp_file = MODULE_DIR / 'src' / 'LLMChatterProximity.cpp'
    content = cpp_file.read_text(encoding='utf-8')

    assert 'static std::string GetBotCurrentActivity(Player* bot)' in content
    assert 'si->SpellName' in content
    assert 'sName.find("fish")' in content
    assert 'sName.find("min")' in content
    assert 'SPELL_AURA_MOD_REGEN' in content
    assert 'bot->IsMounted()' in content
    assert 'bot->isSwimming()' in content
    assert 'std::string BuildBotParticipantJson(Player* bot, Player* player' in content
    assert r'\"current_activity\"' in content
    assert r'\"level\"' in content
    assert r'\"level_diff\"' in content
    assert r'\"same_guild\":true' in content
    assert 'BuildBotStateJson(bot)' in content
    assert 'candidate.bot, player' in content
