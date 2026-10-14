#!/usr/bin/env python3
"""Regression checks for nearby-creature eligibility.

Run directly from the module root:
  python tools/tests/test_nearby_creature_filter.py
"""

import re
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[2]
NEARBY_SOURCE = MODULE_DIR / "src" / "LLMChatterNearby.cpp"


def _interesting_creature_body() -> str:
    source = NEARBY_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r"static bool IsInterestingCreature\([^)]*\)\s*\{"
        r"(?P<body>.*?)\n\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_hostile_creatures_are_rejected():
    body = _interesting_creature_body()
    hostility_check = body.index("cr->IsHostileTo(bot)")
    template_lookup = body.index("cr->GetCreatureTemplate()")
    assert hostility_check < template_lookup


def test_null_observer_is_rejected():
    body = _interesting_creature_body()
    first_guard = body.split("return false;", 1)[0]
    assert "!bot" in first_guard


def main() -> int:
    test_hostile_creatures_are_rejected()
    test_null_observer_is_rejected()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
