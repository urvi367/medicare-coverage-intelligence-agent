"""Dynamic LCD lookup (Phase 3).

LCDs are jurisdictional and too numerous to pre-index sensibly, so they are
resolved at query time: when no NCD governs (or an NCD defers to local
contractors), resolve the MAC for the beneficiary's state, fetch that MAC's
LCDs live, then classify. This package holds the deterministic routing pieces
(state -> MAC, NCD disposition); the dynamic fetch + agentic orchestration
build on top.
"""
