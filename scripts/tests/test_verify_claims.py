"""Test wrapper for the claim guard (scripts/verify_claims.py).

Three layers:

  * Whole-repo clean run — the guard must exit 0 on the tree as committed.
  * Detection power — every banned family must fire on a known-bad sample
    and no family may fire on a known-good sample, so the regexes cannot
    silently rot (or silently over-block honest phrasing).
  * End-to-end — a scratch file planted inside the scanned tree must be
    flagged with ALL four families, then removed.

The known-bad samples are built by string concatenation so this source
file never contains a full banned phrase — the guard scans its own test
suite too, and a literal sample here would fail the whole-repo run.

Run:
    python -m unittest scripts.tests.test_verify_claims -v
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import verify_claims  # noqa: E402  (module-level constants only, no DB)

GUARD = PROJECT_ROOT / "scripts" / "verify_claims.py"
SCRATCH = PROJECT_ROOT / "scripts" / "_claim_guard_scratch.py"

# (piece_a, piece_b) pairs whose CONCATENATION is a banned phrase.
BAD_BY_FAMILY = {
    "lap-accuracy-tolerance": [
        ("+/- ", "1 lap"),
        ("accuracy ", "within 1 lap"),
    ],
    "implausible-latency": [
        ("<", "2ms"),
        ("sub-", "5ms"),
    ],
    "rulebook-battery-attribution": [
        ("FIA ", "Battery Minimums"),
        ("FIA", "-battery rule"),
    ],
    "per-corner-ml": [
        ("corner-", "by-corner"),
        ("corner ", "by ", "corner"),
    ],
}

# Honest phrasings that must NEVER be flagged.
GOOD_SAMPLES = [
    "within one lap of the car ahead",          # race mechanics, not accuracy
    "per-sector deploy deltas applied over one lap",
    "sub-second policy rollout",
    "< 50ms budget",
    "under 200 ms measured",
    "self-imposed 10% management reserve",
    "braking-zone pass-mass attribution",
]


def _patterns_for(family):
    return [rx for name, rx in verify_claims.BANNED_PATTERNS if name == family]


class TestWholeRepoClean(unittest.TestCase):
    """The committed tree must pass the guard (exit 0)."""

    def test_repo_is_clean(self):
        if SCRATCH.exists():          # defensive: a crashed earlier run
            SCRATCH.unlink()
        proc = subprocess.run(
            [sys.executable, str(GUARD)],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        self.assertEqual(
            proc.returncode, 0,
            f"claim guard failed on the committed tree:\n{proc.stdout}\n{proc.stderr}")


class TestPatternPower(unittest.TestCase):
    """Each family catches its bad sample; nothing catches a good sample."""

    def test_every_family_fires_on_its_bad_samples(self):
        for family, pairs in BAD_BY_FAMILY.items():
            pats = _patterns_for(family)
            self.assertTrue(pats, f"family {family!r} has no regex registered")
            for pair in pairs:
                sample = "".join(pair)
                with self.subTest(family=family, sample=sample):
                    hit = [p.pattern for p in pats if p.search(sample)]
                    self.assertTrue(
                        hit, f"{family!r} regexes did not fire on {sample!r}")

    def test_good_samples_never_flag(self):
        for sample in GOOD_SAMPLES:
            with self.subTest(sample=sample):
                hits = [name for name, rx in verify_claims.BANNED_PATTERNS
                        if rx.search(sample)]
                self.assertEqual(hits, [], f"honest phrasing flagged: {sample!r}")


class TestEndToEndScratchFile(unittest.TestCase):
    """A planted file with all four families is flagged, then removed."""

    def test_scratch_file_is_flagged_with_all_families(self):
        lines = [
            "# scratch: deliberately banned claim text (deleted by the test)",
            "LATENCY = " + repr("<" + "2ms"),
            "# accuracy " + "within 1 lap of the mark",
            "RULE = " + repr("FIA " + "Battery Minimums"),
            "MODEL = " + repr("corner-" + "by-corner"),
            "",
        ]
        SCRATCH.write_text("\n".join(lines), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(GUARD)],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            self.assertEqual(proc.returncode, 1, proc.stdout)
            for family in BAD_BY_FAMILY:
                self.assertIn(f"[{family}]", proc.stdout,
                              f"family {family!r} missing from guard output")
        finally:
            if SCRATCH.exists():
                SCRATCH.unlink()
        self.assertFalse(SCRATCH.exists(), "scratch file not cleaned up")


if __name__ == "__main__":
    unittest.main(verbosity=2)
