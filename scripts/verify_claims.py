"""Claim guard: fail loudly if banned unverifiable claim text reappears.

The submission's credibility rests on only claiming what the repo can
reproduce.  Four families of marketing phrasing were therefore purged from
every active doc, comment and generated artifact, and this script keeps
them out:

  1. Lap-level accuracy tolerance   — quoting a lap-count bound on
     prediction accuracy (a signed plus/minus-lap figure, e.g. "within
     N laps").  The repo states MAE / R^2 from committed backtests
     instead; a lap-count tolerance was never measured.
  2. Single-digit millisecond latency  — an inference figure below ten
     milliseconds written with a less-than sign.  Measured budgets are
     committed in code and payloads (single-pair live calls, ~200 ms
     class; policy rollouts, seconds), so such a figure is fiction.
  3. Governing-body rule attribution — attributing the energy-management
     floor to official battery rules.  The 10% floor is a SELF-IMPOSED
     management reserve; no rulebook value is encoded or cited.
  4. Per-corner ML attribution      — the per-corner spatial phrasing.
     The spatial feature is braking-zone pass-mass attribution, which is
     what the artifacts actually compute.

Scope: every text file under the project root with a known source/doc
extension (.py .md .txt .html .js .css .json .bat .ps1 .sh .cfg .ini
.yaml .yml), excluding VCS/venv/cache/database-dump directories.  Files
over 5 MB (bulk generated fleet artifacts) are listed as skipped rather
than scanned; binary files are detected and skipped.

Exit codes: 0 = clean, 1 = at least one banned pattern found.

Run:
    python scripts/verify_claims.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Extensions scanned (source, docs, markup, generated readable artifacts).
SCAN_EXTENSIONS = {
    '.py', '.md', '.txt', '.html', '.js', '.css', '.json',
    '.bat', '.ps1', '.sh', '.cfg', '.ini', '.yaml', '.yml',
}

# Never scanned: VCS internals, caches, environments, SQL dumps.
SKIP_DIRS = {
    '.git', '__pycache__', '.freebuff', '.idea', '.vscode',
    'node_modules', '.venv', 'venv', 'env',
    'database',           # SQL dumps: data, not claims
}

# Bulk generated artifacts above this size are skipped (listed, not silent).
MAX_SCAN_BYTES = 5 * 1024 * 1024

# Files exempt from scanning, with reasons.  DECK_FIXES.md is the deck
# remediation sheet: its job is to QUOTE the banned claim strings as examples
# of what to remove, so it can never satisfy the guard without defeating its
# own purpose.  Keep this list minimal and justified — every entry is a hole
# in the guard.
EXEMPT_FILES = {
    'DECK_FIXES.md',
    'AUDIT_AND_ACTION_PLAN.txt',
}

# Each pattern: (family name, compiled regex).  All case-insensitive.
BANNED_PATTERNS = [
    # 1. Lap-count accuracy tolerance: a signed plus/minus-lap figure.
    #    Deliberately narrow: bare "within N laps" is legitimate when it
    #     describes mechanics (e.g. per-sector deltas applied over a lap),
    #     so the ban requires the accuracy/precision framing or the sign.
    ("lap-accuracy-tolerance",
     re.compile(r"(±|\+/-)\s*\d+(\.\d+)?\s*lap"
                r"|accuracy\s+(within|to)\s+(±|\+/-|a|an|\d|one|two)"
                r"|(proven|precise|accurate)[^\n]{0,40}within\s+\d+(\.\d+)?\s*lap",
                re.IGNORECASE)),
    # 2. Single-digit millisecond latency claims.
    ("implausible-latency",
     re.compile(r"<\s*\d(\.\d+)?\s*ms\b"
                r"|sub-?\s*\d\s*ms\b"
                r"|under\s+\d\s*ms\b",
                re.IGNORECASE)),
    # 3. Official-rulebook attribution for the battery floor.
    ("rulebook-battery-attribution",
     re.compile(r"\bfia[\s_-]*(batter|batt|soc|storage|minimum)",
                re.IGNORECASE)),
    # 4. Per-corner ML attribution: the hyphenated spatial phrasing.
    #    (Written split here so the guard never flags its own comment.)
    ("per-corner-ml",
     re.compile(r"corner[\s_-]*by[\s_-]*corner",
                re.IGNORECASE)),
]


def _is_binary(path: Path) -> bool:
    try:
        with open(path, 'rb') as fh:
            return b'\x00' in fh.read(1024)
    except OSError:
        return True


def scan_text(text: str):
    """Yield (family, match-text) for every banned pattern in `text`."""
    for family, rx in BANNED_PATTERNS:
        for m in rx.finditer(text):
            yield family, m.group(0)


def iter_scan_files():
    """Yield (path, kind) where kind is 'scan' | 'skip-large' | 'skip-binary'."""
    for path in sorted(PROJECT_ROOT.rglob('*')):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in EXEMPT_FILES:
            yield path, 'skip-exempt'
            continue
        if path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_SCAN_BYTES:
            yield path, 'skip-large'
            continue
        if _is_binary(path):
            yield path, 'skip-binary'
            continue
        yield path, 'scan'


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    violations = []
    n_scanned = n_skipped = 0
    for path, kind in iter_scan_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if kind != 'scan':
            print(f"  [SKIPPED {kind}] {rel}")
            n_skipped += 1
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError as exc:
            print(f"  [UNREADABLE] {rel}: {exc}")
            n_skipped += 1
            continue
        n_scanned += 1
        for lineno, line in enumerate(text.splitlines(), start=1):
            for family, snippet in scan_text(line):
                violations.append((rel, lineno, family, snippet.strip()))

    if violations:
        print("BANNED CLAIM STRINGS FOUND "
              f"({len(violations)} in {n_scanned} scanned files):\n")
        for rel, lineno, family, snippet in violations:
            print(f"  {rel}:{lineno}:  [{family}]  {snippet!r}")
        print("\nRewrite these as the claim the repo can actually back:")
        print("  lap-count accuracy  -> committed MAE / R^2 backtest metrics")
        print("  tiny latency        -> the measured budget stated in the code")
        print("  battery floor       -> self-imposed 10% management reserve")
        print("  spatial model       -> braking-zone pass-mass attribution")
        return 1

    print(f"OK - no banned claim strings in {n_scanned} scanned files "
          f"({n_skipped} skipped as generated/binary/large).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
