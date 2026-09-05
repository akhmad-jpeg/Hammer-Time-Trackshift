"""Full-database backup / restore for transferring the platform to another system.

The schema file database/f1_strategy.sql only carries structure + lookup seeds;
the content (laps, telemetry, sessions, ...) lives in MySQL and is dumped by
this script.  Uses the system's own mysqldump / mysql client so the dump is a
standard, portable .sql file (schema + data + routines + triggers), about
7 MB for a typical local dataset.

Usage:
    python scripts/db_transfer.py export            # -> database/exports/f1_strategy_full_<ts>.sql
    python scripts/db_transfer.py export --out db.sql
    python scripts/db_transfer.py restore db.sql    # applies the dump (drops + recreates DB_NAME)

Connection details come from scripts/config.py (os env, falling back to the
project's .env file).  The password is passed to the clients via the MYSQL_PWD
environment variable so it never appears on a command line.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Make scripts/ importable whichever directory the script is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_CONFIG  # noqa: E402  (imports config -> loads .env)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_DIR = PROJECT_ROOT / 'database' / 'exports'

# Where the MySQL client binaries usually live on Windows / Linux.  The PATH is
# tried first; anything found here is a fallback.
_CANDIDATE_BINS = [
    r'C:\Program Files\MySQL\MySQL Server 8.0\bin',
    r'C:\Program Files\MySQL\MySQL Server 8.4\bin',
    '/usr/bin',
    '/usr/local/bin',
    '/opt/homebrew/bin',
]


def _find_client(name: str) -> str:
    """Locate mysqldump / mysql: PATH first, then common install directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in _CANDIDATE_BINS:
        candidate = Path(d) / (name + ('.exe' if os.name == 'nt' else ''))
        if candidate.exists():
            return str(candidate)
    raise SystemExit(
        f"[ERROR] Could not find '{name}'. Install MySQL client tools or add "
        f"its bin directory to PATH.")


def _base_env() -> dict:
    """Environment for the child processes: the DB password only via MYSQL_PWD
    (never on the command line)."""
    env = os.environ.copy()
    env['MYSQL_PWD'] = DB_CONFIG['password']
    return env


def _conn_args() -> list:
    return ['--host=%s' % DB_CONFIG['host'],
            '--port=%s' % DB_CONFIG['port'],
            '--user=%s' % DB_CONFIG['user']]


def export_database(out_path: Path = None) -> Path:
    if out_path is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = DEFAULT_EXPORT_DIR / f'f1_strategy_full_{ts}.sql'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mysqldump = _find_client('mysqldump')

    # --databases <name> makes the dump self-contained: it emits CREATE DATABASE
    # IF NOT EXISTS + USE, so it can be restored onto any server as-is.
    cmd = [mysqldump, *_conn_args(),
           '--single-transaction', '--routines', '--triggers',
           '--default-character-set=utf8mb4',
           '--databases', DB_CONFIG['database']]
    print(f"[BACKUP] Dumping '{DB_CONFIG['database']}' ({mysqldump}) ...")
    with open(out_path, 'wb') as out:
        subprocess.run(cmd, check=True, stdout=out, stderr=subprocess.PIPE,
                       env=_base_env())
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[BACKUP] Wrote {out_path} ({size_mb:.1f} MB)")
    print(f"[BACKUP] Restore on the target machine with:\n"
          f"    python scripts/db_transfer.py restore \"{out_path}\"")
    return out_path


def restore_database(dump_path: Path) -> None:
    if not dump_path.exists():
        raise SystemExit(f"[ERROR] Dump file not found: {dump_path}")
    mysql = _find_client('mysql')
    print(f"[RESTORE] Applying {dump_path} to '{DB_CONFIG['host']}:"
          f"{DB_CONFIG['port']}' ...")
    print(f"[RESTORE] WARNING: tables in database '{DB_CONFIG['database']}' "
          f"will be dropped and recreated.")
    with open(dump_path, 'rb') as src:
        subprocess.run([mysql, *_conn_args()], check=True, stdin=src,
                       stderr=subprocess.PIPE, env=_base_env())
    print('[RESTORE] Done.')


def main():
    parser = argparse.ArgumentParser(
        description='Backup / restore the full f1_strategy database.')
    sub = parser.add_subparsers(dest='command', required=True)

    p_exp = sub.add_parser('export', help='dump schema + data to a .sql file')
    p_exp.add_argument('--out', help='output .sql path (default: '
                                     'database/exports/f1_strategy_full_<ts>.sql)')

    p_res = sub.add_parser('restore', help='apply a dump (drops + recreates the DB)')
    p_res.add_argument('dump', help='path to a .sql file produced by export')

    args = parser.parse_args()
    if args.command == 'export':
        export_database(Path(args.out) if args.out else None)
    else:
        restore_database(Path(args.dump))


if __name__ == '__main__':
    main()
