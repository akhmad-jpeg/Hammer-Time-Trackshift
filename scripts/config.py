"""Central configuration and database connection manager for F1 Telemetry Platform.

Every value is read from environment variables (DB_HOST, DB_USER, DB_PASSWORD,
DB_NAME, DB_PORT).  A <project-root>/.env file is loaded at import time as the
fallback source for any variable the OS environment does not already define
(real environment variables win, so production/CI overrides still work).  This
makes the whole stack portable: copy the project + .env to another machine and
it points at that machine's database without touching code.
"""

import os
from pathlib import Path
import mysql.connector

# Placeholder default -- replace it via the DB_PASSWORD environment variable
# or the .env file.  The app refuses to connect while this placeholder is in
# effect, so a forgotten setup fails loudly instead of guessing a password.
_PLACEHOLDER_PASSWORD = 'CHANGE_ME'


def load_env_file(path: str) -> int:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Dependency-free stand-in for python-dotenv: only keys the OS environment
    does NOT already define are set (real env vars win), blank lines and
    '# comments' are skipped, an optional 'export ' prefix is tolerated and
    values may be single/double-quoted.  Returns how many keys were set.
    """
    loaded = 0
    try:
        with open(path, encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                if line.startswith('export '):
                    line = line[7:].strip()
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] \
                        and value[0] in ('"', "'"):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except OSError:
        pass  # no .env file — plain environment (or defaults) only
    return loaded


# Load <project root>/.env once at import time so every script that imports
# this module (dashboard, trainers, importers, backup/restore, ...) sees the
# same configuration without any code changes.
_env_loaded = load_env_file(str(Path(__file__).resolve().parent.parent / '.env'))

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD', 'password'),
    'database': os.environ.get('DB_NAME', 'f1_strategy'),
    'port': int(os.environ.get('DB_PORT', 3306)),
}


def get_db_connection():
    """Return a MySQL database connection using DB_CONFIG.

    Raises RuntimeError while the password is still the placeholder, so
    real credentials can never be silently replaced by a guess.
    """
    if DB_CONFIG['password'] == _PLACEHOLDER_PASSWORD:
        raise RuntimeError(
            "MySQL password is still the placeholder 'CHANGE_ME' -- set the "
            "DB_PASSWORD environment variable before running. "
            "(Other values can be overridden with DB_HOST / DB_USER / "
            "DB_NAME / DB_PORT; see the README.)"
        )
    return mysql.connector.connect(**DB_CONFIG)
