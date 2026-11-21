set -euo pipefail

cd "$(dirname "$0")"
uv run python gen_random_in.py
uv run python -u smoke_cache.py
