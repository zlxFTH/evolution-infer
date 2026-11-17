cd "$(dirname "$0")"
uv run python -u smoke_vllm.py "$@"