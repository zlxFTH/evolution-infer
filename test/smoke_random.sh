cd "$(dirname "$0")"
uv run python gen_random_in.py     
uv run python gen_random_ans.py    
uv run python run_random_out.py    
uv run python smoke_random.py      