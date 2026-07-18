#!/usr/bin/env bash
set -euo pipefail

echo "== Python =="
python -V

echo
echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo
echo "== Package imports =="
python - <<'PY'
mods = ['torch', 'vllm', 'swift', 'modelscope']
for m in mods:
    try:
        mod = __import__(m)
        print(f'{m}: {getattr(mod, "__version__", "no_version")}')
    except Exception as e:
        print(f'{m}: NOT_INSTALLED ({type(e).__name__}: {e})')
PY

echo
echo "== Notes =="
echo "Set USE_HF=0 to prefer ModelScope."
echo "If swift install/runtime fails under Python 3.12, retry in Python 3.10 or 3.11."
