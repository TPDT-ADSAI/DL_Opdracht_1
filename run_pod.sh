#!/usr/bin/env bash
# One-command pod runner voor DL_Opdracht_1 max-punten rerun.
# Gebruik op runpod: bash run_pod.sh
set -euo pipefail

REPO_DIR="/workspace/DL_Opdracht_1"

if [ ! -d "$REPO_DIR" ]; then
  cd /workspace
  git clone https://github.com/TPDT-ADSAI/DL_Opdracht_1.git
fi
cd "$REPO_DIR"

echo "[1/5] Pull latest"
git pull origin main

if [ ! -d .venv ]; then
  echo "[2/5] First-time setup (python3.11 + torch cu128 + deps)"
  bash setup_5090.sh
else
  echo "[2/5] venv exists, skip setup"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[3/5] Bump CNN seeds 2->3 voor lagere MAPE"
python - <<'PY'
import json
nb = json.load(open('main.ipynb', encoding='utf-8'))
for c in nb['cells']:
    s = ''.join(c.get('source', []))
    if 'N_SEEDS_CNN = 2' in s:
        c['source'] = s.replace('N_SEEDS_CNN = 2', 'N_SEEDS_CNN = 3').splitlines(keepends=True)
        break
json.dump(nb, open('main.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print("seeds bumped 2->3")
PY

echo "[4/5] Execute notebook (45-60 min voor CNN + rest)"
jupyter nbconvert --to notebook --execute main.ipynb \
  --output main.ipynb --ExecutePreprocessor.timeout=14400

echo "[5/5] HTML export + commit + push"
jupyter nbconvert --to html main.ipynb
git add main.ipynb main.html submissions/
git commit -m "rerun: CNN 3 seeds pod + HTML export"
git push origin main

echo ""
echo "KLAAR. Check CNN MAPE:"
grep -A2 'CNN OOF MAPE' main.ipynb 2>/dev/null | head -5 || echo "(run grep lokaal)"
echo ""
echo "Stop pod in Runpod dashboard om factuur te stoppen."
