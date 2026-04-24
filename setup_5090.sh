#!/usr/bin/env bash
# RTX 5090 (Blackwell sm_120) pod setup voor DL_Opdracht_1
# Gebruik: bash setup_5090.sh
set -euo pipefail

REPO_DIR="/workspace/DL_Opdracht_1"

if [ ! -d "$REPO_DIR" ]; then
  echo "[1/6] Clone repo"
  cd /workspace
  git clone https://github.com/TPDT-ADSAI/DL_Opdracht_1.git
fi
cd "$REPO_DIR"

echo "[2/6] Python 3.11 venv"
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "[3/6] Torch cu128 (Blackwell sm_120 support)"
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "[4/6] Project requirements + jupyter"
pip install -r requirements.txt
pip install jupyter ipykernel nbconvert

echo "[5/6] Kaggle creds check"
if [ -f runpod.env ] && [ ! -f .env ]; then
  cp runpod.env .env
fi
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi
mkdir -p ~/.kaggle
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
  printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

echo "[6/6] GPU check"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute capability: sm_{}{}".format(*cap))
PY

cat <<'EOF'

Setup klaar. Run:
  source .venv/bin/activate
  jupyter nbconvert --to notebook --execute main.ipynb \
    --output main.ipynb --ExecutePreprocessor.timeout=14400
  jupyter nbconvert main.ipynb --to html --output main.html
  git add main.ipynb main.html submissions/
  git commit -m "pod-5090: CNN seed-avg + TTA rerun"
  git push origin main
EOF
