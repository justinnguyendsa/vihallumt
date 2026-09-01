#!/usr/bin/env bash
# Tải bộ dữ liệu HalOmi (Dale et al., 2023) - benchmark gốc của paper P1.
# Dùng được cả trên Colab/Kaggle lẫn máy cá nhân.
set -euo pipefail
DEST="${1:-data/raw}"
URL="https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip"
mkdir -p "$DEST"
if [ -f "$DEST/halomi_full.tsv" ]; then
  echo "[skip] HalOmi da ton tai tai $DEST"
  exit 0
fi
echo "[1/2] Tai $URL"
curl -L --fail --progress-bar -o "$DEST/halomi_release_v2.zip" "$URL"
echo "[2/2] Giai nen"
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$DEST/halomi_release_v2.zip" "$DEST"
rm -f "$DEST/halomi_release_v2.zip"
find "$DEST" -name "halomi*" -o -name "*.tsv" | head -20
