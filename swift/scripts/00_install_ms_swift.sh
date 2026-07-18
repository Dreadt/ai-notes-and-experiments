#!/usr/bin/env bash
set -euo pipefail

# Recommended: install SWIFT in a dedicated env.
# Example:
#   conda create -n swift310 python=3.10 -y
#   conda activate swift310

python -V
pip install 'ms-swift' -U

echo
echo "If you prefer a source install:"
echo "  git clone https://github.com/modelscope/ms-swift.git"
echo "  cd ms-swift && pip install -e ."
