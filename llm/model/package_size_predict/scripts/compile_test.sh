#!/usr/bin/env bash
set -euo pipefail

echo "Compiling Python files to bytecode (syntax check)"
while IFS= read -r -d '' f; do
  python -m py_compile "$f"
done < <(find . -type f -name "*.py" -print0)

echo "Bytecode compilation succeeded."

