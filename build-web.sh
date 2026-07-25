#!/usr/bin/env bash
# Build the WebGPU wasm bundle into web/.
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --target wasm32-unknown-unknown

wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir web \
    --out-name fractality \
    target/wasm32-unknown-unknown/release/fractality.wasm

# Optional size pass if binaryen is installed.
if command -v wasm-opt >/dev/null; then
    # rustc enables bulk-memory, nontrapping-fptoint, etc. by default;
    # older wasm-opt (e.g. ubuntu's binaryen) rejects them without flags.
    wasm-opt -O2 --all-features -o web/fractality_bg.wasm web/fractality_bg.wasm
fi

echo "done. serve with e.g.: python3 -m http.server -d web 8080"
