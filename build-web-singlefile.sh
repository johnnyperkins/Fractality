#!/usr/bin/env bash
# Build web/fractality-standalone.html: one self-contained file that runs
# from a file:// double-click (no server). The wasm is base64-embedded, so
# the file is ~4/3 the wasm size; startup pays a one-time decode.
# Needs: rustup target add wasm32-unknown-unknown
#        cargo install wasm-bindgen-cli
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release --target wasm32-unknown-unknown

# no-modules target: classic script defining a global `wasm_bindgen`, the
# only loader shape that works from a file:// opaque origin (ES module
# imports and fetch() are both blocked there).
out=target/wasm-singlefile
wasm-bindgen \
    --target no-modules \
    --no-typescript \
    --out-dir "$out" \
    --out-name fractality \
    target/wasm32-unknown-unknown/release/fractality.wasm

if command -v wasm-opt >/dev/null; then
    wasm-opt -O2 -o "$out/fractality_bg.wasm" "$out/fractality_bg.wasm"
fi

html=web/fractality-standalone.html
{
    cat <<'EOF'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Fractality</title>
  <style>
    html, body {
      margin: 0;
      height: 100%;
      background: #000;
      overflow: hidden;
    }
    #fractality-canvas {
      width: 100%;
      height: 100%;
      display: block;
      outline: none;
    }
    #no-webgpu {
      color: #ccc;
      font: 16px/1.6 system-ui, sans-serif;
      max-width: 40em;
      margin: 20vh auto;
      padding: 0 1em;
      display: none;
    }
  </style>
</head>
<body>
  <canvas id="fractality-canvas"></canvas>
  <div id="no-webgpu">
    <h1>WebGPU required</h1>
    <p>
      Fractality runs its particle simulation in compute shaders, which need
      WebGPU. Use a recent Chrome/Edge, or enable WebGPU in your browser.
    </p>
  </div>
  <script>
EOF
    cat "$out/fractality.js"
    cat <<'EOF'
  </script>
  <script>
    const WASM_B64 =
EOF
    printf '"'
    base64 -w0 "$out/fractality_bg.wasm"
    printf '";\n'
    cat <<'EOF'
    if (!navigator.gpu) {
      document.getElementById("fractality-canvas").style.display = "none";
      document.getElementById("no-webgpu").style.display = "block";
    } else {
      const bytes = Uint8Array.from(atob(WASM_B64), (c) => c.charCodeAt(0));
      wasm_bindgen({ module_or_path: bytes.buffer });
    }
  </script>
</body>
</html>
EOF
} > "$html"

echo "done: $html ($(du -h "$html" | cut -f1)) - double-clickable, no server needed"
