# Fractality

**An audio-reactive fractal particle visualizer.** Millions of GPU particles swarm across the Mandelbrot set (and friends), driven by compute shaders and whatever music your machine is playing.

**[▶ Run it in your browser](https://johnnyperkins.github.io/Fractality/)** (needs WebGPU: Chrome, Edge, or recent Firefox)

Built in Rust with [Bevy](https://bevyengine.org) and WGSL compute shaders. Runs native on Linux and on the web via WASM + WebGPU.

## What it does

- **12 million particles** (4M on web), fully simulated on the GPU in a WGSL compute shader
- **5 fractals**: Mandelbrot, Burning Ship, Tricorn, Multibrot-3, and a morphing Julia set
- **Deep zoom** using perturbation theory: a high-precision reference orbit on the CPU, per-particle deltas on the GPU, with rebasing (Zhuoran's method) when the delta outgrows the reference
- **Audio reactivity**: captures system audio (native) or tab/mic audio (web) and drives particle motion, beat-synced chromatic bloom, drop detection, and an "audio aurora" color mode where bass and treble anchor the palette
- **6 flow modes** (contour, layers, gravity, erupt, pulse, dynamics), each a different way for particles to ride the fractal's field, crossfaded on switch
- **5 palettes**: classic, rings, electric, inferno, audio aurora
- **Kaleidoscope mode** with adjustable folds and spin
- **Trails, dissolve, auto-zoom dive**, view bookmarks you can fly between, and mouse blast/vortex forces
- **Auto-choreographer**: an attract-mode autopilot that dives into the busiest boundary in view, restyles palette, flow, kaleidoscope, and trails as it goes (palette swaps land on music drops), and cuts to a new scene at the dive floor
- **Recording built in**: `P` for a screenshot, `O` to record video (H.264 MP4 on native via openh264, WebM via MediaRecorder on web)

## Controls

| Key | Action |
|-----|--------|
| `W A S D` | Pan |
| Scroll | Zoom at cursor |
| Left / right mouse (hold) | Blast / vortex |
| `F` | Cycle fractal |
| `G` | Cycle flow mode |
| `C` | Cycle palette |
| `K` | Toggle kaleidoscope |
| `V` | Toggle audio reactivity |
| `Space` | Toggle dissolve |
| `Z` | Auto-zoom dive |
| `X` | Toggle auto-choreographer |
| `R` | Reset view |
| `Shift+1..9` | Save bookmark |
| `1..9` | Fly to bookmark |
| `P` | Screenshot |
| `O` | Record video |
| `M` / `Esc` | Toggle menu (sliders for everything) |

With the menu hidden, the choreographer also engages on its own after 30 s without input; any input then hands control back.

## Running

### Native (Linux)

```sh
cargo run --release
cargo run --release -- 4_000_000   # optional starting particle count (default 2M)
```

Audio reactivity records the default output's monitor via PipeWire's `pw-record`, so play music anywhere and the fractal hears it.

### Web

Needs `rustup target add wasm32-unknown-unknown` and `wasm-bindgen-cli` (matching the `wasm-bindgen` version in `Cargo.lock`); `wasm-opt` is used if installed.

```sh
./build-web.sh                        # builds the bundle into web/
python3 -m http.server -d web 8080    # then open http://localhost:8080
```

Or run `./build-web-singlefile.sh` for `web/fractality-standalone.html`, one self-contained file that runs from a double-click with no server. Pushes to `master` deploy to GitHub Pages via `.github/workflows/pages.yml`.

In the browser, audio comes from a shared tab or screen (tick "share audio" in the picker), falling back to the microphone.

## How it works

The CPU computes one high-precision reference orbit per view in double-double arithmetic (~31 digits, good to a view height of 1e-28). The compute shader then iterates only the *delta* from that orbit for every particle in `f32`, which keeps deep zooms stable far past where naive single-precision math falls apart. Particle color, motion, and escape behavior all derive from a smooth escape-time field with an analytic gradient (no finite differences), and the audio pipeline feeds beat, bass, and treble energy into the simulation parameters every frame.

Rendering is instanced quads with additive blending, HDR bloom, an optional trail-accumulation pass with beat-driven RGB splitting, and a kaleidoscope fold in the trail composite.
