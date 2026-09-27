# Fractality

**An audio-reactive fractal particle visualizer.** Millions of GPU particles swarm across the Mandelbrot set (and friends), driven by compute shaders and whatever music your machine is playing.

**[▶ Run it in your browser](https://johnnyperkins.github.io/Fractality/)** (needs WebGPU: Chrome, Edge, or recent Firefox)

Built in Rust with [Bevy](https://bevyengine.org) and WGSL compute shaders. Runs native on Linux and on the web via WASM + WebGPU.

## What it does

- **12 million particles** (4M on web), fully simulated on the GPU in a WGSL compute shader
- **5 fractals**: Mandelbrot, Burning Ship, Tricorn, Multibrot-3, and a morphing Julia set
- **Deep zoom** using perturbation theory: a high-precision reference orbit on the CPU, per-particle deltas on the GPU, with rebasing (Zhuoran's method) when the delta outgrows the reference
- **Audio reactivity**: captures system audio (native) or tab/mic audio (web) and drives particle motion, beat-synced chromatic bloom, drop detection, and an "audio aurora" color mode where bass and treble anchor the palette
- **6 flow modes**: contour, layers, gravity, erupt, pulse, dynamics: different ways particles ride the fractal's field
- **Kaleidoscope mode** with adjustable folds and spin
- **Trails, dissolve, auto-zoom dive**, and view bookmarks you can fly between
- **Recording built in**: `P` for a screenshot, `O` to record video (H.264 MP4 on native via openh264, WebM via MediaRecorder on web)

## Controls

| Key | Action |
|-----|--------|
| `W A S D` | Pan |
| Scroll | Zoom |
| `F` | Cycle fractal |
| `G` | Cycle flow mode |
| `C` | Cycle color mode |
| `K` | Toggle kaleidoscope |
| `V` | Toggle audio reactivity |
| `Space` | Toggle dissolve |
| `Z` | Auto-zoom dive |
| `R` | Reset view |
| `Shift+1..9` | Save bookmark |
| `1..9` | Fly to bookmark |
| `P` | Screenshot |
| `O` | Record video |
| `M` / `Esc` | Menu (sliders for everything) |

## Running

### Native (Linux)

```sh
cargo run --release
```

Audio reactivity captures system audio output, so play music anywhere and the fractal hears it.

### Web

```sh
./build-web.sh        # builds wasm + serves locally
```

Or use `./build-web-singlefile.sh` for a single self-contained HTML file. The hosted build deploys automatically to GitHub Pages from `pages.yml`.

## How it works

The CPU computes one high-precision reference orbit per view in double-double arithmetic (~31 digits, good to a view height of 1e-28). The compute shader then iterates only the *delta* from that orbit for every particle in `f32`, which keeps deep zooms stable far past where naive single-precision math falls apart. Particle color, motion, and escape behavior all derive from a smooth escape-time field with an analytic gradient (no finite differences), and the audio pipeline feeds beat, bass, and treble energy into the simulation parameters every frame.

Rendering is instanced quads with additive blending, HDR bloom, an optional trail-accumulation pass with beat-driven RGB splitting, and a kaleidoscope fold in the trail composite.
