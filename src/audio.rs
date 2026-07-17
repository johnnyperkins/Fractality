// Audio reactivity: captures whatever the PC is playing (default sink
// monitor via `pw-record`, PipeWire) on a background thread, runs a small
// FFT, and publishes normalized band levels, a 16-bin spectrum, stereo pan,
// and beat/drop counters through atomics. The main world smooths these per
// frame into AudioLevels, which update_params folds into the sim uniforms.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::prelude::*;

const RATE: u32 = 44100;
const N: usize = 1024;
/// Log-spaced spectrum resolution published to the shaders.
pub const SPECTRUM_BINS: usize = 16;

/// Lock-free channel from the capture thread. Values are f32 bits in
/// AtomicU32 (bands and spectrum already auto-gained to ~0..1, pan in
/// -1..1); `beats`/`drops` increment per event so the reader can't miss
/// short pulses between frames.
#[derive(Default)]
pub struct AudioShared {
    bass: AtomicU32,
    mid: AtomicU32,
    treble: AtomicU32,
    level: AtomicU32,
    pan: AtomicU32,
    spectrum: [AtomicU32; SPECTRUM_BINS],
    beats: AtomicU32,
    drops: AtomicU32,
    running: AtomicBool,
}

/// Plain-value snapshot of AudioShared, one atomic load per field.
#[derive(Default)]
struct RawLevels {
    bass: f32,
    mid: f32,
    treble: f32,
    level: f32,
    pan: f32,
    spectrum: [f32; SPECTRUM_BINS],
    beats: u32,
    drops: u32,
}

impl AudioShared {
    fn snapshot(&self) -> RawLevels {
        RawLevels {
            bass: load_f32(&self.bass),
            mid: load_f32(&self.mid),
            treble: load_f32(&self.treble),
            level: load_f32(&self.level),
            pan: load_f32(&self.pan),
            spectrum: std::array::from_fn(|i| load_f32(&self.spectrum[i])),
            beats: self.beats.load(Ordering::Relaxed),
            drops: self.drops.load(Ordering::Relaxed),
        }
    }
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Capture lifecycle, toggled with V (or the menu row). Changes only on real
/// transitions (toggle, capture start/stop), so UI systems can gate on
/// is_changed; the per-frame signal lives in AudioLevels.
#[derive(Resource, Default)]
pub struct AudioCapture {
    pub enabled: bool,
    capture: Option<(Child, Arc<AudioShared>)>,
}

/// Smoothed audio levels the sim reads each frame. All zero (decayed) while
/// capture is disabled, so consumers need no enable branch. Written every
/// frame; nothing should gate on change detection for this resource.
#[derive(Resource)]
pub struct AudioLevels {
    pub bass: f32,
    pub mid: f32,
    pub treble: f32,
    pub level: f32,
    /// Stereo balance of the source, -1 (left) .. 1 (right).
    pub pan: f32,
    /// Log-spaced per-band energies, bin 0 = lowest frequencies.
    pub spectrum: [f32; SPECTRUM_BINS],
    /// Beat pulse: jumps to 1 on a detected beat (or drop), exponential decay.
    pub beat: f32,
    /// Drop envelope: 1 at the drop, easing out over ~2 s. The canonical
    /// "how long a drop feels" for scalar consumers; the shader ring
    /// geometry uses drop_age directly because it needs the wavefront radius.
    pub drop: f32,
    /// Seconds since the last beat / drop; drives the expanding ring waves.
    /// Starts saturated so no phantom wave fires at startup.
    pub beat_age: f32,
    pub drop_age: f32,
    /// Accumulated hue offset; music energy spins the palette, a drop kicks
    /// it by the golden angle.
    pub hue_phase: f32,
    /// Accumulated phase for the Julia parameter orbit. Frozen while capture
    /// is off so the Julia reference orbit stays cache-stable.
    pub morph_phase: f32,
    last_beats: u32,
    last_drops: u32,
}

impl Default for AudioLevels {
    fn default() -> Self {
        Self {
            bass: 0.0,
            mid: 0.0,
            treble: 0.0,
            level: 0.0,
            pan: 0.0,
            spectrum: [0.0; SPECTRUM_BINS],
            beat: 0.0,
            drop: 0.0,
            beat_age: 1e3,
            drop_age: 1e3,
            hue_phase: 0.0,
            morph_phase: 0.0,
            last_beats: 0,
            last_drops: 0,
        }
    }
}

/// Values below this settle to exactly 0 (see `snap`), so equality with zero
/// is the one idle definition shared by the smoothing and consumers.
const SNAP_EPS: f32 = 1e-3;

/// Snap near-zero to exact zero so idle gating on == 0.0 settles.
fn snap(v: f32) -> f32 {
    if v < SNAP_EPS {
        0.0
    } else {
        v
    }
}

impl AudioLevels {
    /// Idle: every published level has decayed to exact zero, so recomputing
    /// any audio-modulated value cannot change it, whatever fields the
    /// consumer uses.
    pub fn is_idle(&self) -> bool {
        self.bass == 0.0
            && self.mid == 0.0
            && self.treble == 0.0
            && self.level == 0.0
            && self.pan == 0.0
            && self.beat == 0.0
            && self.drop == 0.0
            && self.spectrum.iter().all(|&s| s == 0.0)
    }
}

/// In-place iterative radix-2 FFT, N a power of two. Plenty for a 1024-point
/// spectrum at ~43 fps; not worth a crate dependency.
fn fft(re: &mut [f32; N], im: &mut [f32; N]) {
    // Bit-reversal permutation.
    let bits = N.trailing_zeros();
    for i in 0..N {
        let j = (i as u32).reverse_bits() >> (32 - bits);
        let j = j as usize;
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= N {
        let ang = -2.0 * std::f32::consts::PI / len as f32;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < N {
            let (mut cr, mut ci) = (1.0f32, 0.0f32);
            for k in 0..len / 2 {
                let (ar, ai) = (re[i + k], im[i + k]);
                let (br, bi) = (re[i + k + len / 2], im[i + k + len / 2]);
                let (tr, ti) = (br * cr - bi * ci, br * ci + bi * cr);
                re[i + k] = ar + tr;
                im[i + k] = ai + ti;
                re[i + k + len / 2] = ar - tr;
                im[i + k + len / 2] = ai - ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Slow-decay peak tracker: returns amp normalized to its own recent peak,
/// so quiet and loud sources both land in a useful 0..1 range without a
/// sensitivity knob.
fn autogain(peak: &mut f32, amp: f32) -> f32 {
    *peak = (*peak * 0.9990).max(amp).max(1e-5);
    (amp / *peak).clamp(0.0, 1.0)
}

/// Capture loop: reads f32le stereo frames from pw-record's stdout, one FFT
/// window per read (~23 ms). Exits when `running` is cleared or the pipe
/// closes (pw-record died / no audio server).
fn capture_loop(mut stdout: impl Read, shared: Arc<AudioShared>) {
    // Hann window, precomputed.
    let mut window = [0.0f32; N];
    for (i, w) in window.iter_mut().enumerate() {
        *w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos();
    }
    // Bin width = RATE / N (~43 Hz). Band edges in Hz -> bins.
    let bin = |hz: f32| ((hz * N as f32 / RATE as f32) as usize).clamp(1, N / 2);
    let (b0, b1) = (bin(30.0), bin(250.0));
    let (m1, t1) = (bin(2000.0), bin(8000.0));
    // Log-spaced 16-bin spectrum edges over the same 30 Hz..8 kHz span, made
    // strictly monotonic (the low bins would otherwise collapse onto one FFT
    // bin at this resolution).
    let mut edges = [0usize; SPECTRUM_BINS + 1];
    for (i, e) in edges.iter_mut().enumerate() {
        *e = bin(30.0 * (8000.0f32 / 30.0).powf(i as f32 / SPECTRUM_BINS as f32));
    }
    for i in 1..edges.len() {
        edges[i] = edges[i].max(edges[i - 1] + 1);
    }

    let mut peaks = [1e-5f32; 4];
    let mut spec_peaks = [1e-5f32; SPECTRUM_BINS];
    // Pan gets auto-gain too: raw L/R imbalance is tiny (~0.05) on typical
    // mostly-centered mixes, so normalize to its own recent peak like the
    // bands. The floor keeps near-mono content from amplifying noise.
    let mut pan_peak = 0.05f32;
    // ~1 s of bass-power history for beat detection (energy flux).
    let mut hist = [0.0f32; 43];
    let mut hist_i = 0usize;
    let mut hist_n = 0usize;
    let mut last_beat = Instant::now();
    // Short/long energy EMAs for drop detection (a sustained surge well above
    // the recent norm). ~0.25 s and ~4 s time constants at 43 windows/s.
    let mut ema_short = 0.0f32;
    let mut ema_long = 0.0f32;
    let mut windows_seen = 0u32;
    let mut last_drop = Instant::now();

    let mut bytes = vec![0u8; N * 8]; // stereo interleaved f32
    let mut re = [0.0f32; N];
    let mut im = [0.0f32; N];
    let mut power = [0.0f32; N / 2];
    while shared.running.load(Ordering::Relaxed) {
        if stdout.read_exact(&mut bytes).is_err() {
            break;
        }
        im.fill(0.0);
        let mut l_pow = 0.0f32;
        let mut r_pow = 0.0f32;
        for ((r, frame), w) in re.iter_mut().zip(bytes.chunks_exact(8)).zip(&window) {
            let l = f32::from_le_bytes(frame[0..4].try_into().unwrap());
            let rt = f32::from_le_bytes(frame[4..8].try_into().unwrap());
            l_pow += l * l;
            r_pow += rt * rt;
            *r = (l + rt) * 0.5 * w;
        }
        fft(&mut re, &mut im);
        // Per-bin power once; every band below is a range sum over this.
        for (p, (r, i_)) in power.iter_mut().zip(re.iter().zip(im.iter())) {
            *p = r * r + i_ * i_;
        }
        let band_sum = |lo: usize, hi: usize| power[lo..hi].iter().sum::<f32>();

        // Stereo pan from raw channel RMS, normalized to its recent width.
        let (l_rms, r_rms) = (l_pow.sqrt(), r_pow.sqrt());
        let pan = ((r_rms - l_rms) / (l_rms + r_rms + 1e-6)).clamp(-1.0, 1.0);
        pan_peak = pan_peak.max(0.05); // re-assert the noise floor each window
        let pan_n = pan.signum() * autogain(&mut pan_peak, pan.abs());
        store_f32(&shared.pan, pan_n);

        // The three bands are contiguous, so the overall level is their union.
        let bass_s = band_sum(b0, b1);
        let mid_s = band_sum(b1, m1);
        let treble_s = band_sum(m1, t1);
        let bass_p = bass_s / (b1 - b0) as f32;
        let mid_p = mid_s / (m1 - b1) as f32;
        let treble_p = treble_s / (t1 - m1) as f32;
        let level_p = (bass_s + mid_s + treble_s) / (t1 - b0) as f32;

        store_f32(&shared.bass, autogain(&mut peaks[0], bass_p.sqrt()));
        store_f32(&shared.mid, autogain(&mut peaks[1], mid_p.sqrt()));
        store_f32(&shared.treble, autogain(&mut peaks[2], treble_p.sqrt()));
        store_f32(&shared.level, autogain(&mut peaks[3], level_p.sqrt()));

        // Per-bin spectrum, each bin auto-gained independently so quiet
        // frequency regions still register visually.
        for i in 0..SPECTRUM_BINS {
            let p = band_sum(edges[i], edges[i + 1]) / (edges[i + 1] - edges[i]) as f32;
            store_f32(&shared.spectrum[i], autogain(&mut spec_peaks[i], p.sqrt()));
        }

        // Beat: bass power spikes well above its recent average.
        if hist_n >= 12 {
            let mean = hist[..hist_n].iter().sum::<f32>() / hist_n as f32;
            if bass_p > mean * 1.6 + 1e-7
                && last_beat.elapsed() > Duration::from_millis(150)
            {
                shared.beats.fetch_add(1, Ordering::Relaxed);
                last_beat = Instant::now();
            }
        }
        hist[hist_i] = bass_p;
        hist_i = (hist_i + 1) % hist.len();
        hist_n = (hist_n + 1).min(hist.len());

        // Drop: short-term energy surges over the long-term norm (build-up
        // then hit). Warm-up and a long cooldown keep it a rare event.
        ema_short += (level_p - ema_short) * 0.093;
        ema_long += (level_p - ema_long) * 0.0058;
        windows_seen += 1;
        if windows_seen > 86
            && ema_long > 1e-7
            && ema_short > ema_long * 3.0
            && last_drop.elapsed() > Duration::from_secs(8)
        {
            shared.drops.fetch_add(1, Ordering::Relaxed);
            last_drop = Instant::now();
        }
    }
    shared.running.store(false, Ordering::Relaxed);
}

/// Spawn pw-record capturing the default sink's monitor (i.e. whatever the
/// PC is playing), f32le stereo RATE Hz to stdout.
fn start_capture() -> std::io::Result<(Child, Arc<AudioShared>)> {
    let mut child = Command::new("pw-record")
        .args(["--properties", "{ stream.capture.sink=true }"])
        .args(["--format", "f32", "--channels", "2", "--rate"])
        .arg(RATE.to_string())
        .arg("-")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let stdout = child.stdout.take().expect("piped stdout");
    let shared = Arc::new(AudioShared {
        running: AtomicBool::new(true),
        ..Default::default()
    });
    let thread_shared = shared.clone();
    std::thread::spawn(move || capture_loop(stdout, thread_shared));
    Ok((child, shared))
}

/// Start/stop the capture process to match the enabled flag, and disable if
/// the capture thread died (no PipeWire, pw-record missing, pipe closed).
/// Steady states only read (Deref), so the resource is marked changed only
/// on real transitions - UI systems rely on that.
pub fn manage_capture(mut audio: ResMut<AudioCapture>) {
    if audio.enabled && audio.capture.is_none() {
        match start_capture() {
            Ok(capture) => {
                info!("audio reactivity ON (capturing system output)");
                audio.capture = Some(capture);
            }
            Err(e) => {
                warn!("audio capture failed to start (pw-record): {e}");
                audio.enabled = false;
            }
        }
    } else if audio.enabled
        && audio
            .capture
            .as_ref()
            .is_some_and(|(_, s)| !s.running.load(Ordering::Relaxed))
    {
        warn!("audio capture stopped (pipe closed); disabling audio reactivity");
        audio.enabled = false;
    }
    if !audio.enabled && audio.capture.is_some() {
        if let Some((mut child, shared)) = audio.capture.take() {
            shared.running.store(false, Ordering::Relaxed);
            let _ = child.kill();
            // Reap off-thread: wait() on the main thread would block the frame.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            info!("audio reactivity OFF");
        }
    }
}

/// Per-frame smoothing of the raw thread levels into AudioLevels: fast
/// attack, slower release (punchy but not strobing), event ages for the ring
/// waves, palette hue and Julia morph accumulation.
pub fn update_audio(
    time: Res<Time>,
    capture: Res<AudioCapture>,
    mut levels: ResMut<AudioLevels>,
) {
    let dt = time.delta_secs();
    let shared = capture.capture.as_ref().filter(|_| capture.enabled).map(|(_, s)| s);
    let raw = shared.map(|s| s.snapshot()).unwrap_or_default();
    // A single non-finite value here poisons brightness/size uniforms and
    // blacks the whole frame; scrub before it enters the smoothing.
    let clean = |v: f32, lo: f32| if v.is_finite() { v.clamp(lo, 1.0) } else { 0.0 };
    let env = |cur: f32, target: f32| {
        let k = if target > cur { 30.0 } else { 5.0 };
        snap(cur + (clean(target, 0.0) - cur) * (k * dt).min(1.0))
    };

    let l = &mut *levels;
    l.bass = env(l.bass, raw.bass);
    l.mid = env(l.mid, raw.mid);
    l.treble = env(l.treble, raw.treble);
    l.level = env(l.level, raw.level);
    for i in 0..SPECTRUM_BINS {
        l.spectrum[i] = env(l.spectrum[i], raw.spectrum[i]);
    }
    // Pan is signed; symmetric smoothing, snapped so idle settles at 0.
    l.pan += (clean(raw.pan, -1.0) - l.pan) * (10.0 * dt).min(1.0);
    if l.pan.abs() < SNAP_EPS {
        l.pan = 0.0;
    }

    // Event pulses and ages. Decay first; an event this frame overwrites.
    l.beat_age = (l.beat_age + dt).min(1e3);
    l.drop_age = (l.drop_age + dt).min(1e3);
    l.beat = snap(l.beat * (-6.0 * dt).exp());
    if shared.is_some() {
        if raw.beats != l.last_beats {
            l.last_beats = raw.beats;
            l.beat = 1.0;
            l.beat_age = 0.0;
        }
        if raw.drops != l.last_drops {
            l.last_drops = raw.drops;
            l.beat = 1.0;
            l.drop_age = 0.0;
            // Golden-angle palette kick so a drop lands on a fresh color.
            l.hue_phase = (l.hue_phase + 0.381_966).fract();
        }
        // Julia parameter orbit phase: idles slowly, races with the mids.
        // Advanced only while capturing, so with audio off the Julia c is
        // constant and the reference orbit cache never churns.
        l.morph_phase = (l.morph_phase + (0.03 + 0.25 * l.mid) * dt).fract();
    } else {
        // Capture off: forget the counters so a fresh capture (which
        // restarts at 0) does not fire phantom events on enable.
        l.last_beats = 0;
        l.last_drops = 0;
    }
    // Canonical scalar drop envelope, derived from the age (exact 0 once
    // the age saturates, so it participates in is_idle).
    l.drop = snap((-l.drop_age * 1.2).exp());
    // Music energy spins the palette; silence leaves it still.
    l.hue_phase = (l.hue_phase + (0.05 * l.mid + 0.1 * l.beat) * dt).fract();
}
