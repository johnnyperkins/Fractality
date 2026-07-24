// Audio reactivity: captures audio on a per-platform path (native: default
// sink monitor via `pw-record`/PipeWire; web: tab/system audio via a screen
// share, falling back to the microphone), runs a small FFT per 1024-sample
// window, and publishes normalized band levels, a 16-bin spectrum, and
// beat/drop counters through atomics. The main world smooths these per frame
// into AudioLevels, which update_params folds into the sim uniforms.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

use bevy::prelude::*;

const RATE: u32 = 44100;
const N: usize = 1024;
/// Log-spaced spectrum resolution published to the shaders.
pub const SPECTRUM_BINS: usize = 16;

/// Lock-free channel from the capture callback/thread. Values are f32 bits in
/// AtomicU32 (bands and spectrum already auto-gained to ~0..1);
/// `beats`/`drops` increment per event so the reader can't miss short
/// pulses between frames.
#[derive(Default)]
pub struct AudioShared {
    bass: AtomicU32,
    mid: AtomicU32,
    treble: AtomicU32,
    level: AtomicU32,
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
    capture: Option<Capture>,
}

/// A live capture: the shared level atomics plus whatever platform object
/// keeps the stream alive. Dropping it tears the capture down.
struct Capture {
    shared: Arc<AudioShared>,
    _platform: platform::Handle,
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

/// One FFT window's worth of analysis state, shared by the native capture
/// thread and the web audio callback. Timing is stream time in seconds,
/// accumulated from windows processed (window duration = N / sample_rate),
/// so it needs no clock - std::time::Instant panics on wasm. The old
/// Instant-based code also only checked at window boundaries, so with a
/// real-time source the decisions are identical.
struct WindowProcessor {
    /// Hann window, precomputed.
    window: [f32; N],
    // Band edges as FFT bin indices (bass 30-250 Hz, mid to 2 kHz, treble to
    // 8 kHz), plus the 16 log-spaced spectrum edges over the same span.
    b0: usize,
    b1: usize,
    m1: usize,
    t1: usize,
    edges: [usize; SPECTRUM_BINS + 1],
    peaks: [f32; 4],
    spec_peaks: [f32; SPECTRUM_BINS],
    /// ~1 s of bass-power history for beat detection (energy flux).
    hist: [f32; 43],
    hist_i: usize,
    hist_n: usize,
    /// Short/long energy EMAs for drop detection (a sustained surge well
    /// above the recent norm). ~0.25 s and ~4 s time constants.
    ema_short: f32,
    ema_long: f32,
    /// Seconds of audio consumed; advances by `window_dt` per window.
    t: f64,
    window_dt: f64,
    last_beat: f64,
    last_drop: f64,
}

impl WindowProcessor {
    fn new(rate: f32) -> Self {
        let mut window = [0.0f32; N];
        for (i, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N as f32).cos();
        }
        // Bin width = rate / N (~43 Hz at 44.1 kHz). Band edges in Hz -> bins.
        let bin = |hz: f32| ((hz * N as f32 / rate) as usize).clamp(1, N / 2);
        // Log-spaced 16-bin spectrum edges over the 30 Hz..8 kHz span, made
        // strictly monotonic (the low bins would otherwise collapse onto one
        // FFT bin at this resolution).
        let mut edges = [0usize; SPECTRUM_BINS + 1];
        for (i, e) in edges.iter_mut().enumerate() {
            *e = bin(30.0 * (8000.0f32 / 30.0).powf(i as f32 / SPECTRUM_BINS as f32));
        }
        for i in 1..edges.len() {
            edges[i] = edges[i].max(edges[i - 1] + 1);
        }
        Self {
            window,
            b0: bin(30.0),
            b1: bin(250.0),
            m1: bin(2000.0),
            t1: bin(8000.0),
            edges,
            peaks: [1e-5; 4],
            spec_peaks: [1e-5; SPECTRUM_BINS],
            hist: [0.0; 43],
            hist_i: 0,
            hist_n: 0,
            ema_short: 0.0,
            ema_long: 0.0,
            t: 0.0,
            window_dt: N as f64 / rate as f64,
            // t starts at 0, so beats hold off ~150 ms and drops 8 s after
            // capture start, matching the old Instant::now() initialization.
            last_beat: 0.0,
            last_drop: 0.0,
        }
    }

    /// Analyze one N-sample mono window and publish the results.
    fn process(&mut self, samples: &[f32; N], shared: &AudioShared) {
        // Advance the stream clock first: after window k, t = k * window_dt,
        // which is the wall time the old Instant-based code observed when it
        // checked cooldowns at this same point (real-time source assumed).
        self.t += self.window_dt;
        let mut re = [0.0f32; N];
        let mut im = [0.0f32; N];
        for ((r, s), w) in re.iter_mut().zip(samples).zip(&self.window) {
            *r = s * w;
        }
        fft(&mut re, &mut im);
        // Per-bin power once; every band below is a range sum over this.
        let mut power = [0.0f32; N / 2];
        for (p, (r, i_)) in power.iter_mut().zip(re.iter().zip(im.iter())) {
            *p = r * r + i_ * i_;
        }
        let band_sum = |lo: usize, hi: usize| power[lo..hi].iter().sum::<f32>();

        // The three bands are contiguous, so the overall level is their union.
        let (b0, b1, m1, t1) = (self.b0, self.b1, self.m1, self.t1);
        let bass_s = band_sum(b0, b1);
        let mid_s = band_sum(b1, m1);
        let treble_s = band_sum(m1, t1);
        let bass_p = bass_s / (b1 - b0) as f32;
        let mid_p = mid_s / (m1 - b1) as f32;
        let treble_p = treble_s / (t1 - m1) as f32;
        let level_p = (bass_s + mid_s + treble_s) / (t1 - b0) as f32;

        store_f32(&shared.bass, autogain(&mut self.peaks[0], bass_p.sqrt()));
        store_f32(&shared.mid, autogain(&mut self.peaks[1], mid_p.sqrt()));
        store_f32(&shared.treble, autogain(&mut self.peaks[2], treble_p.sqrt()));
        store_f32(&shared.level, autogain(&mut self.peaks[3], level_p.sqrt()));

        // Per-bin spectrum, each bin auto-gained independently so quiet
        // frequency regions still register visually.
        for i in 0..SPECTRUM_BINS {
            let (lo, hi) = (self.edges[i], self.edges[i + 1]);
            let p = band_sum(lo, hi) / (hi - lo) as f32;
            store_f32(&shared.spectrum[i], autogain(&mut self.spec_peaks[i], p.sqrt()));
        }

        // Beat: bass power spikes well above its recent average.
        if self.hist_n >= 12 {
            let mean = self.hist[..self.hist_n].iter().sum::<f32>() / self.hist_n as f32;
            if bass_p > mean * 1.6 + 1e-7 && self.t - self.last_beat > 0.15 {
                shared.beats.fetch_add(1, Ordering::Relaxed);
                self.last_beat = self.t;
            }
        }
        self.hist[self.hist_i] = bass_p;
        self.hist_i = (self.hist_i + 1) % self.hist.len();
        self.hist_n = (self.hist_n + 1).min(self.hist.len());

        // Drop: short-term energy surges over the long-term norm (build-up
        // then hit). Warm-up and a long cooldown keep it a rare event.
        self.ema_short += (level_p - self.ema_short) * 0.093;
        self.ema_long += (level_p - self.ema_long) * 0.0058;
        // ~2 s warm-up (86 windows at 44.1 kHz, matching the original gate).
        if self.t > 2.0
            && self.ema_long > 1e-7
            && self.ema_short > self.ema_long * 3.0
            && self.t - self.last_drop > 8.0
        {
            shared.drops.fetch_add(1, Ordering::Relaxed);
            self.last_drop = self.t;
        }
    }
}

/// Native capture: spawn pw-record on the default sink's monitor (whatever
/// the PC is playing) and analyze its stdout on a background thread.
#[cfg(not(target_arch = "wasm32"))]
mod platform {
    use std::io::Read;
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use super::{AudioShared, WindowProcessor, N, RATE};

    pub const DESCRIPTION: &str = "capturing system output";

    /// Owns the pw-record child; dropping it stops the thread and reaps the
    /// process off-thread (wait() on the main thread would block the frame).
    pub struct Handle {
        child: Option<Child>,
        shared: Arc<AudioShared>,
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            self.shared.running.store(false, Ordering::Relaxed);
            if let Some(mut child) = self.child.take() {
                let _ = child.kill();
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
            }
        }
    }

    /// Capture loop: reads f32le mono frames from pw-record's stdout, one FFT
    /// window per read (~23 ms). Exits when `running` is cleared or the pipe
    /// closes (pw-record died / no audio server).
    fn capture_loop(mut stdout: impl Read, shared: Arc<AudioShared>) {
        let mut proc = WindowProcessor::new(RATE as f32);
        let mut bytes = vec![0u8; N * 4];
        let mut samples = [0.0f32; N];
        while shared.running.load(Ordering::Relaxed) {
            if stdout.read_exact(&mut bytes).is_err() {
                break;
            }
            for (s, chunk) in samples.iter_mut().zip(bytes.chunks_exact(4)) {
                *s = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            proc.process(&samples, &shared);
        }
        shared.running.store(false, Ordering::Relaxed);
    }

    pub fn start_capture(shared: Arc<AudioShared>) -> std::io::Result<Handle> {
        let mut child = Command::new("pw-record")
            .args(["--properties", "{ stream.capture.sink=true }"])
            .args(["--format", "f32", "--channels", "1", "--rate"])
            .arg(RATE.to_string())
            .arg("-")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped stdout");
        let thread_shared = shared.clone();
        std::thread::spawn(move || capture_loop(stdout, thread_shared));
        Ok(Handle {
            child: Some(child),
            shared,
        })
    }
}

/// Web capture: getDisplayMedia (user picks a tab / screen and shares its
/// audio; on Chrome+Windows "share system audio" covers everything), falling
/// back to getUserMedia (microphone). A ScriptProcessorNode hands us mono
/// N-sample buffers on the main thread - wasm here is single-threaded, so the
/// "capture thread" is just a JS callback feeding the same WindowProcessor.
#[cfg(target_arch = "wasm32")]
mod platform {
    use std::cell::{Cell, RefCell};
    use std::io;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use bevy::prelude::warn;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::js_sys;
    use web_sys::{
        AudioContext, AudioContextOptions, AudioProcessingEvent, MediaStream, MediaStreamTrack,
        ScriptProcessorNode,
    };

    use super::{AudioShared, WindowProcessor, N, RATE};

    pub const DESCRIPTION: &str = "pick a tab/screen and tick 'share audio' (mic fallback)";

    // The JS audio graph lives in a thread-local, not in the Handle: the
    // Handle sits inside a Bevy Resource, which must be Send+Sync, and JS
    // objects are neither. Wasm without atomics is single-threaded, so the
    // Handle's Drop and the async setup both run on this one thread.
    // `ACTIVE` holds the id of the capture that owns the slot; a setup that
    // finishes after its Handle died (or after a newer capture started) sees
    // the mismatch and tears itself down instead of leaking a live stream.
    thread_local! {
        static GRAPH: RefCell<Option<Graph>> = const { RefCell::new(None) };
        static ACTIVE: Cell<u64> = const { Cell::new(0) };
    }
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

    /// Send+Sync token for one capture attempt; dropping it tears down the
    /// audio graph if this attempt still owns it.
    pub struct Handle(u64);

    struct Graph {
        ctx: AudioContext,
        stream: MediaStream,
        processor: ScriptProcessorNode,
        _on_audio: Closure<dyn FnMut(AudioProcessingEvent)>,
        _on_ended: Closure<dyn FnMut()>,
    }

    fn stop_tracks(tracks: js_sys::Array) {
        for t in tracks.iter() {
            t.unchecked_into::<MediaStreamTrack>().stop();
        }
    }

    fn teardown(g: Graph) {
        g.processor.set_onaudioprocess(None);
        let _ = g.ctx.close();
        stop_tracks(g.stream.get_tracks());
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            if ACTIVE.get() == self.0 {
                ACTIVE.set(0);
                if let Some(g) = GRAPH.with_borrow_mut(Option::take) {
                    teardown(g);
                }
            }
        }
    }

    /// Returns immediately; the permission prompt and graph construction run
    /// async. Failures surface by clearing `shared.running`, which the next
    /// manage_capture pass treats as "capture stopped".
    pub fn start_capture(shared: Arc<AudioShared>) -> io::Result<Handle> {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        ACTIVE.set(id);
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(e) = setup(shared.clone(), id).await {
                warn!("web audio capture failed: {e:?}");
                shared.running.store(false, Ordering::Relaxed);
            }
        });
        Ok(Handle(id))
    }

    async fn setup(shared: Arc<AudioShared>, id: u64) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let devices = window.navigator().media_devices()?;

        // Tab/screen audio first (video must be requested; we stop its track
        // right away). Denied/unsupported -> microphone.
        let display = web_sys::DisplayMediaStreamConstraints::new();
        display.set_audio(&JsValue::TRUE);
        display.set_video(&JsValue::TRUE);
        let stream: MediaStream = match JsFuture::from(
            devices.get_display_media_with_constraints(&display)?,
        )
        .await
        {
            Ok(s) => s.unchecked_into(),
            Err(_) => {
                let mic = web_sys::MediaStreamConstraints::new();
                mic.set_audio(&JsValue::TRUE);
                JsFuture::from(devices.get_user_media_with_constraints(&mic)?)
                    .await?
                    .unchecked_into()
            }
        };
        stop_tracks(stream.get_video_tracks());
        if stream.get_audio_tracks().length() == 0 {
            stop_tracks(stream.get_tracks());
            return Err("no audio track shared (tick 'share tab audio' in the picker)".into());
        }

        // Ask for the native analysis rate; if the browser resamples to
        // something else anyway, ctx.sample_rate() reports it and the
        // processor's band math follows.
        let opts = AudioContextOptions::new();
        opts.set_sample_rate(RATE as f32);
        let ctx = AudioContext::new_with_context_options(&opts)?;
        // The V keypress's user activation may not survive the await chain;
        // resume() makes autoplay-suspended contexts start anyway.
        let _ = ctx.resume();

        let source = ctx.create_media_stream_source(&stream)?;
        let processor = ctx
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                N as u32, 1, 1,
            )?;

        let mut proc = WindowProcessor::new(ctx.sample_rate());
        let mut samples = [0.0f32; N];
        let cb_shared = shared.clone();
        let on_audio = Closure::<dyn FnMut(AudioProcessingEvent)>::new(
            move |ev: AudioProcessingEvent| {
                if !cb_shared.running.load(Ordering::Relaxed) {
                    return;
                }
                if let Ok(buf) = ev.input_buffer() {
                    if buf.copy_from_channel(&mut samples, 0).is_ok() {
                        proc.process(&samples, &cb_shared);
                    }
                }
            },
        );
        processor.set_onaudioprocess(Some(on_audio.as_ref().unchecked_ref()));
        source.connect_with_audio_node(&processor)?;
        // A ScriptProcessorNode only fires while wired to the destination.
        // Its output stays silent (we never write the output buffer), so the
        // captured tab is not echoed.
        processor.connect_with_audio_node(&ctx.destination())?;

        // Browser "stop sharing" bar ends the track; treat it as capture loss.
        let ended_shared = shared.clone();
        let on_ended = Closure::<dyn FnMut()>::new(move || {
            ended_shared.running.store(false, Ordering::Relaxed);
        });
        let track: MediaStreamTrack = stream.get_audio_tracks().get(0).unchecked_into();
        track.set_onended(Some(on_ended.as_ref().unchecked_ref()));

        let graph = Graph {
            ctx,
            stream,
            processor,
            _on_audio: on_audio,
            _on_ended: on_ended,
        };
        // Our Handle died (or was replaced) while the permission prompt was
        // up: don't leak a live capture nobody owns.
        if ACTIVE.get() != id {
            teardown(graph);
            return Ok(());
        }
        GRAPH.with_borrow_mut(|slot| *slot = Some(graph));
        Ok(())
    }
}

/// Start/stop the capture to match the enabled flag, and disable if the
/// capture died (native: no PipeWire / pw-record missing / pipe closed; web:
/// permission denied or sharing stopped). Steady states only read (Deref),
/// so the resource is marked changed only on real transitions - UI systems
/// rely on that.
pub fn manage_capture(mut audio: ResMut<AudioCapture>) {
    if audio.enabled && audio.capture.is_none() {
        let shared = Arc::new(AudioShared {
            running: AtomicBool::new(true),
            ..Default::default()
        });
        match platform::start_capture(shared.clone()) {
            Ok(handle) => {
                info!("audio reactivity ON ({})", platform::DESCRIPTION);
                audio.capture = Some(Capture {
                    shared,
                    _platform: handle,
                });
            }
            Err(e) => {
                warn!("audio capture failed to start: {e}");
                audio.enabled = false;
            }
        }
    } else if audio.enabled
        && audio
            .capture
            .as_ref()
            .is_some_and(|c| !c.shared.running.load(Ordering::Relaxed))
    {
        warn!("audio capture stopped; disabling audio reactivity");
        audio.enabled = false;
    }
    if !audio.enabled && audio.capture.is_some() {
        audio.capture = None;
        info!("audio reactivity OFF");
    }
}

/// Per-frame smoothing of the raw capture levels into AudioLevels: fast
/// attack, slower release (punchy but not strobing), event ages for the ring
/// waves, palette hue and Julia morph accumulation.
pub fn update_audio(
    time: Res<Time>,
    capture: Res<AudioCapture>,
    mut levels: ResMut<AudioLevels>,
) {
    let dt = time.delta_secs();
    let shared = capture.capture.as_ref().filter(|_| capture.enabled).map(|c| &c.shared);
    let raw = shared.map(|s| s.snapshot()).unwrap_or_default();
    // A single non-finite value here poisons brightness/size uniforms and
    // blacks the whole frame; scrub before it enters the smoothing.
    let clean = |v: f32| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
    let env = |cur: f32, target: f32| {
        let k = if target > cur { 30.0 } else { 5.0 };
        snap(cur + (clean(target) - cur) * (k * dt).min(1.0))
    };

    let l = &mut *levels;
    l.bass = env(l.bass, raw.bass);
    l.mid = env(l.mid, raw.mid);
    l.treble = env(l.treble, raw.treble);
    l.level = env(l.level, raw.level);
    for i in 0..SPECTRUM_BINS {
        l.spectrum[i] = env(l.spectrum[i], raw.spectrum[i]);
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
