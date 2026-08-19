// Video recording, toggled with O. Two per-platform paths behind one
// interface, same shape as the audio module: `platform::Handle` is the live
// recording, dropping it stops the take.
//  - native: per-frame GPU screenshot readback, encoded in-process to an
//    H.264 mp4 by the bundled openh264 encoder (compiled in, ~1 MB of
//    binary). No external tools, works on any machine.
//  - web: MediaRecorder on the canvas's captureStream. The browser encodes
//    webm off-thread; stopping triggers a download, like the P screenshot.
// No audio track: the visualizer reacts to system audio it does not own, and
// capturing that is the OS screen recorder's job.

use bevy::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use native as platform;
#[cfg(target_arch = "wasm32")]
use web as platform;

pub struct RecorderPlugin;

impl Plugin for RecorderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Recorder>()
            .init_resource::<RecordSettings>()
            .add_systems(Update, toggle);
        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(Update, native::capture_frames.after(toggle));
    }
}

/// Menu option lists for the Recording card (native path only; the web
/// recorder is the browser's MediaRecorder and takes what it gets).
pub const REC_RES_MODES: [&str; 2] = ["Native", "Half"];
pub const REC_FPS_MODES: [&str; 4] = ["24", "30", "60", "120"];
const REC_FPS_VALUES: [f64; 4] = [24.0, 30.0, 60.0, 120.0];

/// Capture options for the next take, set from the menu's Recording card.
/// Read once when a take starts; a running take keeps what it started with.
/// Half res is the escape hatch for big displays: a 4K60 take encodes ~4x
/// slower than 1080p60 and judders when the encoder cannot keep up.
#[derive(Resource)]
pub struct RecordSettings {
    /// Index into REC_RES_MODES: 0 = native window pixels, 1 = half.
    pub res: u32,
    /// Index into REC_FPS_MODES.
    pub fps: u32,
}

pub const REC_RES_DEFAULT: u32 = 0;
pub const REC_FPS_DEFAULT: u32 = 2; // 60 fps

impl Default for RecordSettings {
    fn default() -> Self {
        Self {
            res: REC_RES_DEFAULT,
            fps: REC_FPS_DEFAULT,
        }
    }
}

/// Recording state: the platform handle while recording, None otherwise.
#[derive(Resource, Default)]
pub struct Recorder {
    active: Option<platform::Handle>,
}

impl Recorder {
    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }
}

fn toggle(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    windows: Query<&Window>,
    settings: Res<RecordSettings>,
    mut rec: ResMut<Recorder>,
) {
    if !keys.just_pressed(KeyCode::KeyO) {
        return;
    }
    // Dropping the handle stops the recording on both platforms.
    if rec.active.take().is_some() {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let stamp = crate::output_stamp(&time);
    let path = format!("fractality_{stamp}.{}", platform::EXT);
    rec.active = platform::start(window, path, &settings);
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::fs::File;
    use std::io::BufWriter;
    use std::sync::mpsc::{Receiver, SyncSender};
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use bevy::prelude::*;
    use bevy::render::render_resource::TextureFormat;
    use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
    use mp4::{
        AvcConfig, Bytes, MediaConfig, Mp4Config, Mp4Sample, Mp4Writer, TrackConfig, TrackType,
    };
    use openh264::encoder::{
        BitRate, Complexity, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod,
        QpRange, UsageType, VuiConfig,
    };
    use openh264::formats::YUVSource;
    use openh264::OpenH264API;

    use super::{RecordSettings, Recorder, REC_FPS_VALUES};

    pub(super) const EXT: &str = "mp4";

    /// High because the content is a full-screen particle field in constant
    /// motion, near worst-case for an encoder. With the QP range clamped
    /// below, rate control treats this as a soft ceiling: quality never
    /// drops past QP_MAX to hit it. In practice the encoder stays content
    /// limited under this (~57 Mbps at 1080p, ~70 at 4K on the harness
    /// field). Local disk is the only consumer, so file size loses every
    /// tradeoff against quality.
    const BITRATE_BPS: u32 = 100_000_000;

    /// Rate-control quantizer clamp. The max is the quality floor: without
    /// it, openh264 runs to QP 51 on motion spikes and the whole frame turns
    /// to mush. The min just stops calm scenes from wasting bitrate on
    /// invisible gains.
    const QP_MIN: u8 = 12;
    const QP_MAX: u8 = 36;

    /// Keyframe every 2s so players can seek without decoding from the top.
    const IDR_INTERVAL_SECS: f64 = 2.0;

    /// Slice size cap, which is what actually unlocks openh264's threading:
    /// with the default single-slice mode the thread count is ignored and one
    /// core encodes everything (~30 fps at 1080p, measured by the
    /// quality_harness test below), falling behind the 60 fps capture and
    /// juddering the output. Multi-slice measured 3-4x faster with slightly
    /// better PSNR. The size is load-bearing: in screen-content mode at
    /// 1080p, 30k or 8k slices both collapsed back to ~40 enc fps while 15k
    /// held ~117, and 4K was indifferent, so 15k it is.
    const MAX_SLICE_LEN: u32 = 15_000;

    /// A live recording. Dropping it closes the channel, which ends the
    /// encoder loop and finalizes the mp4. The join is synchronous: the brief
    /// hitch at stop time is the price of the moov index reliably reaching
    /// disk even when the drop is the app quitting (a detached joiner would
    /// be killed at process exit, truncating the file mid-finalize).
    pub(super) struct Handle {
        /// The channel's only sender, in a slot shared with the in-flight
        /// screenshot observers. Observers must not own sender clones: a
        /// readback still pending at stop time would hold the channel open
        /// while the join below blocks the schedule that would deliver it,
        /// deadlocking the app.
        tx: Arc<Mutex<Option<SyncSender<Frame>>>>,
        join: Option<std::thread::JoinHandle<()>>,
        start: Instant,
        size: UVec2,
        /// Output timeline rate for this take. The app renders vsync-off at
        /// whatever rate the GPU manages, so capture skips frames rendered
        /// faster than this and the encoder stretches sample durations over
        /// frames rendered slower, keeping playback in real time.
        fps: f64,
        /// Timeline ticks already covered by a spawned capture, for pacing.
        captured: u64,
    }

    impl Drop for Handle {
        fn drop(&mut self) {
            info!("recording stopped, finalizing");
            *self.tx.lock().unwrap() = None;
            if let Some(join) = self.join.take() {
                let _ = join.join();
            }
        }
    }

    /// One captured frame, tightly packed (Bevy's screenshot readback strips
    /// the 256-byte row padding). Format rides along because the swapchain
    /// format (BGRA vs RGBA) is only known once the first readback lands.
    struct Frame {
        t: f64,
        w: u32,
        h: u32,
        format: TextureFormat,
        data: Vec<u8>,
    }

    pub(super) fn start(
        window: &Window,
        path: String,
        settings: &RecordSettings,
    ) -> Option<Handle> {
        // Create the file up front so a read-only working dir fails the
        // toggle instead of surfacing minutes later at stop time.
        let file = match File::create(&path) {
            Ok(f) => f,
            Err(e) => {
                warn!("recording: cannot create {path}: {e}");
                return None;
            }
        };
        let fps = REC_FPS_VALUES[(settings.fps as usize).min(REC_FPS_VALUES.len() - 1)];
        let scale = 1usize << settings.res.min(1);
        // Small bound: at 1080p a frame is ~8 MB, and a deep queue only adds
        // latency. When the encoder falls behind, try_send drops frames and
        // the timeline pacing keeps playback speed correct anyway.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(4);
        let thread_path = path.clone();
        let join = std::thread::spawn(move || encode_loop(rx, file, thread_path, fps, scale));
        let size = UVec2::new(window.physical_width(), window.physical_height());
        info!(
            "recording to {path} at {}x{} {fps} fps (O to stop)",
            size.x as usize / scale,
            size.y as usize / scale,
        );
        Some(Handle {
            tx: Arc::new(Mutex::new(Some(tx))),
            join: Some(join),
            start: Instant::now(),
            size,
            fps,
            captured: 0,
        })
    }

    /// Per-frame capture while recording: spawn a screenshot of the window
    /// and forward the readback into the encoder channel. The observer runs
    /// a few frames later when the GPU copy lands; the timestamp taken there
    /// is what the encoder's sample durations pace against.
    pub(super) fn capture_frames(
        mut rec: ResMut<Recorder>,
        windows: Query<&Window>,
        scene: Res<crate::SceneTarget>,
        mut commands: Commands,
    ) {
        let Some(handle) = rec.active.as_mut() else {
            return;
        };
        let Ok(window) = windows.single() else {
            return;
        };
        // The encoder and mp4 track are fixed-size for the whole stream, so
        // a resize has to end the take.
        let size = UVec2::new(window.physical_width(), window.physical_height());
        if size != handle.size {
            info!("window resized; stopping recording");
            rec.active = None;
            return;
        }
        // Pace at capture time: a frame the encoder would drop anyway is not
        // worth the GPU copy and readback. Rendering above the take's fps,
        // most frames skip here; below it, every frame captures.
        let due = (handle.start.elapsed().as_secs_f64() * handle.fps) as u64 + 1;
        if due <= handle.captured {
            return;
        }
        handle.captured = due;
        let tx = handle.tx.clone();
        let start = handle.start;
        // The offscreen scene target, not the window: the window frame has
        // the UI composited on top (menu, REC overlay).
        commands.spawn(Screenshot::image(scene.0.clone())).observe(
            move |mut trigger: Trigger<ScreenshotCaptured>| {
                let img = &mut trigger.event_mut().0;
                // Move the multi-MB buffer out instead of cloning it; the
                // screenshot entity is despawned right after this anyway.
                let Some(data) = img.data.take() else {
                    return;
                };
                let frame = Frame {
                    t: start.elapsed().as_secs_f64(),
                    w: img.width(),
                    h: img.height(),
                    format: img.texture_descriptor.format,
                    data,
                };
                // Slot empty means the take already stopped; full channel
                // means the encoder is behind. Drop the frame either way.
                if let Some(tx) = tx.lock().unwrap().as_ref() {
                    let _ = tx.try_send(frame);
                }
            },
        );
    }

    /// Byte offsets of (red, blue) within a 4-byte pixel; green is always at
    /// offset 1. Swapchain formats seen in practice; anything else (a 10-bit
    /// or float surface) aborts with a clear error instead of writing garbage.
    fn rb_offsets(format: TextureFormat) -> Option<(usize, usize)> {
        match format {
            TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb => Some((2, 0)),
            TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb => Some((0, 2)),
            _ => None,
        }
    }

    /// Reusable I420 frame fed straight to the encoder. Hand-rolled integer
    /// converter because the openh264 crate's fast path only covers 3-byte
    /// RGB; its generic path goes through f32 per pixel, too slow at 1080p60.
    struct I420 {
        w: usize,
        h: usize,
        data: Vec<u8>,
    }

    impl I420 {
        fn new(w: usize, h: usize) -> Self {
            Self {
                w,
                h,
                data: vec![0; w * h * 3 / 2],
            }
        }

        /// BT.709 limited-range conversion with 2x2 chroma averaging,
        /// cropping the source (row length `src_w`) down to the even
        /// (self.w, self.h). Works row-pair at a time on pre-sliced rows so
        /// the compiler can hoist the bounds checks out of the hot loop. The
        /// coefficient sums keep every result inside u8 range, no clamping.
        fn fill(&mut self, src: &[u8], src_w: usize, r: usize, b: usize) {
            let (w, h) = (self.w, self.h);
            let (y_plane, uv) = self.data.split_at_mut(w * h);
            let (u_plane, v_plane) = uv.split_at_mut(w * h / 4);
            for by in 0..h / 2 {
                let src0 = &src[(by * 2) * src_w * 4..][..w * 4];
                let src1 = &src[(by * 2 + 1) * src_w * 4..][..w * 4];
                let (y0, rest) = y_plane[(by * 2) * w..].split_at_mut(w);
                let y1 = &mut rest[..w];
                let u_row = &mut u_plane[by * (w / 2)..][..w / 2];
                let v_row = &mut v_plane[by * (w / 2)..][..w / 2];
                for bx in 0..w / 2 {
                    let (mut sr, mut sg, mut sb) = (0i32, 0i32, 0i32);
                    let mut luma = |px: &[u8]| {
                        let (pr, pg, pb) = (px[r] as i32, px[1] as i32, px[b] as i32);
                        sr += pr;
                        sg += pg;
                        sb += pb;
                        (16 + ((47 * pr + 157 * pg + 16 * pb + 128) >> 8)) as u8
                    };
                    y0[bx * 2] = luma(&src0[bx * 8..][..4]);
                    y0[bx * 2 + 1] = luma(&src0[bx * 8 + 4..][..4]);
                    y1[bx * 2] = luma(&src1[bx * 8..][..4]);
                    y1[bx * 2 + 1] = luma(&src1[bx * 8 + 4..][..4]);
                    let (pr, pg, pb) = (sr / 4, sg / 4, sb / 4);
                    u_row[bx] = (128 + ((-26 * pr - 86 * pg + 112 * pb + 128) >> 8)) as u8;
                    v_row[bx] = (128 + ((112 * pr - 102 * pg - 10 * pb + 128) >> 8)) as u8;
                }
            }
        }
    }

    impl YUVSource for I420 {
        fn dimensions(&self) -> (usize, usize) {
            (self.w, self.h)
        }
        fn strides(&self) -> (usize, usize, usize) {
            (self.w, self.w / 2, self.w / 2)
        }
        fn y(&self) -> &[u8] {
            &self.data[..self.w * self.h]
        }
        fn u(&self) -> &[u8] {
            let base = self.w * self.h;
            &self.data[base..base + base / 4]
        }
        fn v(&self) -> &[u8] {
            let base = self.w * self.h;
            &self.data[base + base / 4..]
        }
    }

    /// openh264 NAL units come with their annex-b start code attached; the
    /// mp4 sample format wants bare payloads behind 4-byte length prefixes.
    fn strip_start_code(nal: &[u8]) -> &[u8] {
        if let Some(p) = nal.strip_prefix(&[0, 0, 0, 1]) {
            p
        } else if let Some(p) = nal.strip_prefix(&[0, 0, 1]) {
            p
        } else {
            nal
        }
    }

    struct EncState {
        encoder: Encoder,
        yuv: I420,
        writer: Mp4Writer<BufWriter<File>>,
        track_added: bool,
        sps: Option<Vec<u8>>,
        pps: Option<Vec<u8>>,
        /// Largest sample so far, to pre-size the next one's buffer.
        sample_hint: usize,
    }

    /// Encoder thread count: most of the machine, minus headroom for the
    /// game loop and the capture readback.
    fn threads() -> u16 {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        cores.saturating_sub(2).clamp(2, 8) as u16
    }

    fn init_state(w: usize, h: usize, fps: f64, file: File) -> Result<EncState, String> {
        let config = EncoderConfig::new()
            .bitrate(BitRate::from_bps(BITRATE_BPS))
            .max_frame_rate(FrameRate::from_hz(fps as f32))
            // Never let rate control drop frames wholesale: over-budget
            // spikes (the default, with this content) turned into skipped
            // frames and visible stutter. The QP clamp bounds quality
            // instead, letting bitrate overshoot on chaos.
            .skip_frames(false)
            .qp(QpRange::new(QP_MIN, QP_MAX))
            .intra_frame_period(IntraFramePeriod::from_num_frames(
                (fps * IDR_INTERVAL_SECS) as u32,
            ))
            // Declare the BT.709 limited-range conversion done in fill();
            // unflagged HD video gets assumed 709 anyway, but players fed
            // the old unflagged 601 conversion showed shifted colors.
            .vui(VuiConfig::bt709())
            // Leave two cores for the render side; the harness measured Low
            // complexity as fast as it sounds and no worse than Medium on
            // this content (the QP clamp holds quality, bitrate absorbs it).
            .num_threads(threads())
            .max_slice_len(MAX_SLICE_LEN)
            .complexity(Complexity::Low)
            // Screen-content mode reads as a misnomer for a particle field,
            // but the harness measured it +2.5 dB at 1080p and +4 dB at 4K
            // over camera mode at the same bitrate: its intra tools fit
            // sharp dots on flat dark background far better.
            .usage_type(UsageType::ScreenContentRealTime);
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| format!("encoder init failed: {e}"))?;
        let writer = Mp4Writer::write_start(
            BufWriter::new(file),
            &Mp4Config {
                major_brand: str::parse("isom").unwrap(),
                minor_version: 512,
                compatible_brands: vec![
                    str::parse("isom").unwrap(),
                    str::parse("iso2").unwrap(),
                    str::parse("avc1").unwrap(),
                    str::parse("mp41").unwrap(),
                ],
                timescale: 1000,
            },
        )
        .map_err(|e| format!("mp4 header write failed: {e}"))?;
        Ok(EncState {
            encoder,
            yuv: I420::new(w, h),
            writer,
            track_added: false,
            sps: None,
            pps: None,
            sample_hint: 0,
        })
    }

    /// 2x2 RGBA box average into dst (tw x th), for half-res capture.
    /// Alpha is skipped: the yuv conversion never reads it.
    fn downsample2(src: &[u8], src_w: usize, tw: usize, th: usize, dst: &mut Vec<u8>) {
        dst.resize(tw * th * 4, 0);
        for y in 0..th {
            let row0 = &src[(y * 2) * src_w * 4..][..tw * 8];
            let row1 = &src[(y * 2 + 1) * src_w * 4..][..tw * 8];
            let out = &mut dst[y * tw * 4..][..tw * 4];
            for x in 0..tw {
                let (a, b) = (&row0[x * 8..][..8], &row1[x * 8..][..8]);
                for c in 0..3 {
                    let sum = a[c] as u16 + a[c + 4] as u16 + b[c] as u16 + b[c + 4] as u16;
                    out[x * 4 + c] = ((sum + 2) >> 2) as u8;
                }
            }
        }
    }

    fn encode_loop(rx: Receiver<Frame>, file: File, path: String, fps: f64, scale: usize) {
        // Delete the eagerly created file if no sample ever reached it
        // (encoder init failure, or a take stopped before the first
        // readback landed): a zero-byte mp4 on disk helps nobody.
        if encode_frames(rx, file, &path, fps, scale) == 0 {
            let _ = std::fs::remove_file(&path);
        }
    }

    /// Runs the encode until the channel closes or a fatal error; returns
    /// the number of timeline frames written into the mp4.
    fn encode_frames(rx: Receiver<Frame>, file: File, path: &str, fps: f64, scale: usize) -> u64 {
        let mut file = Some(file);
        let mut state: Option<EncState> = None;
        let mut written: u64 = 0;
        // Distinct encoded frames, vs `written` timeline ticks: the ratio is
        // the effective frame rate, the first thing to check when a take
        // looks choppy (it means capture or encode fell behind).
        let mut samples: u64 = 0;
        // Reused half-res buffer, allocated on first use when scale == 2.
        let mut half: Vec<u8> = Vec::new();
        while let Ok(frame) = rx.recv() {
            // Sample durations against the fixed timeline: a frame at time t
            // owes the stream samples up to index t*fps, so one slow-rendered
            // frame becomes one sample stretched over n ticks. The capture
            // side paces the fast direction; n = 0 only for stragglers that
            // slipped through, and the cap bounds the jump after a stall.
            let due = (frame.t * fps) as u64 + 1;
            let n = due.saturating_sub(written).min(240) as u32;
            if n == 0 {
                continue;
            }
            // yuv420 needs even dimensions; crop a stray odd row/column
            // (after the optional downscale).
            let (w, h) = (
                (frame.w as usize / scale) & !1,
                (frame.h as usize / scale) & !1,
            );
            if state.is_none() {
                if w == 0 || h == 0 {
                    continue;
                }
                match init_state(w, h, fps, file.take().unwrap()) {
                    Ok(s) => state = Some(s),
                    Err(e) => {
                        error!("recording: {e}");
                        return written;
                    }
                }
            }
            let st = state.as_mut().unwrap();
            // A frame captured mid-resize can land at a different size than
            // the encoder was initialized with; slicing it with the old
            // dimensions would panic the thread. Skip such stragglers (the
            // capture side stops the take on the next frame anyway).
            if (w, h) != (st.yuv.w, st.yuv.h) {
                continue;
            }
            let Some((r, b)) = rb_offsets(frame.format) else {
                error!("recording: unsupported surface format {:?}", frame.format);
                return written;
            };
            if scale == 2 {
                downsample2(&frame.data, frame.w as usize, w, h, &mut half);
                st.yuv.fill(&half, w, r, b);
            } else {
                st.yuv.fill(&frame.data, frame.w as usize, r, b);
            }
            let bs = match st.encoder.encode(&st.yuv) {
                Ok(bs) => bs,
                Err(e) => {
                    error!("recording: encode failed: {e}");
                    return written;
                }
            };
            let is_sync = matches!(bs.frame_type(), FrameType::IDR);
            // Repack NALs straight from the encoder's layer slices: SPS and
            // PPS go into the mp4 avcC header (they never change after the
            // first IDR), everything else into the sample.
            let mut sample = Vec::with_capacity(st.sample_hint);
            for l in 0..bs.num_layers() {
                let layer = bs.layer(l).unwrap();
                for i in 0..layer.nal_count() {
                    let nal = strip_start_code(layer.nal_unit(i).unwrap());
                    match nal.first().map_or(0, |b| b & 0x1f) {
                        7 => st.sps.get_or_insert_with(|| nal.to_vec()),
                        8 => st.pps.get_or_insert_with(|| nal.to_vec()),
                        _ => {
                            sample.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                            sample.extend_from_slice(nal);
                            continue;
                        }
                    };
                }
            }
            st.sample_hint = st.sample_hint.max(sample.len());
            // A frame the encoder skipped entirely (rate control, despite
            // skip_frames(false) being asked for) has no slice NALs; a
            // zero-byte sample corrupts the track, and the timeline pacing
            // absorbs the gap on the next frame anyway.
            if sample.is_empty() {
                continue;
            }
            if !st.track_added {
                // The first frame is an IDR carrying SPS+PPS; the track's
                // avcC header needs them, so it cannot be added any earlier.
                let (Some(sps), Some(pps)) = (&st.sps, &st.pps) else {
                    continue;
                };
                let track = TrackConfig {
                    track_type: TrackType::Video,
                    timescale: fps as u32,
                    language: "und".into(),
                    media_conf: MediaConfig::AvcConfig(AvcConfig {
                        width: w as u16,
                        height: h as u16,
                        seq_param_set: sps.clone(),
                        pic_param_set: pps.clone(),
                    }),
                };
                if let Err(e) = st.writer.add_track(&track) {
                    error!("recording: mp4 track setup failed: {e}");
                    return written;
                }
                st.track_added = true;
            }
            let mp4_sample = Mp4Sample {
                start_time: written,
                duration: n,
                rendering_offset: 0,
                is_sync,
                bytes: Bytes::from(sample),
            };
            if let Err(e) = st.writer.write_sample(1, &mp4_sample) {
                error!("recording: mp4 sample write failed: {e}");
                return written;
            }
            written += n as u64;
            samples += 1;
        }
        if let Some(mut st) = state {
            if let Err(e) = st.writer.write_end() {
                error!("recording: mp4 finalize failed: {e}");
                return written;
            }
            let secs = written as f64 / fps;
            let mb = std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / 1e6);
            let mbps = mb * 8.0 / secs.max(1e-6);
            let fps_eff = samples as f64 / secs.max(1e-6);
            info!(
                "recording saved to {path}: {secs:.1}s, {samples} frames \
                 ({fps_eff:.0} fps effective), {mb:.0} MB, {mbps:.0} Mbps"
            );
        }
        written
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Timeline rate used by every test take.
        const FPS: f64 = 60.0;

        /// Deterministic stand-in for the real content: bright moving dots
        /// over a dark drifting gradient, the near-noise motion that makes
        /// particle fields hard to encode.
        fn synth_rgba(w: usize, h: usize, frames: usize) -> Vec<Vec<u8>> {
            const N: usize = 3000;
            let mut seed = 0x1234_5678_9abc_def0_u64;
            let mut rand = move || {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 40) as u32
            };
            let mut pos: Vec<(f32, f32)> = (0..N)
                .map(|_| ((rand() % w as u32) as f32, (rand() % h as u32) as f32))
                .collect();
            let vel: Vec<(f32, f32)> = (0..N)
                .map(|_| {
                    (
                        (rand() % 1024) as f32 / 128.0 - 4.0,
                        (rand() % 1024) as f32 / 128.0 - 4.0,
                    )
                })
                .collect();
            let color: Vec<[u8; 3]> = (0..N)
                .map(|_| {
                    [
                        64 + (rand() % 192) as u8,
                        64 + (rand() % 192) as u8,
                        64 + (rand() % 192) as u8,
                    ]
                })
                .collect();
            (0..frames)
                .map(|t| {
                    let mut f = vec![0u8; w * h * 4];
                    // Separable background so it costs w+h trig calls, not w*h.
                    let sx: Vec<f32> = (0..w)
                        .map(|x| (x as f32 / 97.0 + t as f32 * 0.05).sin())
                        .collect();
                    let cy: Vec<f32> = (0..h)
                        .map(|y| (y as f32 / 71.0 - t as f32 * 0.03).cos())
                        .collect();
                    for (y, cy) in cy.iter().enumerate() {
                        for (x, sx) in sx.iter().enumerate() {
                            let v = 12.0 + 10.0 * sx * cy;
                            let p = (y * w + x) * 4;
                            f[p] = v as u8;
                            f[p + 1] = v as u8;
                            f[p + 2] = (v * 1.5) as u8;
                            f[p + 3] = 255;
                        }
                    }
                    for i in 0..N {
                        pos[i].0 = (pos[i].0 + vel[i].0).rem_euclid(w as f32 - 2.0);
                        pos[i].1 = (pos[i].1 + vel[i].1).rem_euclid(h as f32 - 2.0);
                        let (px, py) = (pos[i].0 as usize, pos[i].1 as usize);
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let p = ((py + dy) * w + px + dx) * 4;
                                f[p] = color[i][0];
                                f[p + 1] = color[i][1];
                                f[p + 2] = color[i][2];
                            }
                        }
                    }
                    f
                })
                .collect()
        }

        /// Encoder tuning harness, not a pass/fail test: encodes the
        /// synthetic field under config variants, decodes it back, and
        /// prints PSNR, the bitrate actually produced, and encode speed.
        /// Run: cargo test --release quality_harness -- --ignored --nocapture
        #[test]
        #[ignore = "tuning harness, run manually with --ignored --nocapture"]
        fn quality_harness() {
            use openh264::decoder::Decoder;

            let (w, h) = match std::env::var("HARNESS_4K") {
                Ok(_) => (3840usize, 2160usize),
                Err(_) => (1920usize, 1080usize),
            };
            let n_frames = 90usize;
            let rgba = synth_rgba(w, h, n_frames);

            let base = EncoderConfig::new()
                .bitrate(BitRate::from_bps(BITRATE_BPS))
                .max_frame_rate(FrameRate::from_hz(FPS as f32))
                .skip_frames(false)
                .qp(QpRange::new(QP_MIN, QP_MAX))
                .intra_frame_period(IntraFramePeriod::from_num_frames(
                    (FPS * IDR_INTERVAL_SECS) as u32,
                ))
                .vui(VuiConfig::bt709());
            // "prod" mirrors init_state's production config (fixed thread
            // count instead of the machine-derived one).
            let prod = base
                .num_threads(6)
                .max_slice_len(MAX_SLICE_LEN)
                .complexity(Complexity::Low)
                .usage_type(UsageType::ScreenContentRealTime);
            let variants: Vec<(&str, EncoderConfig)> = vec![
                ("prod", prod),
                (
                    "prod camera",
                    prod.usage_type(UsageType::CameraVideoRealTime),
                ),
                ("prod 50M", prod.bitrate(BitRate::from_bps(50_000_000))),
                ("prod 150M", prod.bitrate(BitRate::from_bps(150_000_000))),
                ("prod medium", prod.complexity(Complexity::Medium)),
            ];

            for (name, cfg) in variants {
                let mut enc = Encoder::with_api_config(OpenH264API::from_source(), cfg).unwrap();
                let mut dec = Decoder::new().unwrap();
                let mut yuv = I420::new(w, h);
                let (mut bytes, mut se, mut px) = (0usize, 0u64, 0u64);
                let mut enc_time = std::time::Duration::ZERO;
                for f in &rgba {
                    yuv.fill(f, w, 0, 2);
                    let t0 = std::time::Instant::now();
                    let bs = enc.encode(&yuv).unwrap();
                    let packet = bs.to_vec();
                    enc_time += t0.elapsed();
                    bytes += packet.len();
                    if let Some(dy) = dec.decode(&packet).ok().flatten() {
                        let ys = dy.strides().0;
                        let (dec_y, src_y) = (dy.y(), yuv.y());
                        for row in 0..h {
                            let a = &src_y[row * w..][..w];
                            let b = &dec_y[row * ys..][..w];
                            for i in 0..w {
                                let d = a[i] as i64 - b[i] as i64;
                                se += (d * d) as u64;
                            }
                        }
                        px += (w * h) as u64;
                    }
                }
                let psnr = 10.0 * (255.0f64 * 255.0 / (se as f64 / px as f64)).log10();
                let mbps = (bytes * 8) as f64 * FPS / n_frames as f64 / 1e6;
                let enc_fps = n_frames as f64 / enc_time.as_secs_f64();
                println!("{name:>18}: {psnr:5.2} dB  {mbps:7.2} Mbps  {enc_fps:5.1} enc fps");
            }
        }

        /// Full pipeline on synthetic frames: convert, encode, mux, then
        /// reparse the mp4 and check the track holds the expected samples.
        #[test]
        fn encodes_valid_mp4() {
            let dir = std::env::temp_dir().join("fractality_rec_test");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("out.mp4");
            let file = File::create(&path).unwrap();

            let (w, h) = (320u32, 240u32);
            let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(64);
            for i in 0..30u32 {
                // Moving gradient so inter frames have real motion to code.
                let mut data = vec![0u8; (w * h * 4) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let p = ((y * w + x) * 4) as usize;
                        data[p] = (x + i * 4) as u8;
                        data[p + 1] = (y + i * 2) as u8;
                        data[p + 2] = 128;
                        data[p + 3] = 255;
                    }
                }
                tx.send(Frame {
                    t: i as f64 / FPS,
                    w,
                    h,
                    format: TextureFormat::Bgra8UnormSrgb,
                    data,
                })
                .unwrap();
            }
            drop(tx);
            encode_loop(rx, file, path.to_string_lossy().into_owned(), FPS, 1);

            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[4..8], b"ftyp");
            let size = bytes.len() as u64;
            let reader = std::io::Cursor::new(bytes);
            let mp4 = mp4::Mp4Reader::read_header(reader, size).unwrap();
            let track = mp4.tracks().values().next().unwrap();
            assert_eq!(track.sample_count(), 30);
            assert_eq!(track.width(), w as u16);
            assert_eq!(track.height(), h as u16);
        }

        /// Half-res path: same pipeline at scale 2 muxes a track at half the
        /// source dimensions, and the box filter averages exact 2x2 blocks.
        #[test]
        fn encodes_half_res() {
            let dir = std::env::temp_dir().join("fractality_rec_test");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("half.mp4");
            let file = File::create(&path).unwrap();

            let (w, h) = (320u32, 240u32);
            let (tx, rx) = std::sync::mpsc::sync_channel::<Frame>(64);
            for i in 0..10u32 {
                let mut data = vec![0u8; (w * h * 4) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let p = ((y * w + x) * 4) as usize;
                        data[p] = (x + i * 4) as u8;
                        data[p + 1] = (y + i * 2) as u8;
                        data[p + 2] = 128;
                        data[p + 3] = 255;
                    }
                }
                tx.send(Frame {
                    t: i as f64 / FPS,
                    w,
                    h,
                    format: TextureFormat::Bgra8UnormSrgb,
                    data,
                })
                .unwrap();
            }
            drop(tx);
            encode_loop(rx, file, path.to_string_lossy().into_owned(), FPS, 2);

            let bytes = std::fs::read(&path).unwrap();
            let size = bytes.len() as u64;
            let mp4 = mp4::Mp4Reader::read_header(std::io::Cursor::new(bytes), size).unwrap();
            let track = mp4.tracks().values().next().unwrap();
            assert_eq!(track.sample_count(), 10);
            assert_eq!(track.width(), (w / 2) as u16);
            assert_eq!(track.height(), (h / 2) as u16);

            // Box filter itself: a 2x2 block averages to one pixel, rounded.
            let src = [
                10, 20, 30, 255, 20, 30, 40, 255, //
                30, 40, 50, 255, 40, 50, 60, 255,
            ];
            let mut dst = Vec::new();
            downsample2(&src, 2, 1, 1, &mut dst);
            assert_eq!(&dst[..3], &[25, 35, 45]);
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use std::cell::RefCell;

    use bevy::prelude::*;
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;

    use crate::webutil;

    pub(super) const EXT: &str = "webm";

    thread_local! {
        /// The live MediaRecorder. JS objects are !Send, so it lives here
        /// instead of in the Recorder resource; wasm is single-threaded so
        /// this is always the right thread.
        static ACTIVE: RefCell<Option<web_sys::MediaRecorder>> = RefCell::new(None);
    }

    /// Send token for the Recorder resource; the real state is in ACTIVE.
    /// Dropping it stops the MediaRecorder, which fires onstop and downloads.
    pub(super) struct Handle;

    impl Drop for Handle {
        fn drop(&mut self) {
            ACTIVE.with(|a| {
                if let Some(rec) = a.borrow_mut().take() {
                    let _ = rec.stop();
                }
            });
            info!("recording stopped, downloading webm");
        }
    }

    /// Settings are ignored here: MediaRecorder encodes whatever the canvas
    /// stream provides at the browser's chosen rate.
    pub(super) fn start(
        _window: &Window,
        name: String,
        _settings: &super::RecordSettings,
    ) -> Option<Handle> {
        let Some(canvas) = webutil::canvas() else {
            warn!("recording: canvas not found");
            return None;
        };
        let Ok(stream) = canvas.capture_stream() else {
            warn!("recording: canvas.captureStream failed");
            return None;
        };
        let Ok(rec) = web_sys::MediaRecorder::new_with_media_stream(&stream) else {
            warn!("recording: MediaRecorder unavailable");
            return None;
        };

        let chunks = js_sys::Array::new();
        let sink = chunks.clone();
        let on_data = Closure::<dyn FnMut(web_sys::BlobEvent)>::new(move |ev| {
            if let Some(blob) = web_sys::BlobEvent::data(&ev) {
                sink.push(&blob);
            }
        });
        rec.set_ondataavailable(Some(on_data.as_ref().unchecked_ref()));

        let on_stop = Closure::<dyn FnMut()>::new(move || {
            let opts = web_sys::BlobPropertyBag::new();
            opts.set_type("video/webm");
            if let Ok(blob) = web_sys::Blob::new_with_blob_sequence_and_options(&chunks, &opts) {
                webutil::download_blob(&blob, &name);
            }
            // The forgotten closures pin this array for the page's lifetime;
            // empty it so the take's blobs (potentially hundreds of MB) can
            // be collected once the download is handed off.
            chunks.set_length(0);
        });
        rec.set_onstop(Some(on_stop.as_ref().unchecked_ref()));
        // The browser calls these after stop(); dropping them now would kill
        // the callbacks, so leak them (a few hundred bytes per recording).
        on_data.forget();
        on_stop.forget();

        // 1s timeslices so a long take streams into chunks instead of one
        // giant in-memory blob assembled at stop time.
        if rec.start_with_time_slice(1000).is_err() {
            warn!("recording: MediaRecorder.start failed");
            return None;
        }
        ACTIVE.with(|a| *a.borrow_mut() = Some(rec));
        info!("recording webm (O to stop)");
        Some(Handle)
    }
}
