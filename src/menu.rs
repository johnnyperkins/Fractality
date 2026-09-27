// In-app settings panel (native bevy_ui, no extra deps). Toggle with Esc or M.
// Layout is a fixed-width column of cards; each card is a titled group of rows.
// Rows come in three shapes - slider (label + track + readout), select (label +
// value chip + dropdown), toggle (label + pill switch) - all driven by the
// Settings resource and the mode resources, which the sim reads live every
// frame (see update_params in main.rs).

use bevy::core_pipeline::bloom::Bloom;
use bevy::prelude::*;
use bevy::ui::{ComputedNode, RelativeCursorPosition};

use crate::audio::{AudioCapture, AudioLevels};
use crate::choreographer::Choreographer;
use crate::fractal::Fractal;
use crate::particles::{ParticleCamera, MAX_PARTICLES};
use crate::recorder::{
    RecordSettings, Recorder, REC_FPS_DEFAULT, REC_FPS_MODES, REC_RES_DEFAULT, REC_RES_MODES,
};
use crate::{ColorMode, FlowMode, FractalType, Kaleido, COLOR_MODES, FLOW_MODES};

/// Live-tunable knobs. Source of truth for the whole app.
#[derive(Resource)]
pub struct Settings {
    /// Active particle count (buffer is allocated at MAX_PARTICLES; this just
    /// caps how many are simulated/drawn, so changes are instant, no realloc).
    pub particle_count: u32,
    /// Iteration-count multiplier: scales the depth-based max_iter. Higher =
    /// finer boundary filaments at every zoom (costs GPU).
    pub detail: f32,
    /// Iso-contour advection speed (fraction of view height / sec).
    pub flow_speed: f32,
    /// Alignment force: closing rate (per second) pulling each particle onto
    /// its home fractal iso-band (band_k in the compute shader). Higher =
    /// tighter to the shape; monotonic, cannot overshoot.
    pub align_force: f32,
    /// Boundary condensation: how hard particles freeze onto the fractal
    /// shell (calm_of in the compute shader). 1 = classic feel; higher =
    /// freeze reaches further out and holds harder, so edges render as
    /// crisp filigree; 0 = nothing freezes, everything streams.
    pub condensation: f32,
    /// Brightness multiplier.
    pub brightness: f32,
    /// Particle dot size in pixels.
    pub dot_px: f32,
    /// Bloom intensity.
    pub bloom: f32,
    /// Trail keep factor per frame (at 60 FPS). 0 disables trails entirely;
    /// higher = longer light-trails / motion blur.
    pub trail: f32,
    /// Audio effect gains, live only while audio reactivity is on (levels are
    /// zero otherwise, so these multiply nothing). 1.0 = designed intensity.
    /// Beat/drop ring wave strength.
    pub audio_pulse: f32,
    /// Beat strobe + treble glitter strength.
    pub audio_flash: f32,
    /// Spectrum glow strength (frequency bands painted onto iteration depth).
    pub audio_glow: f32,
    /// Bass zoom-breathe amount.
    pub audio_breathe: f32,
    /// Julia parameter morph radius (fractal type julia only).
    pub audio_morph: f32,
    /// Mid-driven flow speed boost.
    pub audio_flow: f32,
    /// Kaleidoscope fold count (mirror wedges), live only while the
    /// kaleidoscope is on. Integer-snapped by the slider.
    pub kaleido_folds: f32,
    /// Kaleidoscope spin gain: scales the drift + music-driven rotation.
    /// 0 = static wedges.
    pub kaleido_spin: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            particle_count: crate::DEFAULT_COUNT,
            detail: 1.0,
            flow_speed: 0.081,
            align_force: 15.0,
            condensation: 1.0,
            brightness: 1.1,
            dot_px: 1.0,
            bloom: 0.3,
            trail: 0.3,
            audio_pulse: 1.0,
            audio_flash: 0.4,
            audio_glow: 0.5,
            audio_breathe: 1.0,
            audio_morph: 1.0,
            audio_flow: 1.0,
            kaleido_folds: 6.0,
            kaleido_spin: 1.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Setting {
    ParticleCount,
    Detail,
    FlowSpeed,
    AlignForce,
    Condensation,
    Brightness,
    DotSize,
    Bloom,
    Trail,
    AudioPulse,
    AudioFlash,
    AudioGlow,
    AudioBreathe,
    AudioMorph,
    AudioFlow,
    KaleidoFolds,
    KaleidoSpin,
}

/// Sim-shaping rows.
const SIM_SETTINGS: [Setting; 5] = [
    Setting::ParticleCount,
    Setting::Detail,
    Setting::FlowSpeed,
    Setting::AlignForce,
    Setting::Condensation,
];

/// Look rows (nothing here changes particle motion).
const RENDER_SETTINGS: [Setting; 4] = [
    Setting::Brightness,
    Setting::DotSize,
    Setting::Bloom,
    Setting::Trail,
];

/// Audio-effect rows, shown only while audio reactivity is on.
const AUDIO_SETTINGS: [Setting; 6] = [
    Setting::AudioPulse,
    Setting::AudioFlash,
    Setting::AudioGlow,
    Setting::AudioBreathe,
    Setting::AudioMorph,
    Setting::AudioFlow,
];

/// Kaleidoscope rows, shown only while the kaleidoscope is on.
const KALEIDO_SETTINGS: [Setting; 2] = [Setting::KaleidoFolds, Setting::KaleidoSpin];

impl Setting {
    fn label(self) -> &'static str {
        match self {
            Setting::ParticleCount => "Particles",
            Setting::Detail => "Detail",
            Setting::FlowSpeed => "Flow speed",
            Setting::AlignForce => "Align force",
            Setting::Condensation => "Condensation",
            Setting::Brightness => "Brightness",
            Setting::DotSize => "Dot size",
            Setting::Bloom => "Bloom",
            Setting::Trail => "Trails",
            Setting::AudioPulse => "Pulse",
            Setting::AudioFlash => "Flash",
            Setting::AudioGlow => "Spec glow",
            Setting::AudioBreathe => "Breathe",
            Setting::AudioMorph => "Julia morph",
            Setting::AudioFlow => "Flow boost",
            Setting::KaleidoFolds => "Folds",
            Setting::KaleidoSpin => "Spin",
        }
    }

    /// (min, max) in the setting's own units.
    fn range(self) -> (f32, f32) {
        match self {
            Setting::ParticleCount => (100_000.0, MAX_PARTICLES as f32),
            Setting::Detail => (0.5, 8.0),
            Setting::FlowSpeed => (0.0, 0.4),
            Setting::AlignForce => (0.0, 200.0),
            Setting::Condensation => (0.0, 5.0),
            Setting::Brightness => (0.1, 4.0),
            Setting::DotSize => (0.1, 4.0),
            Setting::Bloom => (0.0, 1.0),
            Setting::Trail => (0.0, 0.98),
            // All audio gains share one scale: 0 = effect off, 1 = designed
            // intensity, 2 = double.
            Setting::AudioPulse
            | Setting::AudioFlash
            | Setting::AudioGlow
            | Setting::AudioBreathe
            | Setting::AudioMorph
            | Setting::AudioFlow => (0.0, 2.0),
            Setting::KaleidoFolds => (2.0, 16.0),
            Setting::KaleidoSpin => (0.0, 2.0),
        }
    }

    fn get(self, s: &Settings) -> f32 {
        match self {
            Setting::ParticleCount => s.particle_count as f32,
            Setting::Detail => s.detail,
            Setting::FlowSpeed => s.flow_speed,
            Setting::AlignForce => s.align_force,
            Setting::Condensation => s.condensation,
            Setting::Brightness => s.brightness,
            Setting::DotSize => s.dot_px,
            Setting::Bloom => s.bloom,
            Setting::Trail => s.trail,
            Setting::AudioPulse => s.audio_pulse,
            Setting::AudioFlash => s.audio_flash,
            Setting::AudioGlow => s.audio_glow,
            Setting::AudioBreathe => s.audio_breathe,
            Setting::AudioMorph => s.audio_morph,
            Setting::AudioFlow => s.audio_flow,
            Setting::KaleidoFolds => s.kaleido_folds,
            Setting::KaleidoSpin => s.kaleido_spin,
        }
    }

    fn set(self, s: &mut Settings, v: f32) {
        match self {
            // Snap to 100k steps so the readout stays tidy while dragging.
            Setting::ParticleCount => {
                s.particle_count = ((v / 100_000.0).round() * 100_000.0) as u32
            }
            Setting::Detail => s.detail = v,
            Setting::FlowSpeed => s.flow_speed = v,
            Setting::AlignForce => s.align_force = v,
            Setting::Condensation => s.condensation = v,
            Setting::Brightness => s.brightness = v,
            Setting::DotSize => s.dot_px = v,
            Setting::Bloom => s.bloom = v,
            Setting::Trail => s.trail = v,
            Setting::AudioPulse => s.audio_pulse = v,
            Setting::AudioFlash => s.audio_flash = v,
            Setting::AudioGlow => s.audio_glow = v,
            Setting::AudioBreathe => s.audio_breathe = v,
            Setting::AudioMorph => s.audio_morph = v,
            Setting::AudioFlow => s.audio_flow = v,
            // Snap to whole folds: fractional mirror counts make no sense.
            Setting::KaleidoFolds => s.kaleido_folds = v.round(),
            Setting::KaleidoSpin => s.kaleido_spin = v,
        }
    }

    /// Current value as a 0..1 fraction of the range.
    fn fraction(self, s: &Settings) -> f32 {
        let (lo, hi) = self.range();
        ((self.get(s) - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    fn format(self, s: &Settings) -> String {
        let v = self.get(s);
        match self {
            Setting::ParticleCount => {
                if v >= 1_000_000.0 {
                    format!("{:.1}M", v / 1_000_000.0)
                } else {
                    format!("{:.0}k", v / 1_000.0)
                }
            }
            Setting::KaleidoFolds => format!("{v:.0}"),
            _ => format!("{v:.2}"),
        }
    }
}

/// The list-valued modes. One row shape, one dropdown, one set of
/// handlers for all of them; the per-mode resource is picked in get/set.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Select {
    Fractal,
    Flow,
    Palette,
    // The Recording card is native-only (see REC_SELECTS).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    RecRes,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    RecFps,
}

const SELECTS: [Select; 3] = [Select::Fractal, Select::Flow, Select::Palette];

/// Recording card rows (native builds only; the web recorder has no knobs).
#[cfg(not(target_arch = "wasm32"))]
const REC_SELECTS: [Select; 2] = [Select::RecRes, Select::RecFps];

impl Select {
    fn label(self) -> &'static str {
        match self {
            Select::Fractal => "Fractal",
            Select::Flow => "Flow",
            Select::Palette => "Palette",
            Select::RecRes => "Res",
            Select::RecFps => "FPS",
        }
    }

    /// Key that cycles the same value, shown as a chip on the row.
    /// Empty = no shortcut, the chip is hidden.
    fn key(self) -> &'static str {
        match self {
            Select::Fractal => "F",
            Select::Flow => "G",
            Select::Palette => "C",
            Select::RecRes | Select::RecFps => "",
        }
    }

    fn options(self) -> &'static [&'static str] {
        match self {
            Select::Fractal => &Fractal::NAMES,
            Select::Flow => &FLOW_MODES,
            Select::Palette => &COLOR_MODES,
            Select::RecRes => &REC_RES_MODES,
            Select::RecFps => &REC_FPS_MODES,
        }
    }

    /// Index shown before the user touches anything; must match the owning
    /// resource's Default.
    fn initial(self) -> u32 {
        match self {
            Select::RecRes => REC_RES_DEFAULT,
            Select::RecFps => REC_FPS_DEFAULT,
            _ => 0,
        }
    }

    fn get(
        self,
        fractal: &FractalType,
        flow: &FlowMode,
        palette: &ColorMode,
        rec: &RecordSettings,
    ) -> u32 {
        match self {
            Select::Fractal => fractal.0.id(),
            Select::Flow => flow.0,
            Select::Palette => palette.0,
            Select::RecRes => rec.res,
            Select::RecFps => rec.fps,
        }
    }

    /// Writes only on a real change: a ResMut deref marks the resource changed
    /// even when the value is identical, and the sim resets its view on a
    /// fractal change.
    fn set(
        self,
        index: u32,
        fractal: &mut ResMut<FractalType>,
        flow: &mut ResMut<FlowMode>,
        palette: &mut ResMut<ColorMode>,
        rec: &mut ResMut<RecordSettings>,
    ) {
        match self {
            Select::Fractal if fractal.0.id() != index => fractal.0 = Fractal::from_id(index),
            Select::Flow if flow.0 != index => flow.0 = index,
            Select::Palette if palette.0 != index => palette.0 = index,
            Select::RecRes if rec.res != index => rec.res = index,
            Select::RecFps if rec.fps != index => rec.fps = index,
            _ => {}
        }
    }
}

/// The three on/off features, each a pill-switch row.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Toggle {
    Kaleido,
    Choreographer,
    Audio,
}

impl Toggle {
    fn label(self) -> &'static str {
        match self {
            Toggle::Kaleido => "Kaleidoscope",
            Toggle::Choreographer => "Choreographer",
            Toggle::Audio => "Audio react",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Toggle::Kaleido => "K",
            Toggle::Choreographer => "X",
            Toggle::Audio => "V",
        }
    }

    fn state(self, kaleido: &Kaleido, choreo: &Choreographer, audio: &AudioCapture) -> bool {
        match self {
            Toggle::Kaleido => kaleido.on,
            Toggle::Choreographer => choreo.on,
            Toggle::Audio => audio.enabled,
        }
    }
}

/// Whether the panel is currently shown.
#[derive(Resource)]
pub struct MenuOpen(pub bool);

impl Default for MenuOpen {
    fn default() -> Self {
        Self(true)
    }
}

/// True while the cursor is over the panel or a slider drag is in progress,
/// so the sim ignores those clicks (see update_params in main.rs). Clicks on
/// the fractal outside the panel keep working even with the menu open.
#[derive(Resource, Default)]
pub struct PointerOverMenu(pub bool);

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct SettingValue(Setting);

#[derive(Component)]
struct SliderTrack(Setting);

#[derive(Component)]
struct SliderFill(Setting);

/// Clickable select row; clicking opens/closes its dropdown.
#[derive(Component)]
struct SelectRow(Select);

/// The text showing the active option on a select row.
#[derive(Component)]
struct SelectValue(Select);

/// A dropdown overlay (hidden until its row is clicked).
#[derive(Component)]
struct SelectDropdown(Select);

/// One selectable entry in a dropdown.
#[derive(Component)]
struct SelectOption(Select, u32);

/// Clickable toggle row.
#[derive(Component)]
struct ToggleRow(Toggle);

/// The pill switch on a toggle row: background = state, knob side = state.
#[derive(Component)]
struct TogglePill(Toggle);

/// Slider group revealed only while its toggle is on (kaleidoscope / audio).
#[derive(Component)]
struct ToggleSection(Toggle);

/// The collapsible cards; each pairs a clickable header with a foldable body.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CollapseSection {
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Recording,
    Controls,
}

/// Clickable card header that folds its section's body away. Open state
/// lives here, so each section keeps its own.
#[derive(Component)]
struct CollapseHeader {
    section: CollapseSection,
    open: bool,
}

/// The foldable part of a collapsible card.
#[derive(Component)]
struct CollapseBody(CollapseSection);

/// Caret glyph on a collapsible header, flipped on collapse.
#[derive(Component)]
struct CollapseCaret(CollapseSection);

/// REC chip in the panel header, hidden unless a take is running.
#[derive(Component)]
struct RecChip;

/// The pulsing dot inside the REC chip.
#[derive(Component)]
struct RecDot;

/// The elapsed-time text inside the REC chip.
#[derive(Component)]
struct RecTime;

/// Marker on every clickable menu widget, so track_pointer_over_menu shields
/// the sim from their clicks without enumerating each row type.
#[derive(Component)]
struct MenuInteractive;

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuOpen>()
            .init_resource::<PointerOverMenu>()
            .add_systems(Startup, build_menu)
            .add_systems(
                Update,
                (
                    toggle_menu,
                    track_pointer_over_menu,
                    drag_sliders,
                    update_sliders,
                    click_select_row,
                    click_select_option,
                    close_dropdowns_on_outside_click,
                    update_select_ui,
                    click_toggle,
                    sync_toggles,
                    click_collapse_header,
                    update_rec_chip,
                    apply_bloom,
                ),
            );
    }
}

// Panel chrome.
const PANEL_BG: Color = Color::srgba(0.030, 0.038, 0.062, 0.92);
const PANEL_BORDER: Color = Color::srgba(0.42, 0.55, 0.95, 0.22);
const CARD_BG: Color = Color::srgba(0.075, 0.090, 0.140, 0.55);
const CARD_BORDER: Color = Color::srgba(0.50, 0.60, 0.90, 0.10);
const ROW_HOVER: Color = Color::srgba(0.55, 0.65, 1.00, 0.10);
// Text.
const TEXT_TITLE: Color = Color::srgb(0.93, 0.95, 1.00);
const TEXT_LABEL: Color = Color::srgb(0.72, 0.78, 0.90);
const TEXT_MUTED: Color = Color::srgb(0.46, 0.52, 0.64);
const TEXT_VALUE: Color = Color::srgb(1.00, 0.86, 0.48);
// Widgets.
const TRACK_IDLE: Color = Color::srgb(0.115, 0.135, 0.200);
const TRACK_HOVER: Color = Color::srgb(0.170, 0.200, 0.290);
const FILL_COLOR: Color = Color::srgb(0.38, 0.50, 0.86);
const KNOB_COLOR: Color = Color::srgb(0.80, 0.87, 1.00);
const ACCENT: Color = Color::srgb(0.42, 0.58, 1.00);
const REC_RED: Color = Color::srgb(0.96, 0.30, 0.30);
const PILL_OFF: Color = Color::srgb(0.14, 0.16, 0.23);
const CHIP_BG: Color = Color::srgba(0.60, 0.70, 1.00, 0.10);
const DROPDOWN_BG: Color = Color::srgba(0.055, 0.065, 0.105, 0.98);
const OPTION_ACTIVE: Color = Color::srgba(0.42, 0.58, 1.00, 0.30);

// Metrics. One panel width, one label column, one readout column: every row
// shape lines up because they all use these.
const PANEL_W: f32 = 376.0;
const LABEL_W: f32 = 104.0;
const VALUE_W: f32 = 52.0;
const ROW_H: f32 = 22.0;
const FONT_ROW: f32 = 13.0;
const FONT_SMALL: f32 = 11.0;

/// Concrete return type, not `impl Bundle`: the widget builders below embed
/// this in their own opaque types, and an opaque type there would drag the
/// borrowed &str lifetime along with it.
fn text_bundle(
    text: impl Into<String>,
    size: f32,
    color: Color,
) -> (Text, TextFont, TextColor, TextLayout) {
    (
        Text::new(text),
        TextFont {
            font_size: size,
            ..default()
        },
        TextColor(color),
        TextLayout::new_with_no_wrap(),
    )
}

/// Keyboard-shortcut chip, e.g. the "K" next to the kaleidoscope row.
/// An empty key collapses the chip so keyless rows keep the same shell.
fn key_chip(key: &str) -> impl Bundle {
    (
        Node {
            display: if key.is_empty() {
                Display::None
            } else {
                Display::Flex
            },
            padding: UiRect::axes(Val::Px(5.0), Val::Px(1.0)),
            min_width: Val::Px(20.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(CHIP_BG),
        BorderRadius::all(Val::Px(4.0)),
        children![text_bundle(key, FONT_SMALL, TEXT_MUTED)],
    )
}

/// Card: a bordered, titled group. Rows are appended by the caller.
fn card_node() -> impl Bundle {
    (
        Node {
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(6.0),
            padding: UiRect::all(Val::Px(10.0)),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(CARD_BG),
        BorderColor(CARD_BORDER),
        BorderRadius::all(Val::Px(9.0)),
    )
}

/// The accent tick that leads every card title.
fn title_tick() -> impl Bundle {
    (
        Node {
            width: Val::Px(3.0),
            height: Val::Px(11.0),
            ..default()
        },
        BackgroundColor(ACCENT),
        BorderRadius::all(Val::Px(2.0)),
    )
}

/// Accent tick + small caps title, used at the top of every card.
fn card_title(title: &str) -> impl Bundle {
    (
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(6.0),
            margin: UiRect::bottom(Val::Px(1.0)),
            ..default()
        },
        children![title_tick(), text_bundle(title, FONT_SMALL, TEXT_MUTED)],
    )
}

/// Caret glyphs: down for an open section (and a select's dropdown hint),
/// right for a folded one.
const CARET_DOWN: &str = "\u{25be}";
const CARET_RIGHT: &str = "\u{25b8}";

fn caret_glyph(open: bool) -> &'static str {
    if open {
        CARET_DOWN
    } else {
        CARET_RIGHT
    }
}

/// Clickable title of a collapsible card: card_title's tick and text plus a
/// caret showing the fold state. Carries the section's open state.
fn collapse_header(section: CollapseSection, title: &str, open: bool) -> impl Bundle {
    (
        Button,
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(6.0),
            ..default()
        },
        BackgroundColor(Color::NONE),
        MenuInteractive,
        CollapseHeader { section, open },
        children![
            title_tick(),
            text_bundle(title, FONT_SMALL, TEXT_MUTED),
            (
                text_bundle(caret_glyph(open), 10.0, TEXT_MUTED),
                CollapseCaret(section),
            ),
        ],
    )
}

fn label_bundle(text: &str) -> impl Bundle {
    (
        Node {
            width: Val::Px(LABEL_W),
            ..default()
        },
        children![text_bundle(text, FONT_ROW, TEXT_LABEL)],
    )
}

fn value_bundle(setting: Setting) -> impl Bundle {
    (
        Text::new(String::new()),
        TextFont {
            font_size: FONT_ROW,
            ..default()
        },
        TextColor(TEXT_VALUE),
        TextLayout::new_with_justify(JustifyText::Right),
        Node {
            width: Val::Px(VALUE_W),
            ..default()
        },
        SettingValue(setting),
    )
}

/// Track + fill + knob. The track grows into whatever width the row leaves.
fn slider_bundle(setting: Setting) -> impl Bundle {
    (
        Button,
        RelativeCursorPosition::default(),
        Node {
            flex_grow: 1.0,
            height: Val::Px(14.0),
            padding: UiRect::all(Val::Px(3.0)),
            ..default()
        },
        BackgroundColor(TRACK_IDLE),
        BorderRadius::all(Val::Px(7.0)),
        SliderTrack(setting),
        children![(
            Node {
                width: Val::Percent(50.0),
                height: Val::Percent(100.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::End,
                ..default()
            },
            BackgroundColor(FILL_COLOR),
            BorderRadius::all(Val::Px(4.0)),
            SliderFill(setting),
            // Handle rides the fill's leading edge; the negative margin lets it
            // straddle the boundary instead of sitting inside the fill.
            children![(
                Node {
                    width: Val::Px(12.0),
                    height: Val::Px(12.0),
                    margin: UiRect::right(Val::Px(-6.0)),
                    ..default()
                },
                BackgroundColor(KNOB_COLOR),
                BorderRadius::MAX,
            )],
        )],
    )
}

fn slider_row(setting: Setting) -> impl Bundle {
    (
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            height: Val::Px(ROW_H),
            column_gap: Val::Px(8.0),
            ..default()
        },
        children![
            label_bundle(setting.label()),
            slider_bundle(setting),
            value_bundle(setting),
        ],
    )
}

/// Row shell shared by selects and toggles: hover-highlighted button with a
/// label, a caller-supplied control, and a shortcut chip on the right.
fn interactive_row(label: &'static str, key: &'static str, control: impl Bundle) -> impl Bundle {
    (
        Button,
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            height: Val::Px(ROW_H),
            column_gap: Val::Px(8.0),
            padding: UiRect::horizontal(Val::Px(4.0)),
            margin: UiRect::horizontal(Val::Px(-4.0)),
            ..default()
        },
        BackgroundColor(Color::NONE),
        BorderRadius::all(Val::Px(5.0)),
        MenuInteractive,
        children![label_bundle(label), control, key_chip(key)],
    )
}

/// Value chip + caret: the closed state of a select.
fn select_control(select: Select, initial: &'static str) -> impl Bundle {
    (
        Node {
            flex_grow: 1.0,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            justify_content: JustifyContent::SpaceBetween,
            padding: UiRect::axes(Val::Px(7.0), Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(CHIP_BG),
        BorderRadius::all(Val::Px(5.0)),
        children![
            (
                text_bundle(initial, FONT_ROW, TEXT_VALUE),
                SelectValue(select),
            ),
            text_bundle(CARET_DOWN, 10.0, TEXT_MUTED),
        ],
    )
}

/// Pill switch: the knob sits left when off, right when on.
fn toggle_control(toggle: Toggle) -> impl Bundle {
    (
        Node {
            width: Val::Px(34.0),
            height: Val::Px(18.0),
            padding: UiRect::all(Val::Px(2.0)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Start,
            flex_grow: 0.0,
            margin: UiRect::right(Val::Auto),
            ..default()
        },
        BackgroundColor(PILL_OFF),
        BorderRadius::MAX,
        TogglePill(toggle),
        children![(
            Node {
                width: Val::Px(14.0),
                height: Val::Px(14.0),
                ..default()
            },
            BackgroundColor(KNOB_COLOR),
            BorderRadius::MAX,
        )],
    )
}

/// Key list, split into two columns of (chip, description) in the panel and
/// printed to the console at startup.
pub const CONTROLS: [(&str, &str); 18] = [
    ("W A S D", "pan"),
    ("Scroll", "zoom at cursor"),
    ("LMB", "blast (hold)"),
    ("RMB", "vortex (hold)"),
    ("Space", "dissolve"),
    ("Z", "auto-zoom dive"),
    ("R", "reset view"),
    ("P", "screenshot"),
    ("O", "record video"),
    ("F", "fractal"),
    ("G", "flow mode"),
    ("C", "palette"),
    ("K", "kaleidoscope"),
    ("V", "audio react"),
    ("X", "choreographer"),
    ("Shift+1-9", "save view"),
    ("1-9", "fly to view"),
    ("Esc / M", "menu"),
];

/// Select row plus its floating dropdown, the shape shared by the scene and
/// recording cards.
fn spawn_select_row(card: &mut ChildSpawnerCommands, select: Select) {
    card.spawn((
        interactive_row(
            select.label(),
            select.key(),
            select_control(select, select.options()[select.initial() as usize]),
        ),
        SelectRow(select),
    ))
    .with_children(|row| {
        // Absolutely positioned so the open list floats over
        // the rows below instead of pushing them down.
        row.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(LABEL_W + 12.0),
                top: Val::Px(ROW_H + 1.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(4.0)),
                border: UiRect::all(Val::Px(1.0)),
                row_gap: Val::Px(1.0),
                ..default()
            },
            BackgroundColor(DROPDOWN_BG),
            BorderColor(PANEL_BORDER),
            BorderRadius::all(Val::Px(8.0)),
            BoxShadow::new(
                Color::srgba(0.0, 0.0, 0.0, 0.6),
                Val::Px(0.0),
                Val::Px(6.0),
                Val::Px(0.0),
                Val::Px(14.0),
            ),
            GlobalZIndex(10),
            // Inherited (not Visible) when open, so closing the
            // whole menu also hides an open dropdown.
            Visibility::Hidden,
            Interaction::default(),
            MenuInteractive,
            SelectDropdown(select),
        ))
        .with_children(|dd| {
            for (i, name) in select.options().iter().enumerate() {
                dd.spawn((
                    Button,
                    Node {
                        padding: UiRect::axes(Val::Px(9.0), Val::Px(3.0)),
                        min_width: Val::Px(130.0),
                        ..default()
                    },
                    // The default starts tinted as the active option.
                    BackgroundColor(if i as u32 == select.initial() {
                        OPTION_ACTIVE
                    } else {
                        Color::NONE
                    }),
                    BorderRadius::all(Val::Px(5.0)),
                    MenuInteractive,
                    SelectOption(select, i as u32),
                    children![text_bundle(*name, FONT_ROW, TEXT_LABEL)],
                ));
            }
        });
    });
}

/// Slider group inside a card, collapsed to nothing until its effect is on.
fn section_node() -> Node {
    Node {
        display: Display::None,
        flex_direction: FlexDirection::Column,
        row_gap: Val::Px(6.0),
        padding: UiRect::left(Val::Px(6.0)),
        margin: UiRect::bottom(Val::Px(2.0)),
        ..default()
    }
}

/// Toggle row, followed by its slider group when it has one.
fn spawn_toggle_row(card: &mut ChildSpawnerCommands, toggle: Toggle, sliders: &[Setting]) {
    card.spawn((
        interactive_row(toggle.label(), toggle.key(), toggle_control(toggle)),
        ToggleRow(toggle),
    ));
    if !sliders.is_empty() {
        card.spawn((section_node(), ToggleSection(toggle)))
            .with_children(|col| {
                for &setting in sliders {
                    col.spawn(slider_row(setting));
                }
            });
    }
}

fn build_menu(mut commands: Commands) {
    // Floating REC indicator, top-right, outside the panel so it shows with
    // the menu closed too. Kept out of native captures: the UI only exists
    // on the present camera while captures read the offscreen scene target
    // (the web recorder takes the whole canvas, UI included).
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(18.0),
            top: Val::Px(18.0),
            display: Display::None,
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(6.0),
            padding: UiRect::axes(Val::Px(9.0), Val::Px(4.0)),
            border: UiRect::all(Val::Px(1.0)),
            ..default()
        },
        BackgroundColor(PANEL_BG),
        BorderColor(REC_RED.with_alpha(0.35)),
        BorderRadius::all(Val::Px(8.0)),
        RecChip,
        children![
            (
                Node {
                    width: Val::Px(8.0),
                    height: Val::Px(8.0),
                    ..default()
                },
                BackgroundColor(REC_RED),
                BorderRadius::MAX,
                RecDot,
            ),
            text_bundle("REC", FONT_SMALL, TEXT_LABEL),
            (text_bundle("0:00", FONT_ROW, REC_RED), RecTime),
        ],
    ));

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(18.0),
                top: Val::Px(18.0),
                width: Val::Px(PANEL_W),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(13.0)),
                border: UiRect::all(Val::Px(1.0)),
                row_gap: Val::Px(9.0),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            BorderColor(PANEL_BORDER),
            BorderRadius::all(Val::Px(14.0)),
            BoxShadow::new(
                Color::srgba(0.0, 0.0, 0.0, 0.55),
                Val::Px(0.0),
                Val::Px(8.0),
                Val::Px(0.0),
                Val::Px(22.0),
            ),
            // Starts visible to match MenuOpen's default.
            Visibility::Visible,
            MenuRoot,
        ))
        .with_children(|parent| {
            // Header: mark, title, and the key that hides the panel.
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(9.0),
                    padding: UiRect::bottom(Val::Px(9.0)),
                    border: UiRect::bottom(Val::Px(1.0)),
                    ..default()
                },
                BorderColor(PANEL_BORDER),
                children![
                    (
                        Node {
                            width: Val::Px(9.0),
                            height: Val::Px(9.0),
                            ..default()
                        },
                        BackgroundColor(ACCENT),
                        BorderRadius::MAX,
                    ),
                    (
                        Node {
                            flex_grow: 1.0,
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(1.0),
                            ..default()
                        },
                        children![
                            text_bundle("FRACTALITY", 17.0, TEXT_TITLE),
                            text_bundle("particle fractal explorer", 10.0, TEXT_MUTED),
                        ],
                    ),
                    key_chip("Esc"),
                ],
            ));

            // Scene: the three list-valued modes, each with a dropdown.
            parent.spawn(card_node()).with_children(|card| {
                card.spawn(card_title("SCENE"));
                for select in SELECTS {
                    spawn_select_row(card, select);
                }
            });

            parent.spawn(card_node()).with_children(|card| {
                card.spawn(card_title("SIMULATION"));
                for setting in SIM_SETTINGS {
                    card.spawn(slider_row(setting));
                }
            });

            parent.spawn(card_node()).with_children(|card| {
                card.spawn(card_title("RENDER"));
                for setting in RENDER_SETTINGS {
                    card.spawn(slider_row(setting));
                }
            });

            // Effects: toggles, kaleidoscope and audio each with a slider
            // group folded away until the effect is on.
            parent.spawn(card_node()).with_children(|card| {
                card.spawn(card_title("EFFECTS"));
                spawn_toggle_row(card, Toggle::Kaleido, &KALEIDO_SETTINGS);
                spawn_toggle_row(card, Toggle::Choreographer, &[]);
                spawn_toggle_row(card, Toggle::Audio, &AUDIO_SETTINGS);
            });

            // Recording: capture options applied to the next O take,
            // collapsed until its title is clicked. Native only; the web
            // path records via MediaRecorder with no knobs.
            #[cfg(not(target_arch = "wasm32"))]
            parent.spawn(card_node()).with_children(|card| {
                // The O chip pushed to the header's right edge.
                card.spawn(collapse_header(
                    CollapseSection::Recording,
                    "RECORDING",
                    false,
                ))
                .with_children(|header| {
                    header.spawn(Node {
                        flex_grow: 1.0,
                        ..default()
                    });
                    header.spawn(key_chip("O"));
                });
                card.spawn((
                    Node {
                        display: Display::None,
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(6.0),
                        margin: UiRect::top(Val::Px(2.0)),
                        ..default()
                    },
                    CollapseBody(CollapseSection::Recording),
                ))
                .with_children(|body| {
                    for select in REC_SELECTS {
                        spawn_select_row(body, select);
                    }
                });
            });

            // Controls: clickable title collapses the list.
            parent.spawn(card_node()).with_children(|card| {
                card.spawn(collapse_header(CollapseSection::Controls, "CONTROLS", true));
                card.spawn((
                    Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(14.0),
                        margin: UiRect::top(Val::Px(2.0)),
                        ..default()
                    },
                    CollapseBody(CollapseSection::Controls),
                ))
                .with_children(|body| {
                    for column in CONTROLS.chunks(CONTROLS.len() / 2) {
                        body.spawn(Node {
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(3.0),
                            ..default()
                        })
                        .with_children(|col| {
                            for (key, what) in column {
                                col.spawn(Node {
                                    flex_direction: FlexDirection::Row,
                                    align_items: AlignItems::Center,
                                    column_gap: Val::Px(6.0),
                                    ..default()
                                })
                                .with_children(|line| {
                                    line.spawn(key_chip(key));
                                    line.spawn(text_bundle(*what, FONT_SMALL, TEXT_MUTED));
                                });
                            }
                        });
                    }
                });
            });
        });
}

fn toggle_menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut open: ResMut<MenuOpen>,
    mut root: Query<&mut Visibility, With<MenuRoot>>,
    mut dropdowns: Query<&mut Visibility, (With<SelectDropdown>, Without<MenuRoot>)>,
) {
    if keys.just_pressed(KeyCode::Escape) || keys.just_pressed(KeyCode::KeyM) {
        open.0 = !open.0;
        for mut vis in &mut root {
            *vis = if open.0 {
                Visibility::Visible
            } else {
                Visibility::Hidden
            };
        }
        // Never reopen the panel with a stale dropdown hanging over the rows.
        for mut vis in &mut dropdowns {
            *vis = Visibility::Hidden;
        }
    }
}

/// Cursor inside the panel rect, or a drag/hover on one of its widgets (a
/// track stays Pressed even after the cursor leaves it). The rect test is what
/// makes this reliable: bevy_ui's focus system stops at the first hovered node,
/// so an Interaction on the root reports None whenever a label or card sits
/// under the cursor.
fn track_pointer_over_menu(
    mut over: ResMut<PointerOverMenu>,
    open: Res<MenuOpen>,
    windows: Query<&Window>,
    roots: Query<(&ComputedNode, &GlobalTransform), With<MenuRoot>>,
    widgets: Query<&Interaction, Or<(With<SliderTrack>, With<MenuInteractive>)>>,
) {
    let busy = widgets.iter().any(|i| *i != Interaction::None);
    let inside = open.0
        && windows
            .single()
            .ok()
            .and_then(|w| w.cursor_position().map(|c| c * w.scale_factor()))
            .is_some_and(|cursor| {
                roots.iter().any(|(node, transform)| {
                    Rect::from_center_size(transform.translation().truncate(), node.size())
                        .contains(cursor)
                })
            });
    let now = busy || inside;
    // Write only on transitions so the resource is not marked changed every
    // frame.
    if over.0 != now {
        over.0 = now;
    }
}

/// While a track is pressed, map cursor x to the setting's range. Interaction
/// stays Pressed until mouse release even if the cursor leaves the node, and
/// RelativeCursorPosition keeps updating, so this drags smoothly.
fn drag_sliders(
    mut settings: ResMut<Settings>,
    mut tracks: Query<(
        &Interaction,
        &RelativeCursorPosition,
        &SliderTrack,
        &mut BackgroundColor,
    )>,
) {
    for (interaction, rel, track, mut bg) in &mut tracks {
        // Compare before writing: an unconditional write marks every track's
        // BackgroundColor changed every frame, re-extracting them all into
        // the render world.
        let color = match interaction {
            Interaction::Pressed => {
                if let Some(pos) = rel.normalized {
                    let (lo, hi) = track.0.range();
                    track
                        .0
                        .set(&mut settings, lo + pos.x.clamp(0.0, 1.0) * (hi - lo));
                }
                TRACK_HOVER
            }
            Interaction::Hovered => TRACK_HOVER,
            Interaction::None => TRACK_IDLE,
        };
        if bg.0 != color {
            bg.0 = color;
        }
    }
}

fn update_sliders(
    open: Res<MenuOpen>,
    settings: Res<Settings>,
    mut values: Query<(&mut Text, &SettingValue)>,
    mut fills: Query<(&mut Node, &SliderFill)>,
) {
    // Rewriting the texts and fill widths unconditionally forced a text
    // reshape and a full UI relayout every frame the menu was open (and it
    // opens by default). Settings change only via slider drags or the
    // choreographer's style ticks; refresh only then, plus on reopen so the
    // panel never shows stale values.
    if !open.0 || (!settings.is_changed() && !open.is_changed()) {
        return;
    }
    for (mut text, sv) in &mut values {
        *text = Text::new(sv.0.format(&settings));
    }
    for (mut node, fill) in &mut fills {
        node.width = Val::Percent(fill.0.fraction(&settings) * 100.0);
    }
}

/// Clicking a select row opens its dropdown and closes any other. Options are
/// children of the dropdown overlay and capture their own clicks (focus blocks
/// the row), so a selection click never re-toggles here.
fn click_select_row(
    mut rows: Query<(&Interaction, &SelectRow, &mut BackgroundColor), Changed<Interaction>>,
    mut dropdowns: Query<(&mut Visibility, &SelectDropdown)>,
) {
    for (interaction, row, mut bg) in &mut rows {
        let color = match interaction {
            Interaction::Pressed => {
                for (mut vis, dd) in &mut dropdowns {
                    *vis = if dd.0 == row.0 && *vis == Visibility::Hidden {
                        Visibility::Inherited
                    } else {
                        Visibility::Hidden
                    };
                }
                ROW_HOVER
            }
            Interaction::Hovered => ROW_HOVER,
            Interaction::None => Color::NONE,
        };
        if bg.0 != color {
            bg.0 = color;
        }
    }
}

/// Option hover highlight + click-to-select, closing the dropdown.
fn click_select_option(
    mut fractal: ResMut<FractalType>,
    mut flow: ResMut<FlowMode>,
    mut palette: ResMut<ColorMode>,
    mut rec: ResMut<RecordSettings>,
    mut options: Query<(&Interaction, &SelectOption, &mut BackgroundColor), Changed<Interaction>>,
    mut dropdowns: Query<&mut Visibility, With<SelectDropdown>>,
) {
    for (interaction, option, mut bg) in &mut options {
        match interaction {
            Interaction::Pressed => {
                option
                    .0
                    .set(option.1, &mut fractal, &mut flow, &mut palette, &mut rec);
                for mut vis in &mut dropdowns {
                    *vis = Visibility::Hidden;
                }
            }
            Interaction::Hovered => bg.0 = ROW_HOVER,
            // Restore the active-option tint rather than clearing it.
            Interaction::None => {
                let active = option.0.get(&fractal, &flow, &palette, &rec) == option.1;
                bg.0 = if active { OPTION_ACTIVE } else { Color::NONE };
            }
        }
    }
}

/// A click anywhere else (canvas or another part of the panel) dismisses an
/// open dropdown, so it never lingers over the rows it covers.
fn close_dropdowns_on_outside_click(
    mouse: Res<ButtonInput<MouseButton>>,
    inside: Query<&Interaction, Or<(With<SelectRow>, With<SelectDropdown>, With<SelectOption>)>>,
    mut dropdowns: Query<&mut Visibility, With<SelectDropdown>>,
) {
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if inside.iter().any(|i| *i != Interaction::None) {
        return;
    }
    for mut vis in &mut dropdowns {
        if *vis != Visibility::Hidden {
            *vis = Visibility::Hidden;
        }
    }
}

/// Keep the closed-state text and the active-option tint in sync however a
/// mode changes (dropdown or the F / G / C keys).
fn update_select_ui(
    fractal: Res<FractalType>,
    flow: Res<FlowMode>,
    palette: Res<ColorMode>,
    rec: Res<RecordSettings>,
    mut values: Query<(&mut Text, &SelectValue)>,
    mut options: Query<(&mut BackgroundColor, &SelectOption)>,
) {
    if !fractal.is_changed() && !flow.is_changed() && !palette.is_changed() && !rec.is_changed() {
        return;
    }
    for (mut text, value) in &mut values {
        let index = value.0.get(&fractal, &flow, &palette, &rec) as usize;
        *text = Text::new(value.0.options()[index]);
    }
    for (mut bg, option) in &mut options {
        let active = option.0.get(&fractal, &flow, &palette, &rec) == option.1;
        let color = if active { OPTION_ACTIVE } else { Color::NONE };
        if bg.0 != color {
            bg.0 = color;
        }
    }
}

/// Clicking a toggle row flips its feature, same as the K / X / V keys. The
/// choreographer click only files a request; update_choreographer does the
/// engage/disengage (it owns the save/restore of the settings it touches).
fn click_toggle(
    mut kaleido: ResMut<Kaleido>,
    mut choreo: ResMut<Choreographer>,
    mut audio: ResMut<AudioCapture>,
    mut rows: Query<(&Interaction, &ToggleRow, &mut BackgroundColor), Changed<Interaction>>,
) {
    for (interaction, row, mut bg) in &mut rows {
        let color = match interaction {
            Interaction::Pressed => {
                match row.0 {
                    Toggle::Kaleido => kaleido.on = !kaleido.on,
                    Toggle::Choreographer => choreo.want_toggle = true,
                    Toggle::Audio => audio.enabled = !audio.enabled,
                }
                ROW_HOVER
            }
            Interaction::Hovered => ROW_HOVER,
            Interaction::None => Color::NONE,
        };
        if bg.0 != color {
            bg.0 = color;
        }
    }
}

/// Pills and their slider sections follow the real state, however it changed
/// (click, key, idle engagement, input exit). update_choreographer writes
/// Choreographer (idle timer, want_toggle) every frame, so is_changed is
/// useless here; diff the three flags instead. Display::None collapses a
/// section entirely (no reserved space), accordion-style.
fn sync_toggles(
    kaleido: Res<Kaleido>,
    choreo: Res<Choreographer>,
    audio: Res<AudioCapture>,
    // Explicit Without: naming a marker in the query data does not narrow the
    // Node access, so the two &mut Node queries are otherwise a conflict.
    mut pills: Query<(&mut BackgroundColor, &mut Node, &TogglePill), Without<ToggleSection>>,
    mut sections: Query<(&mut Node, &ToggleSection), Without<TogglePill>>,
    mut last: Local<Option<[bool; 3]>>,
) {
    let now = [
        Toggle::Kaleido.state(&kaleido, &choreo, &audio),
        Toggle::Choreographer.state(&kaleido, &choreo, &audio),
        Toggle::Audio.state(&kaleido, &choreo, &audio),
    ];
    if *last == Some(now) {
        return;
    }
    *last = Some(now);
    for (mut bg, mut node, pill) in &mut pills {
        let on = pill.0.state(&kaleido, &choreo, &audio);
        bg.0 = if on { ACCENT } else { PILL_OFF };
        node.justify_content = if on {
            JustifyContent::End
        } else {
            JustifyContent::Start
        };
    }
    for (mut node, section) in &mut sections {
        let on = section.0.state(&kaleido, &choreo, &audio);
        node.display = if on { Display::Flex } else { Display::None };
    }
}

/// Clicking a collapsible card's title (CONTROLS, RECORDING) folds its body
/// away and flips its caret.
fn click_collapse_header(
    mut headers: Query<
        (&Interaction, &mut CollapseHeader, &mut BackgroundColor),
        Changed<Interaction>,
    >,
    mut bodies: Query<(&mut Node, &CollapseBody)>,
    mut carets: Query<(&mut Text, &CollapseCaret)>,
) {
    for (interaction, mut header, mut bg) in &mut headers {
        let color = match interaction {
            Interaction::Pressed => {
                header.open = !header.open;
                for (mut node, body) in &mut bodies {
                    if body.0 == header.section {
                        node.display = if header.open {
                            Display::Flex
                        } else {
                            Display::None
                        };
                    }
                }
                for (mut text, caret) in &mut carets {
                    if caret.0 == header.section {
                        *text = Text::new(caret_glyph(header.open));
                    }
                }
                ROW_HOVER
            }
            Interaction::Hovered => ROW_HOVER,
            Interaction::None => Color::NONE,
        };
        if bg.0 != color {
            bg.0 = color;
        }
    }
}

/// The floating REC indicator: shown while a take runs, dot pulsing,
/// elapsed time ticking.
fn update_rec_chip(
    recorder: Res<Recorder>,
    time: Res<Time>,
    mut chip: Query<&mut Node, With<RecChip>>,
    mut dot: Query<&mut BackgroundColor, With<RecDot>>,
    mut label: Query<&mut Text, With<RecTime>>,
    mut state: Local<(f32, u32)>,
) {
    let active = recorder.is_active();
    let display = if active { Display::Flex } else { Display::None };
    for mut node in &mut chip {
        if node.display != display {
            node.display = display;
        }
    }
    let (elapsed, last_sec) = &mut *state;
    if !active {
        (*elapsed, *last_sec) = (0.0, u32::MAX);
        return;
    }
    *elapsed += time.delta_secs();
    // ~1 Hz breathe so the dot reads as live rather than static decoration.
    let pulse = 0.55 + 0.45 * (*elapsed * std::f32::consts::TAU).sin();
    for mut bg in &mut dot {
        bg.0 = REC_RED.with_alpha(pulse);
    }
    let sec = *elapsed as u32;
    if *last_sec != sec {
        *last_sec = sec;
        for mut text in &mut label {
            *text = Text::new(format!("{}:{:02}", sec / 60, sec % 60));
        }
    }
}

/// Bloom follows the slider; with audio reactivity on, bass and beats pump it
/// every frame. Once the levels have decayed to idle, only a slider change
/// re-applies. At zero the component is removed entirely: Bevy's bloom node
/// only runs for views that carry it, so slider 0 skips the whole
/// downsample/upsample mip chain instead of computing a no-op blur.
fn apply_bloom(
    settings: Res<Settings>,
    audio: Res<AudioLevels>,
    // ParticleCamera keeps this off the native present camera, which only
    // composites the finished scene under the UI.
    mut cameras: Query<(Entity, Option<&mut Bloom>), (With<Camera>, With<ParticleCamera>)>,
    mut commands: Commands,
) {
    if audio.is_idle() && !settings.is_changed() {
        return;
    }
    let target = settings.bloom * (1.0 + audio.bass * 1.1 + audio.beat * 0.6);
    for (entity, bloom) in &mut cameras {
        match bloom {
            Some(mut b) if target > 0.0 => {
                if b.intensity != target {
                    b.intensity = target;
                }
            }
            Some(_) => {
                commands.entity(entity).remove::<Bloom>();
            }
            None if target > 0.0 => {
                commands.entity(entity).insert(Bloom {
                    intensity: target,
                    ..Bloom::NATURAL
                });
            }
            None => {}
        }
    }
}
