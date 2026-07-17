// In-app settings menu (native bevy_ui, no extra deps). Toggle with Esc or M.
// Each row is a label + draggable slider + value readout driven by the Settings
// resource, which the sim reads live every frame (see update_params in main.rs).

use bevy::core_pipeline::bloom::Bloom;
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::audio::{AudioCapture, AudioLevels};
use crate::particles::MAX_PARTICLES;
use crate::{ColorMode, FractalType, COLOR_MODES, FRACTAL_MODES};

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
    /// Stereo pan push strength.
    pub audio_stereo: f32,
    /// Mid-driven flow speed boost.
    pub audio_flow: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            particle_count: 2_000_000,
            detail: 1.0,
            flow_speed: 0.081,
            align_force: 15.0,
            brightness: 1.1,
            dot_px: 1.0,
            bloom: 0.3,
            trail: 0.3,
            audio_pulse: 1.0,
            audio_flash: 1.0,
            audio_glow: 1.0,
            audio_breathe: 1.0,
            audio_morph: 1.0,
            audio_stereo: 1.0,
            audio_flow: 1.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Setting {
    ParticleCount,
    Detail,
    FlowSpeed,
    AlignForce,
    Brightness,
    DotSize,
    Bloom,
    Trail,
    AudioPulse,
    AudioFlash,
    AudioGlow,
    AudioBreathe,
    AudioMorph,
    AudioStereo,
    AudioFlow,
}

/// Rows, in display order.
const SETTINGS: [Setting; 8] = [
    Setting::ParticleCount,
    Setting::Detail,
    Setting::FlowSpeed,
    Setting::AlignForce,
    Setting::Brightness,
    Setting::DotSize,
    Setting::Bloom,
    Setting::Trail,
];

/// Audio-effect rows, shown only while audio reactivity is on.
const AUDIO_SETTINGS: [Setting; 7] = [
    Setting::AudioPulse,
    Setting::AudioFlash,
    Setting::AudioGlow,
    Setting::AudioBreathe,
    Setting::AudioMorph,
    Setting::AudioStereo,
    Setting::AudioFlow,
];

impl Setting {
    fn label(self) -> &'static str {
        match self {
            Setting::ParticleCount => "Particles",
            Setting::Detail => "Detail",
            Setting::FlowSpeed => "Flow speed",
            Setting::AlignForce => "Align force",
            Setting::Brightness => "Brightness",
            Setting::DotSize => "Dot size",
            Setting::Bloom => "Bloom",
            Setting::Trail => "Trails",
            Setting::AudioPulse => "Pulse",
            Setting::AudioFlash => "Flash",
            Setting::AudioGlow => "Spec glow",
            Setting::AudioBreathe => "Breathe",
            Setting::AudioMorph => "Julia morph",
            Setting::AudioStereo => "Stereo push",
            Setting::AudioFlow => "Flow boost",
        }
    }

    /// (min, max) in the setting's own units.
    fn range(self) -> (f32, f32) {
        match self {
            Setting::ParticleCount => (100_000.0, MAX_PARTICLES as f32),
            Setting::Detail => (0.5, 8.0),
            Setting::FlowSpeed => (0.0, 0.4),
            Setting::AlignForce => (0.0, 200.0),
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
            | Setting::AudioStereo
            | Setting::AudioFlow => (0.0, 2.0),
        }
    }

    fn get(self, s: &Settings) -> f32 {
        match self {
            Setting::ParticleCount => s.particle_count as f32,
            Setting::Detail => s.detail,
            Setting::FlowSpeed => s.flow_speed,
            Setting::AlignForce => s.align_force,
            Setting::Brightness => s.brightness,
            Setting::DotSize => s.dot_px,
            Setting::Bloom => s.bloom,
            Setting::Trail => s.trail,
            Setting::AudioPulse => s.audio_pulse,
            Setting::AudioFlash => s.audio_flash,
            Setting::AudioGlow => s.audio_glow,
            Setting::AudioBreathe => s.audio_breathe,
            Setting::AudioMorph => s.audio_morph,
            Setting::AudioStereo => s.audio_stereo,
            Setting::AudioFlow => s.audio_flow,
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
            Setting::Brightness => s.brightness = v,
            Setting::DotSize => s.dot_px = v,
            Setting::Bloom => s.bloom = v,
            Setting::Trail => s.trail = v,
            Setting::AudioPulse => s.audio_pulse = v,
            Setting::AudioFlash => s.audio_flash = v,
            Setting::AudioGlow => s.audio_glow = v,
            Setting::AudioBreathe => s.audio_breathe = v,
            Setting::AudioMorph => s.audio_morph = v,
            Setting::AudioStereo => s.audio_stereo = v,
            Setting::AudioFlow => s.audio_flow = v,
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
            _ => format!("{v:.2}"),
        }
    }
}

/// Whether the panel is currently shown.
#[derive(Resource, Default)]
pub struct MenuOpen(pub bool);

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

/// Clickable "Color mode" row; clicking cycles the palette (same as C).
#[derive(Component)]
struct ColorModeRow;

/// The text showing the active palette name.
#[derive(Component)]
struct ColorModeValue;

/// Clickable "Fractal" row; clicking cycles the fractal type (same as F).
#[derive(Component)]
struct FractalRow;

/// The text showing the active fractal name.
#[derive(Component)]
struct FractalValue;

/// Clickable "Audio react" row; clicking toggles audio reactivity (same as V).
#[derive(Component)]
struct AudioRow;

/// The text showing whether audio reactivity is on.
#[derive(Component)]
struct AudioValue;

/// Marker on every clickable menu row, so track_pointer_over_menu shields the
/// sim from their clicks without enumerating each row type.
#[derive(Component)]
struct MenuInteractive;

/// Container for the audio-effect sliders; visible only while audio
/// reactivity is on.
#[derive(Component)]
struct AudioSection;

#[derive(Component)]
struct SliderFill(Setting);

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
                    click_color_mode,
                    update_color_mode_text,
                    click_fractal,
                    update_fractal_text,
                    click_audio,
                    sync_audio_ui,
                    apply_bloom,
                ),
            );
    }
}

const TRACK_IDLE: Color = Color::srgb(0.18, 0.20, 0.28);
const TRACK_HOVER: Color = Color::srgb(0.24, 0.27, 0.37);
const FILL_COLOR: Color = Color::srgb(0.45, 0.55, 0.85);

fn label_bundle(setting: Setting) -> impl Bundle {
    (
        Text::new(setting.label()),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.80, 0.85, 0.95)),
        Node {
            width: Val::Px(110.0),
            ..default()
        },
    )
}

fn value_bundle(setting: Setting) -> impl Bundle {
    (
        Text::new(String::new()),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(1.0, 0.88, 0.5)),
        Node {
            width: Val::Px(60.0),
            justify_content: JustifyContent::End,
            ..default()
        },
        SettingValue(setting),
    )
}

fn slider_bundle(setting: Setting) -> impl Bundle {
    (
        Button,
        RelativeCursorPosition::default(),
        Node {
            width: Val::Px(190.0),
            height: Val::Px(16.0),
            padding: UiRect::all(Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(TRACK_IDLE),
        BorderRadius::all(Val::Px(4.0)),
        SliderTrack(setting),
        children![(
            Node {
                width: Val::Percent(50.0),
                height: Val::Percent(100.0),
                ..default()
            },
            BackgroundColor(FILL_COLOR),
            BorderRadius::all(Val::Px(3.0)),
            SliderFill(setting),
        )],
    )
}

fn row_bundle(setting: Setting) -> impl Bundle {
    (
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        },
        children![
            label_bundle(setting),
            slider_bundle(setting),
            value_bundle(setting),
        ],
    )
}

/// Clickable label + value row (fractal / color mode / audio react). The row
/// marker drives the click handler, the value marker the label sync.
fn value_row(
    label: &'static str,
    initial: &'static str,
    row_marker: impl Component,
    value_marker: impl Component,
) -> impl Bundle {
    (
        Button,
        Node {
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: Val::Px(10.0),
            ..default()
        },
        MenuInteractive,
        row_marker,
        children![
            (
                Text::new(label),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(0.80, 0.85, 0.95)),
                Node {
                    width: Val::Px(110.0),
                    ..default()
                },
            ),
            (
                Text::new(initial),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.88, 0.5)),
                value_marker,
            ),
        ],
    )
}

fn heading_bundle(text: &str) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font_size: 22.0,
            ..default()
        },
        TextColor(Color::WHITE),
    )
}

fn control_bundle(text: &str) -> impl Bundle {
    (
        Text::new(text),
        TextFont {
            font_size: 13.0,
            ..default()
        },
        TextColor(Color::srgb(0.5, 0.55, 0.65)),
    )
}

const CONTROLS: [&str; 11] = [
    "W / A / S / D  pan",
    "Scroll  zoom at cursor",
    "Left hold  blast   Right hold  vortex",
    "Space  dissolve",
    "Z  auto-zoom dive   C  color mode",
    "F  fractal type",
    "V  audio reactivity",
    "P  screenshot",
    "Shift+1..9  save view   1..9  fly to it",
    "R  reset view",
    "Esc / M  toggle menu",
];

fn build_menu(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(20.0),
                top: Val::Px(20.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(16.0)),
                row_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.05, 0.09, 0.88)),
            Visibility::Hidden,
            // Interaction on the root reports Hovered anywhere over the panel,
            // which track_pointer_over_menu uses to shield the sim from clicks.
            Interaction::default(),
            MenuRoot,
        ))
        .with_children(|parent| {
            parent.spawn(heading_bundle("SETTINGS"));
            for setting in SETTINGS {
                parent.spawn(row_bundle(setting));
            }
            parent.spawn(value_row("Fractal", FRACTAL_MODES[0], FractalRow, FractalValue));
            parent.spawn(value_row(
                "Color mode",
                COLOR_MODES[0],
                ColorModeRow,
                ColorModeValue,
            ));
            parent.spawn(value_row("Audio react", "off", AudioRow, AudioValue));
            parent
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(8.0),
                        margin: UiRect::top(Val::Px(4.0)),
                        ..default()
                    },
                    Visibility::Hidden,
                    AudioSection,
                ))
                .with_children(|col| {
                    col.spawn((
                        Text::new("AUDIO FX"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.7, 0.75, 0.85)),
                    ));
                    for setting in AUDIO_SETTINGS {
                        col.spawn(row_bundle(setting));
                    }
                });
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    margin: UiRect::top(Val::Px(10.0)),
                    ..default()
                })
                .with_children(|col| {
                    col.spawn((
                        Text::new("CONTROLS"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.7, 0.75, 0.85)),
                    ));
                    for line in CONTROLS {
                        col.spawn(control_bundle(line));
                    }
                });
        });
}

fn toggle_menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut open: ResMut<MenuOpen>,
    mut root: Query<&mut Visibility, With<MenuRoot>>,
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
    }
}

/// Cursor over the panel (root or any slider reports interaction), or a drag
/// in progress (a track stays Pressed even after the cursor leaves it).
fn track_pointer_over_menu(
    mut over: ResMut<PointerOverMenu>,
    widgets: Query<
        &Interaction,
        Or<(With<MenuRoot>, With<SliderTrack>, With<MenuInteractive>)>,
    >,
) {
    over.0 = widgets.iter().any(|i| *i != Interaction::None);
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
        match interaction {
            Interaction::Pressed => {
                bg.0 = TRACK_HOVER;
                if let Some(pos) = rel.normalized {
                    let (lo, hi) = track.0.range();
                    track.0.set(&mut settings, lo + pos.x.clamp(0.0, 1.0) * (hi - lo));
                }
            }
            Interaction::Hovered => bg.0 = TRACK_HOVER,
            Interaction::None => bg.0 = TRACK_IDLE,
        }
    }
}

fn update_sliders(
    open: Res<MenuOpen>,
    settings: Res<Settings>,
    mut values: Query<(&mut Text, &SettingValue)>,
    mut fills: Query<(&mut Node, &SliderFill)>,
) {
    if !open.0 {
        return;
    }
    for (mut text, sv) in &mut values {
        *text = Text::new(sv.0.format(&settings));
    }
    for (mut node, fill) in &mut fills {
        node.width = Val::Percent(fill.0.fraction(&settings) * 100.0);
    }
}

/// Clicking the row cycles the palette, same as the C key.
fn click_color_mode(
    mut mode: ResMut<ColorMode>,
    rows: Query<&Interaction, (Changed<Interaction>, With<ColorModeRow>)>,
) {
    for interaction in &rows {
        if *interaction == Interaction::Pressed {
            mode.0 = (mode.0 + 1) % COLOR_MODES.len() as u32;
        }
    }
}

/// Keep the palette name in sync however the mode changes (click or C key).
fn update_color_mode_text(
    mode: Res<ColorMode>,
    mut texts: Query<&mut Text, With<ColorModeValue>>,
) {
    if !mode.is_changed() {
        return;
    }
    for mut text in &mut texts {
        *text = Text::new(COLOR_MODES[mode.0 as usize]);
    }
}

/// Clicking the row cycles the fractal type, same as the F key.
fn click_fractal(
    mut fractal: ResMut<FractalType>,
    rows: Query<&Interaction, (Changed<Interaction>, With<FractalRow>)>,
) {
    for interaction in &rows {
        if *interaction == Interaction::Pressed {
            fractal.0 = (fractal.0 + 1) % FRACTAL_MODES.len() as u32;
        }
    }
}

/// Keep the fractal name in sync however the type changes (click or F key).
fn update_fractal_text(
    fractal: Res<FractalType>,
    mut texts: Query<&mut Text, With<FractalValue>>,
) {
    if !fractal.is_changed() {
        return;
    }
    for mut text in &mut texts {
        *text = Text::new(FRACTAL_MODES[fractal.0 as usize]);
    }
}

/// Clicking the row toggles audio reactivity, same as the V key.
fn click_audio(
    mut audio: ResMut<AudioCapture>,
    rows: Query<&Interaction, (Changed<Interaction>, With<AudioRow>)>,
) {
    for interaction in &rows {
        if *interaction == Interaction::Pressed {
            audio.enabled = !audio.enabled;
        }
    }
}

/// Keep the on/off label and the AUDIO FX section in sync however the toggle
/// happens (click or V). AudioCapture changes only on real transitions, so
/// is_changed suffices. Visibility::Inherited keeps the section tied to the
/// panel's own visibility.
fn sync_audio_ui(
    audio: Res<AudioCapture>,
    mut texts: Query<&mut Text, With<AudioValue>>,
    mut sections: Query<&mut Visibility, With<AudioSection>>,
) {
    if !audio.is_changed() {
        return;
    }
    for mut text in &mut texts {
        *text = Text::new(if audio.enabled { "on" } else { "off" });
    }
    for mut vis in &mut sections {
        *vis = if audio.enabled {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Bloom follows the slider; with audio reactivity on, bass and beats pump it
/// every frame. Once the levels have decayed to idle, only a slider change
/// re-applies.
fn apply_bloom(settings: Res<Settings>, audio: Res<AudioLevels>, mut bloom: Query<&mut Bloom>) {
    if audio.is_idle() && !settings.is_changed() {
        return;
    }
    let target = settings.bloom * (1.0 + audio.bass * 1.1 + audio.beat * 0.6);
    for mut b in &mut bloom {
        b.intensity = target;
    }
}
