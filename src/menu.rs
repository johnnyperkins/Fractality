// In-app settings menu (native bevy_ui, no extra deps). Toggle with Esc or M.
// Each row is a label + draggable slider + value readout driven by the Settings
// resource, which the sim reads live every frame (see update_params in main.rs).

use bevy::core_pipeline::bloom::Bloom;
use bevy::prelude::*;
use bevy::ui::RelativeCursorPosition;

use crate::particles::MAX_PARTICLES;

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
}

/// Rows, in display order.
const SETTINGS: [Setting; 7] = [
    Setting::ParticleCount,
    Setting::Detail,
    Setting::FlowSpeed,
    Setting::AlignForce,
    Setting::Brightness,
    Setting::DotSize,
    Setting::Bloom,
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

const CONTROLS: [&str; 6] = [
    "W / A / S / D  pan",
    "Scroll  zoom at cursor",
    "Left hold  blast   Right hold  vortex",
    "Space  pause",
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
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(2.0),
                    margin: UiRect::top(Val::Px(10.0)),
                    ..default()
                },
                children![
                    (
                        Text::new("CONTROLS"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.7, 0.75, 0.85)),
                    ),
                    control_bundle(CONTROLS[0]),
                    control_bundle(CONTROLS[1]),
                    control_bundle(CONTROLS[2]),
                    control_bundle(CONTROLS[3]),
                    control_bundle(CONTROLS[4]),
                    control_bundle(CONTROLS[5]),
                ],
            ));
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
    widgets: Query<&Interaction, Or<(With<MenuRoot>, With<SliderTrack>)>>,
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

fn apply_bloom(settings: Res<Settings>, mut bloom: Query<&mut Bloom>) {
    if !settings.is_changed() {
        return;
    }
    for mut b in &mut bloom {
        b.intensity = settings.bloom;
    }
}
