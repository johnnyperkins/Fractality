// In-app settings menu (native bevy_ui, no extra deps). Toggle with Esc or M.
// Each row is a label + [-] value [+] triple driven by the Settings resource,
// which the sim reads live every frame (see update_params in main.rs).

use bevy::core_pipeline::bloom::Bloom;
use bevy::prelude::*;

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
    Brightness,
    DotSize,
    Bloom,
}

/// Rows, in display order.
const SETTINGS: [Setting; 6] = [
    Setting::ParticleCount,
    Setting::Detail,
    Setting::FlowSpeed,
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
            Setting::Brightness => "Brightness",
            Setting::DotSize => "Dot size",
            Setting::Bloom => "Bloom",
        }
    }

    /// (step, min, max) in the setting's own units.
    fn range(self) -> (f32, f32, f32) {
        match self {
            Setting::ParticleCount => (500_000.0, 100_000.0, MAX_PARTICLES as f32),
            Setting::Detail => (0.5, 0.5, 8.0),
            Setting::FlowSpeed => (0.02, 0.0, 0.4),
            Setting::Brightness => (0.15, 0.1, 4.0),
            Setting::DotSize => (0.1, 0.1, 4.0),
            Setting::Bloom => (0.05, 0.0, 1.0),
        }
    }

    fn get(self, s: &Settings) -> f32 {
        match self {
            Setting::ParticleCount => s.particle_count as f32,
            Setting::Detail => s.detail,
            Setting::FlowSpeed => s.flow_speed,
            Setting::Brightness => s.brightness,
            Setting::DotSize => s.dot_px,
            Setting::Bloom => s.bloom,
        }
    }

    fn set(self, s: &mut Settings, v: f32) {
        match self {
            Setting::ParticleCount => s.particle_count = v.round() as u32,
            Setting::Detail => s.detail = v,
            Setting::FlowSpeed => s.flow_speed = v,
            Setting::Brightness => s.brightness = v,
            Setting::DotSize => s.dot_px = v,
            Setting::Bloom => s.bloom = v,
        }
    }

    fn adjust(self, s: &mut Settings, dir: f32) {
        let (step, lo, hi) = self.range();
        let v = (self.get(s) + step * dir).clamp(lo, hi);
        self.set(s, v);
    }

    fn format(self, s: &Settings) -> String {
        let v = self.get(s);
        match self {
            Setting::ParticleCount => {
                if v >= 1_000_000.0 {
                    format!("{:.2}M", v / 1_000_000.0)
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

#[derive(Component)]
struct MenuRoot;

#[derive(Component)]
struct SettingValue(Setting);

#[derive(Component)]
struct AdjustButton {
    setting: Setting,
    dir: f32,
}

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuOpen>()
            .add_systems(Startup, build_menu)
            .add_systems(
                Update,
                (toggle_menu, button_system, update_values, apply_bloom),
            );
    }
}

fn label_bundle(setting: Setting) -> impl Bundle {
    (
        Text::new(setting.label()),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::srgb(0.80, 0.85, 0.95)),
        Node {
            width: Val::Px(150.0),
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
            width: Val::Px(80.0),
            justify_content: JustifyContent::Center,
            ..default()
        },
        SettingValue(setting),
    )
}

fn button_bundle(setting: Setting, dir: f32) -> impl Bundle {
    (
        Button,
        Node {
            width: Val::Px(30.0),
            height: Val::Px(28.0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(BUTTON_IDLE),
        AdjustButton { setting, dir },
        children![(
            Text::new(if dir < 0.0 { "-" } else { "+" }),
            TextFont {
                font_size: 20.0,
                ..default()
            },
            TextColor(Color::WHITE),
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
            button_bundle(setting, -1.0),
            value_bundle(setting),
            button_bundle(setting, 1.0),
        ],
    )
}

const BUTTON_IDLE: Color = Color::srgb(0.18, 0.20, 0.28);
const BUTTON_HOVER: Color = Color::srgb(0.30, 0.34, 0.46);
const BUTTON_PRESS: Color = Color::srgb(0.45, 0.55, 0.85);

fn build_menu(mut commands: Commands) {
    commands.spawn((
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
        MenuRoot,
        children![
            (
                Text::new("SETTINGS"),
                TextFont {
                    font_size: 22.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ),
            row_bundle(Setting::ParticleCount),
            row_bundle(Setting::Detail),
            row_bundle(Setting::FlowSpeed),
            row_bundle(Setting::Brightness),
            row_bundle(Setting::DotSize),
            row_bundle(Setting::Bloom),
            (
                Text::new("Esc / M to close"),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.5, 0.55, 0.65)),
            ),
        ],
    ));
    // Silence unused-const in case SETTINGS drifts from the explicit rows above.
    let _ = SETTINGS;
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

fn button_system(
    mut settings: ResMut<Settings>,
    mut buttons: Query<
        (&Interaction, &AdjustButton, &mut BackgroundColor),
        Changed<Interaction>,
    >,
) {
    for (interaction, btn, mut bg) in &mut buttons {
        match interaction {
            Interaction::Pressed => {
                btn.setting.adjust(&mut settings, btn.dir);
                bg.0 = BUTTON_PRESS;
            }
            Interaction::Hovered => bg.0 = BUTTON_HOVER,
            Interaction::None => bg.0 = BUTTON_IDLE,
        }
    }
}

fn update_values(
    open: Res<MenuOpen>,
    settings: Res<Settings>,
    mut values: Query<(&mut Text, &SettingValue)>,
) {
    if !open.0 {
        return;
    }
    for (mut text, sv) in &mut values {
        *text = Text::new(sv.0.format(&settings));
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
