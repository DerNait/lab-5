use nalgebra_glm::{Vec3, Mat4};
use nalgebra_glm::normalize;
use minifb::{Key, Window, WindowOptions};
use std::time::{Duration, Instant};
use std::f32::consts::PI;

mod framebuffer;
mod triangle;
mod line;
mod vertex;
mod obj;
mod color;
mod fragment;
mod shaders;
mod noise;

use framebuffer::Framebuffer;
use vertex::Vertex;
use obj::Obj;
use triangle::triangle_with_shader;
use shaders::{
    vertex_shader, Uniforms, FragmentShader,
    ProceduralLayerShader, NoiseParams, NoiseType, VoronoiDistance,
    ColorStop, AlphaMode, FlowParams,
};
use color::Color;

// ===================== Cámara simple (ortográfica sobre tu “espacio pantalla”) =====================

struct Camera {
    position: Vec3,   // traslación en tu espacio actual (px)
    rotation: Vec3,   // pitch (x), yaw (y), roll (z)
}

impl Camera {
    fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    fn view_matrix(&self) -> Mat4 {
        let (sx, cx) = self.rotation.x.sin_cos();
        let (sy, cy) = self.rotation.y.sin_cos();
        let (sz, cz) = self.rotation.z.sin_cos();

        let rx = Mat4::new(
            1.0, 0.0,  0.0, 0.0,
            0.0, cx,  -sx, 0.0,
            0.0, sx,   cx, 0.0,
            0.0, 0.0,  0.0, 1.0,
        );
        let ry = Mat4::new(
             cy, 0.0, sy, 0.0,
             0.0, 1.0, 0.0, 0.0,
            -sy, 0.0, cy, 0.0,
             0.0, 0.0, 0.0, 1.0,
        );
        let rz = Mat4::new(
            cz, -sz, 0.0, 0.0,
            sz,  cz, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // vista = R^T * T(-pos)  (inversa de la pose de cámara)
        let r = rz * ry * rx;
        let t = Mat4::new(
            1.0, 0.0, 0.0, -self.position.x,
            0.0, 1.0, 0.0, -self.position.y,
            0.0, 0.0, 1.0, -self.position.z,
            0.0, 0.0, 0.0,  1.0,
        );

        r.transpose() * t
    }
}

// ===================== Helpers de modelo =====================

pub struct ModelMatrices { pub base: Mat4, pub overlay: Mat4 }

fn create_model_matrix(translation: Vec3, scale: f32, rotation: Vec3) -> Mat4 {
    let (sin_x, cos_x) = rotation.x.sin_cos();
    let (sin_y, cos_y) = rotation.y.sin_cos();
    let (sin_z, cos_z) = rotation.z.sin_cos();

    let rotation_matrix_x = Mat4::new(
        1.0,  0.0,    0.0,   0.0,
        0.0,  cos_x, -sin_x, 0.0,
        0.0,  sin_x,  cos_x, 0.0,
        0.0,  0.0,    0.0,   1.0,
    );
    let rotation_matrix_y = Mat4::new(
        cos_y,  0.0,  sin_y, 0.0,
        0.0,    1.0,  0.0,   0.0,
        -sin_y, 0.0,  cos_y, 0.0,
        0.0,    0.0,  0.0,   1.0,
    );
    let rotation_matrix_z = Mat4::new(
        cos_z, -sin_z, 0.0, 0.0,
        sin_z,  cos_z, 0.0, 0.0,
        0.0,    0.0,  1.0, 0.0,
        0.0,    0.0,  0.0, 1.0,
    );

    let rotation_matrix = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x;

    let transform_matrix = Mat4::new(
        scale, 0.0,   0.0,   translation.x,
        0.0,   scale, 0.0,   translation.y,
        0.0,   0.0,   scale, translation.z,
        0.0,   0.0,   0.0,   1.0,
    );

    transform_matrix * rotation_matrix
}

// ===================== Render passes =====================

fn render_pass(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &dyn FragmentShader
) {
    let (positions, normals, uvs) = obj.mesh_buffers();
    let mut all_fragments = Vec::new();

    obj.for_each_face(|i0, i1, i2| {
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let n0 = normals.get(i0).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n1 = normals.get(i1).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n2 = normals.get(i2).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));

        let t0 = uvs.get(i0).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t1 = uvs.get(i1).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t2 = uvs.get(i2).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));

        let v0 = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1 = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2 = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);

        all_fragments.extend(triangle_with_shader(&v0, &v1, &v2, shader, uniforms));
    });

    for fragment in all_fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            framebuffer.draw_rgba(x, y, fragment.depth, fragment.color.to_hex(), 1.0);
        }
    }
}

fn render_pass_alpha(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &impl FragmentShader,
    z_bias: f32,  // << sesgo de profundidad (negativo = más cerca de la cámara)
) {
    let (positions, normals, uvs) = obj.mesh_buffers();
    let mut all_fragments = Vec::new();

    obj.for_each_face(|i0, i1, i2| {
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let n0 = normals.get(i0).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n1 = normals.get(i1).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));
        let n2 = normals.get(i2).cloned().unwrap_or(nalgebra_glm::Vec3::new(0.0, 1.0, 0.0));

        let t0 = uvs.get(i0).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t1 = uvs.get(i1).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));
        let t2 = uvs.get(i2).cloned().unwrap_or(nalgebra_glm::Vec2::new(0.0, 0.0));

        let v0 = vertex_shader(&Vertex::new(p0, n0, t0), uniforms);
        let v1 = vertex_shader(&Vertex::new(p1, n1, t1), uniforms);
        let v2 = vertex_shader(&Vertex::new(p2, n2, t2), uniforms);

        all_fragments.extend(crate::triangle::triangle_with_shader(&v0, &v1, &v2, shader, uniforms));
    });

    for frag in all_fragments {
        let x = frag.position.x as usize;
        let y = frag.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            framebuffer.draw_rgba(x, y, frag.depth + z_bias, frag.color.to_hex(), frag.alpha);
        }
    }
}

// ===================== Sistema de lunas =====================

struct Moon {
    obj: Obj,
    scale_rel: f32,
    orbit_px: f32,
    orbit_depth_px: f32,
    orbit_speed: f32,
    phase0: f32,
    tilt: Vec3,
    shader: Box<dyn FragmentShader + Send + Sync>,
    seed: i32,
}

impl Moon {
    fn model_matrix(&self, planet_translation: Vec3, planet_scale: f32, t: f32) -> Mat4 {
        let angle = self.phase0 + t * self.orbit_speed;
        let dx = angle.cos() * self.orbit_px;
        let dz = angle.sin() * self.orbit_depth_px;

        let tr = Vec3::new(
            planet_translation.x + dx,
            planet_translation.y + 0.0,
            planet_translation.z + dz
        );

        let spin = Vec3::new(self.tilt.x, self.tilt.y + t * 0.35, self.tilt.z);
        create_model_matrix(tr, planet_scale * self.scale_rel, spin)
    }
}

fn main() {
    let window_width = 800;
    let window_height = 600;
    let framebuffer_width = 800;
    let framebuffer_height = 600;
    let frame_delay = Duration::from_millis(16);

    let mut framebuffer = Framebuffer::new(framebuffer_width, framebuffer_height);
    let mut window = Window::new(
        "Sistema Epsilon Eridani - Mini Renderizador 3D",
        window_width,
        window_height,
        WindowOptions::default(),
    ).unwrap();

    window.set_position(500, 500);
    window.update();

    framebuffer.set_background_color(0x000000);

    // ===== Carga de modelos =====
    let obj = Obj::load("assets/models/Planet.obj").expect("Failed to load obj");
    let rings_obj = Obj::load("assets/models/PlanetRing.obj").expect("Failed to load rings obj");
    let moon_obj = Obj::load("assets/models/Planet.obj").expect("Failed to load moon sphere");

    // ===== Ajuste de escala/traslación del planeta base =====
    let (min_v, max_v) = obj.bounds();
    let size = max_v - min_v;
    let center = (min_v + max_v) * 0.5;

    let target_w = framebuffer_width as f32 * 0.8;
    let target_h = framebuffer_height as f32 * 0.8;
    let sx = if size.x.abs() < 1e-6 { 1.0 } else { target_w / size.x.abs() };
    let sy = if size.y.abs() < 1e-6 { 1.0 } else { target_h / size.y.abs() };
    let mut scale = sx.min(sy);
    scale *= 0.45; // un poco de margen

    // Traslación base (coloca el planeta en el centro visual)
    let mut translation = Vec3::new(
        (framebuffer_width as f32) * 0.5 - center.x * scale,
        (framebuffer_height as f32) * 0.5 - center.y * scale,
        -center.z * scale
    );
    let mut rotation = Vec3::new(0.0, 0.0, 0.0);

    // ===== Cámara =====
    let mut camera = Camera::new();

    // ===== Shaders base (Tierra / Gas) =====
    let terrain_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 3.0,
            octaves: 5,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.35,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false,
            time_speed: 1.0,
            animate_spin: false,
            spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.35, color: Color::from_hex(0x1B3494) },
            ColorStop { threshold: 0.48, color: Color::from_hex(0x203FB0) },
            ColorStop { threshold: 0.50, color: Color::from_hex(0x4B87DB) },
            ColorStop { threshold: 0.51, color: Color::from_hex(0xA4957F) },
            ColorStop { threshold: 0.52, color: Color::from_hex(0x88B04B) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0x668736) },
            ColorStop { threshold: 0.70, color: Color::from_hex(0x597A2A) },
        ],
        color_hardness: 0.25,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    };

    let clouds_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Value,
            scale: 5.0,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.25,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: true,
            time_speed: 0.1,
            animate_spin: true,
            spin_speed: 0.05,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) },
            ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) },
        ],
        color_hardness: 0.0,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.8,
        light_max: 1.0,
        alpha_mode: AlphaMode::Threshold {
            threshold: 0.50,
            sharpness: 6.0,
            coverage_bias: 0.05,
            invert: false,
        },
    };

    let gas_palette = vec![
        ColorStop { threshold: 0.00, color: Color::from_hex(0x734D1E) },
        ColorStop { threshold: 0.20, color: Color::from_hex(0x6E5034) },
        ColorStop { threshold: 0.40, color: Color::from_hex(0xB58C5A) },
        ColorStop { threshold: 0.60, color: Color::from_hex(0xD9B98B) },
        ColorStop { threshold: 0.80, color: Color::from_hex(0xA57442) },
        ColorStop { threshold: 1.00, color: Color::from_hex(0xE4D1B5) },
    ];

    let gas_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0, octaves: 4, lacunarity: 2.0, gain: 0.5,
            cell_size: 0.35, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false,  spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 4.0,
            band_contrast: 1.0,
            lat_shear: 0.2,
            turb_scale: 10.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 0.35,
            flow: FlowParams {
                enabled: true,
                flow_scale: 3.0,
                strength: 0.04,
                time_speed: 0.6,
                jets_base_speed: 0.12,
                jets_frequency: 6.0,
                phase_amp: 3.0,
            },
        },
        color_stops: gas_palette,
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    };

    // ===== PLANETA ROCOso tipo Marte =====
    let mars_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Perlin,
            scale: 4.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.40,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false,
            time_speed: 0.0,
            animate_spin: false,
            spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 6.0,
            turb_octaves: 3,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.2,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.10, color: Color::from_hex(0x3D1F1A) }, // sombras rocas
            ColorStop { threshold: 0.35, color: Color::from_hex(0x6B2F26) }, // rojo oscuro
            ColorStop { threshold: 0.60, color: Color::from_hex(0xA94432) }, // rojo óxido
            ColorStop { threshold: 0.85, color: Color::from_hex(0xD27A4A) }, // polvo
            ColorStop { threshold: 1.00, color: Color::from_hex(0xF3C9A2) }, // zonas claras
        ],
        color_hardness: 0.35,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.35,
        light_max: 1.05,
        alpha_mode: AlphaMode::Opaque,
    };

     let mars_clouds_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::Value,
            scale: 2.0,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.25,
            w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: true,
            time_speed: 0.05,
            animate_spin: true,
            spin_speed: 0.02,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.0, color: Color::from_hex(0xEDEDED) },
            ColorStop { threshold: 1.0, color: Color::from_hex(0xFFFFFF) },
        ],
        color_hardness: 0.0,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.8,
        light_max: 1.0,
        alpha_mode: AlphaMode::Threshold {
            threshold: 0.60,
            sharpness: 6.0,
            coverage_bias: 0.05,
            invert: false,
        },
    };

    // ===== PLANETA tipo Urano (pastel, bandas suaves azuladas) =====
    let uranus_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0, octaves: 3, lacunarity: 2.0, gain: 0.55,
            cell_size: 0.35, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: false,  spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 3.0,
            band_contrast: 0.5,
            lat_shear: 0.05,
            turb_scale: 4.0,
            turb_octaves: 3,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.15,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x0A2F4F) },
            ColorStop { threshold: 0.30, color: Color::from_hex(0x145D7A) },
            ColorStop { threshold: 0.60, color: Color::from_hex(0x42A8C8) },
            ColorStop { threshold: 0.85, color: Color::from_hex(0x7FD8E6) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xC5F6FF) },
        ],
        color_hardness: 0.25,
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.1,
        alpha_mode: AlphaMode::Opaque,
    };

    // ===== PLANETA tipo Saturno (gaseoso amarillento, bandas suaves) =====
    let saturn_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,
            scale: 1.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            cell_size: 0.35,
            w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,

            animate_time: false,
            time_speed: 0.0,
            animate_spin: false,
            spin_speed: 0.0,

            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            band_frequency: 5.0,
            band_contrast: 1.0,

            lat_shear: 0.10,

            turb_scale: 1.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.5,
            turb_amp: 0.35,

            flow: FlowParams {
                enabled: true,
                flow_scale: 3.0,
                strength: 0.04,
                time_speed: 0.25,
                jets_base_speed: 0.1,
                jets_frequency: 6.0,
                phase_amp: 3.0,
            },
        },
        color_stops: vec![
            // zona de sombras ligeramente marrón
            ColorStop { threshold: 0.00, color: Color::from_hex(0xEDD198) },
            ColorStop { threshold: 0.18, color: Color::from_hex(0xB48D5A) },

            // crema suave bastante ancho
            ColorStop { threshold: 0.32, color: Color::from_hex(0xD5BE8A) },

            // franja anaranjada finita
            ColorStop { threshold: 0.36, color: Color::from_hex(0xE5A95C) },

            // vuelve a crema más claro
            ColorStop { threshold: 0.44, color: Color::from_hex(0xEFD49C) },

            // otra zona amplia crema muy clara
            ColorStop { threshold: 0.58, color: Color::from_hex(0xF4E0B6) },

            // tramo final casi todo crema muy luminoso
            ColorStop { threshold: 0.72, color: Color::from_hex(0xF6E6C4) },
            ColorStop { threshold: 0.86, color: Color::from_hex(0xFBF0D9) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9E6) },
        ],
        // Un poquito más de dureza para que se lean las franjas
        color_hardness: 0.24,

        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.50,
        light_max: 1.08,
        alpha_mode: AlphaMode::Opaque,
    };

    // ===== ESTRELLA / SOL =====
    let star_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::BandedGas,

            // Base de ruido
            scale: 1.0,
            octaves: 16,
            lacunarity: 2.2,
            gain: 0.52,
            cell_size: 0.35,
            w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,

            // El "tiempo" principal lo vamos a manejar con el flow,
            // aquí lo dejamos neutro
            animate_time: false,
            time_speed: 0.0,

            // Ligero spin global para que todo gire despacio
            animate_spin: true,
            spin_speed: 0.12,

            // No usamos swirl de anillos aquí
            ring_swirl_amp: 0.0,
            ring_swirl_freq: 8.0,

            // Bandas base (se rompen con turbulencia + flow)
            band_frequency: 20.0,   // pocas bandas grandes
            band_contrast: 0.05,   // no tan duras, más "lava"

            lat_shear: 0.05,       // un pelín de inclinación

            // Turbulencia fuerte para romper las bandas
            turb_scale: 8.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 1.4,         // alto = formas muy retorcidas

            // Flowmap: aquí está la magia de la "lava" que fluye
            flow: FlowParams {
                enabled: true,
                flow_scale: 2.5,   // tamaño de los remolinos
                strength: 0.09,    // cuánto se desplazan lon/lat
                time_speed: 0.7,   // velocidad de animación del flow
                jets_base_speed: 0.04,
                jets_frequency: 5.0,
                phase_amp: 2.0,    // qué tan fuerte se siente la transición de fases
            },
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0xA33600) }, // sombras muy oscuras
            ColorStop { threshold: 0.25, color: Color::from_hex(0x7A1400) }, // rojo profundo
            ColorStop { threshold: 0.50, color: Color::from_hex(0xE04800) }, // naranja brillante
            ColorStop { threshold: 0.75, color: Color::from_hex(0xFFC640) }, // amarillo intenso
            ColorStop { threshold: 1.00, color: Color::from_hex(0xFFF9D0) }, // casi blanco
        ],
        color_hardness: 0.15,         // bordes algo marcados entre "placas" de lava
        lighting_enabled: false,      // autoiluminado
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0,
        light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    };

    // ===== Shader de anillos =====
    let rings_shader = ProceduralLayerShader {
        noise: NoiseParams {
            kind: NoiseType::RadialGradient {
                inner: 0.70,   // fracción del borde exterior
                outer: 1.0,    // 1.0 == borde exterior del mesh
                invert: false,
                bias: 0.10,
                gamma: 1.0,
            },
            // resto sin uso
            scale: 1.0, octaves: 1, lacunarity: 2.0, gain: 0.5,
            cell_size: 1.0, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
            dist: VoronoiDistance::Euclidean,
            animate_time: false, time_speed: 0.0,
            animate_spin: true, spin_speed: 1.2,

            ring_swirl_amp: 0.05,
            ring_swirl_freq: 10.0,

            band_frequency: 0.0,
            band_contrast: 0.0,
            lat_shear: 0.0,
            turb_scale: 0.0,
            turb_octaves: 0,
            turb_lacunarity: 0.0,
            turb_gain: 0.0,
            turb_amp: 0.0,
            flow: FlowParams::default(),
        },
        color_stops: vec![
            ColorStop { threshold: 0.00, color: Color::from_hex(0x0E0F12) },
            ColorStop { threshold: 0.12, color: Color::from_hex(0xDCCEB3) },
            ColorStop { threshold: 0.28, color: Color::from_hex(0x2A2E36) },
            ColorStop { threshold: 0.45, color: Color::from_hex(0xEFE8D8) },
            ColorStop { threshold: 0.62, color: Color::from_hex(0xC0B29A) },
            ColorStop { threshold: 0.80, color: Color::from_hex(0x1F2329) },
            ColorStop { threshold: 1.00, color: Color::from_hex(0x0A0C10) },
        ],
        color_hardness: 0.0,
        lighting_enabled: false,
        light_dir: normalize(&Vec3::new(0.0, 1.0, 0.0)),
        light_min: 1.0, light_max: 1.0,
        alpha_mode: AlphaMode::Opaque,
    };

    // ===== Lunas =====
    let mut moons: Vec<Moon> = vec![
        Moon {
            obj: moon_obj.clone(),
            scale_rel: 0.22,
            orbit_px: 200.0,
            orbit_depth_px: 120.0,
            orbit_speed: 0.5,
            phase0: 0.0,
            tilt: Vec3::new(0.05, 0.0, 0.05),
            shader: Box::new(ProceduralLayerShader {
                noise: NoiseParams {
                    kind: NoiseType::Perlin,
                    scale: 4.0, octaves: 4, lacunarity: 2.0, gain: 0.5,
                    cell_size: 0.35, w1: 1.0, w2: 0.0, w3: 0.0, w4: 0.0,
                    dist: VoronoiDistance::Euclidean,
                    animate_time: false, time_speed: 0.0,
                    animate_spin: false, spin_speed: 0.0,

                    ring_swirl_amp: 0.0,
                    ring_swirl_freq: 8.0,

                    band_frequency: 0.0,
                    band_contrast: 0.0,
                    lat_shear: 0.0,
                    turb_scale: 0.0,
                    turb_octaves: 0,
                    turb_lacunarity: 0.0,
                    turb_gain: 0.0,
                    turb_amp: 0.0,
                    flow: FlowParams::default(),
                },
                color_stops: vec![
                    ColorStop { threshold: 0.30, color: Color::from_hex(0x4D4D4D) },
                    ColorStop { threshold: 0.55, color: Color::from_hex(0x7A7A7A) },
                    ColorStop { threshold: 0.80, color: Color::from_hex(0xB5B5B5) },
                ],
                color_hardness: 0.35,
                lighting_enabled: true,
                light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
                light_min: 0.35,
                light_max: 1.05,
                alpha_mode: AlphaMode::Opaque,
            }),
            seed: 8888,
        },
        Moon {
            obj: moon_obj.clone(),
            scale_rel: 0.18,
            orbit_px: 260.0,
            orbit_depth_px: 290.0,
            orbit_speed: 0.32,
            phase0: 1.2,
            tilt: Vec3::new(0.15, 0.0, -0.05),
            shader: Box::new(ProceduralLayerShader {
                noise: NoiseParams {
                    kind: NoiseType::Voronoi,
                    scale: 2.6, octaves: 3, lacunarity: 2.0, gain: 0.5,
                    cell_size: 0.40, w1: 1.0, w2: 1.0, w3: 1.0, w4: 0.0,
                    dist: VoronoiDistance::Euclidean,
                    animate_time: false, time_speed: 0.0,
                    animate_spin: false, spin_speed: 0.0,

                    ring_swirl_amp: 0.0,
                    ring_swirl_freq: 8.0,

                    band_frequency: 0.0,
                    band_contrast: 0.0,
                    lat_shear: 0.0,
                    turb_scale: 0.0,
                    turb_octaves: 0,
                    turb_lacunarity: 0.0,
                    turb_gain: 0.0,
                    turb_amp: 0.0,
                    flow: FlowParams::default(),
                },
                color_stops: vec![
                    ColorStop { threshold: 0.25, color: Color::from_hex(0x5E5347) },
                    ColorStop { threshold: 0.55, color: Color::from_hex(0x8C7B68) },
                    ColorStop { threshold: 0.85, color: Color::from_hex(0xCBB79F) },
                ],
                color_hardness: 0.25,
                lighting_enabled: true,
                light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
                light_min: 0.35,
                light_max: 1.05,
                alpha_mode: AlphaMode::Opaque,
            }),
            seed: 9999,
        },
    ];

    // ===== Medimos los semiejes del anillo (en espacio OBJ) =====
    let (rmin, rmax) = rings_obj.bounds();
    let ext = rmax - rmin;

    let (ring_a, ring_b, ring_plane_xy) = {
        let ex = ext.x.abs();
        let ey = ext.y.abs();
        let ez = ext.z.abs();

        if ez <= ex.min(ey) {
            let a = ((rmax.x - rmin.x).abs() * 0.5).max(1e-6);
            let b = ((rmax.y - rmin.y).abs() * 0.5).max(1e-6);
            (a, b, true)
        } else if ey <= ex.min(ez) {
            let a = ((rmax.x - rmin.x).abs() * 0.5).max(1e-6);
            let b = ((rmax.z - rmin.z).abs() * 0.5).max(1e-6);
            (a, b, false)
        } else {
            let a = ((rmax.y - rmin.y).abs() * 0.5).max(1e-6);
            let b = ((rmax.z - rmin.z).abs() * 0.5).max(1e-6);
            (a, b, true)
        }
    };

    // ===== Estado de selección de planeta/shader y toggles =====
    let mut planet_mode: u8 = 1; // 1..6
    let time_origin = Instant::now();

    let mut show_rings = false;
    let mut show_moon1 = true;
    let mut show_moon2 = false;

    let mut prev_z = false;
    let mut prev_x = false;
    let mut prev_c = false;

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }

        handle_input_camera(&window, &mut camera, &mut scale);

        // ==== Selección de planeta / estrella con teclas 1–6 ====
        if window.is_key_down(Key::Key1) { planet_mode = 1; }
        if window.is_key_down(Key::Key2) { planet_mode = 2; }
        if window.is_key_down(Key::Key3) { planet_mode = 3; }
        if window.is_key_down(Key::Key4) { planet_mode = 4; }
        if window.is_key_down(Key::Key5) { planet_mode = 5; }
        if window.is_key_down(Key::Key6) { planet_mode = 6; }

        // ==== Toggles con Z (anillos), X (luna1), C (luna2) ====
        let z_down = window.is_key_down(Key::Z);
        if z_down && !prev_z {
            show_rings = !show_rings;
        }
        prev_z = z_down;

        let x_down = window.is_key_down(Key::X);
        if x_down && !prev_x {
            show_moon1 = !show_moon1;
        }
        prev_x = x_down;

        let c_down = window.is_key_down(Key::C);
        if c_down && !prev_c {
            show_moon2 = !show_moon2;
        }
        prev_c = c_down;

        framebuffer.clear();

        let elapsed = time_origin.elapsed().as_secs_f32();
        let auto_spin_y = elapsed * 0.20;
        let rotation_auto = nalgebra_glm::Vec3::new(rotation.x, rotation.y + auto_spin_y, rotation.z);

        // ===== Planeta / Estrella =====
        let model_matrix = create_model_matrix(translation, scale, rotation_auto);
        let uniforms_base = Uniforms {
            model_matrix,
            view_matrix: camera.view_matrix(),
            time: elapsed,
            seed: 1337,
            ring_a, ring_b, ring_plane_xy,
        };

        match planet_mode {
            // 1: Tierra con nubes
            1 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &terrain_shader);

                let overlay_scale = scale * 1.02;
                let model_matrix_overlay = create_model_matrix(translation, overlay_scale, rotation_auto);
                let uniforms_overlay = Uniforms {
                    model_matrix: model_matrix_overlay,
                    view_matrix: camera.view_matrix(),
                    time: elapsed,
                    seed: 4242,
                    ring_a, ring_b, ring_plane_xy,
                };
                render_pass_alpha(&mut framebuffer, &uniforms_overlay, &obj, &clouds_shader, 0.0);
            }
            // 2: Gigante gaseoso tipo Júpiter
            2 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &gas_shader);
            }
            // 3: Marte rocoso
            3 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &mars_shader);

                let overlay_scale = scale * 1.02;
                let model_matrix_overlay = create_model_matrix(translation, overlay_scale, rotation_auto);
                let uniforms_overlay = Uniforms {
                    model_matrix: model_matrix_overlay,
                    view_matrix: camera.view_matrix(),
                    time: elapsed,
                    seed: 4242,
                    ring_a, ring_b, ring_plane_xy,
                };
                render_pass_alpha(&mut framebuffer, &uniforms_overlay, &obj, &mars_clouds_shader, 0.0);
            }
            // 4: Urano
            4 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &uranus_shader);
            }
            // 5: Saturno
            5 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &saturn_shader);
            }
            // 6: Estrella / sol
            6 => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &star_shader);
            }
            _ => {
                render_pass(&mut framebuffer, &uniforms_base, &obj, &terrain_shader);
            }
        }

        // ===== Anillos (z-bias negativo para evitar “corte”), si están activos =====
        if show_rings {
            let rings_tilt = Vec3::new(0.35, rotation_auto.y, 0.05);
            let rings_scale = scale * 1.2;
            let rings_matrix = create_model_matrix(translation, rings_scale, rings_tilt);
            let rings_uniforms = Uniforms {
                model_matrix: rings_matrix,
                view_matrix: camera.view_matrix(),
                time: elapsed,
                seed: 2025,
                ring_a, ring_b, ring_plane_xy,
            };
            render_pass(&mut framebuffer, &rings_uniforms, &rings_obj, &rings_shader);
        }

        // ===== Lunas (según toggles) =====
        for (i, m) in moons.iter().enumerate() {
            if (i == 0 && !show_moon1) || (i == 1 && !show_moon2) {
                continue;
            }
            let mm = m.model_matrix(translation, scale, elapsed);
            let mu = Uniforms {
                model_matrix: mm,
                view_matrix: camera.view_matrix(),
                time: elapsed,
                seed: m.seed,
                ring_a, ring_b, ring_plane_xy,
            };
            render_pass(&mut framebuffer, &mu, &m.obj, &*m.shader);
        }

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

// ===================== Input de cámara =====================

fn handle_input_camera(window: &Window, camera: &mut Camera, scale: &mut f32) {
    // mover cámara
    if window.is_key_down(Key::Right) { camera.position.x += 10.0; }
    if window.is_key_down(Key::Left)  { camera.position.x -= 10.0; }
    if window.is_key_down(Key::Up)    { camera.position.y -= 10.0; }
    if window.is_key_down(Key::Down)  { camera.position.y += 10.0; }

    // zoom (mantengo tu escala de objeto)
    if window.is_key_down(Key::S)     { *scale += 2.0; }
    if window.is_key_down(Key::A)     { *scale -= 2.0; }

    // rotar cámara
    if window.is_key_down(Key::Q) { camera.rotation.x -= PI / 20.0; }
    if window.is_key_down(Key::W) { camera.rotation.x += PI / 20.0; }
    if window.is_key_down(Key::E) { camera.rotation.y -= PI / 20.0; }
    if window.is_key_down(Key::R) { camera.rotation.y += PI / 20.0; }
    if window.is_key_down(Key::T) { camera.rotation.z -= PI / 20.0; }
    if window.is_key_down(Key::Y) { camera.rotation.z += PI / 20.0; }
}
