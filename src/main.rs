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
    ColorStop, AlphaMode,
    GasGiantShader, GasFlowParams,
};
use color::Color;

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

fn render_pass(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &impl FragmentShader
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
    shader: &impl FragmentShader
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
            framebuffer.draw_rgba(x, y, frag.depth, frag.color.to_hex(), frag.alpha);
        }
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

    let obj = Obj::load("assets/models/Planet.obj").expect("Failed to load obj");

    let (min_v, max_v) = obj.bounds();
    let size = max_v - min_v;
    let center = (min_v + max_v) * 0.5;

    let target_w = framebuffer_width as f32 * 0.8;
    let target_h = framebuffer_height as f32 * 0.8;
    let sx = if size.x.abs() < 1e-6 { 1.0 } else { target_w / size.x.abs() };
    let sy = if size.y.abs() < 1e-6 { 1.0 } else { target_h / size.y.abs() };
    let mut scale = sx.min(sy);

    let mut translation = Vec3::new(
        (framebuffer_width as f32) * 0.5 - center.x * scale,
        (framebuffer_height as f32) * 0.5 - center.y * scale,
        -center.z * scale
    );

    let mut rotation = Vec3::new(0.0, 0.0, 0.0);

    // ===== Tierra: terreno
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

    // ===== Tierra: nubes
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

    // ===== Gas giant con flow “2 fases 0xFFFFFF”
    let gas_palette = vec![
        ColorStop { threshold: 0.00, color: Color::from_hex(0x734D1E) },
        ColorStop { threshold: 0.20, color: Color::from_hex(0x6E5034) },
        ColorStop { threshold: 0.40, color: Color::from_hex(0xB58C5A) },
        ColorStop { threshold: 0.60, color: Color::from_hex(0xD9B98B) },
        ColorStop { threshold: 0.80, color: Color::from_hex(0xA57442) },
        ColorStop { threshold: 1.00, color: Color::from_hex(0xE4D1B5) },
    ];

    let gas_shader = GasGiantShader {
        color_stops: gas_palette,
        color_hardness: 0.35,
        band_frequency: 4.0,
        band_contrast: 1.0,
        lat_shear: 0.2,
        turb_scale: 10.0,
        turb_octaves: 4,
        turb_lacunarity: 2.0,
        turb_gain: 0.55,
        flow: GasFlowParams {
            enabled: true,
            flow_scale: 3.0,      // ↑ un poco la frecuencia espacial
            strength: 0.04,       // ↑ más desplazamiento base
            time_speed: 0.6,      // ↑ ciclos más rápidos
            jets_base_speed: 0.12,// ↑ componente zonal (sigue acotado por la fase)
            jets_frequency: 6.0,
            shear: 0.25,
            phase_amp: 3.0,       // << NUEVO: amplificador visual (se nota ya)
        },
        lighting_enabled: true,
        light_dir: normalize(&Vec3::new(0.25, 0.6, -1.0)),
        light_min: 0.45,
        light_max: 1.05,
    };

    let mut use_gas_giant = true; // ← Toggle runtime: G/H
    let mut time_origin = Instant::now();

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }

        handle_input(&window, &mut translation, &mut rotation, &mut scale);

        // Toggle shaders (evita rebotes: tecla “modo set”)
        if window.is_key_down(Key::G) { use_gas_giant = true; }
        if window.is_key_down(Key::H) { use_gas_giant = false; }

        framebuffer.clear();

        let elapsed = time_origin.elapsed().as_secs_f32();
        let auto_spin_y = elapsed * 0.20;

        let rotation_auto = nalgebra_glm::Vec3::new(rotation.x, rotation.y + auto_spin_y, rotation.z);
        let model_matrix = create_model_matrix(translation, scale, rotation_auto);
        let uniforms_base = Uniforms { model_matrix, time: elapsed, seed: 1337 };

        if use_gas_giant {
            // Gas giant (opaco)
            render_pass(&mut framebuffer, &uniforms_base, &obj, &gas_shader);
        } else {
            // Tierra (base + nubes alpha)
            render_pass(&mut framebuffer, &uniforms_base, &obj, &terrain_shader);

            let overlay_scale = scale * 1.02;
            let model_matrix_overlay = create_model_matrix(translation, overlay_scale, rotation_auto);
            let uniforms_overlay = Uniforms { model_matrix: model_matrix_overlay, time: elapsed, seed: 4242 };
            render_pass_alpha(&mut framebuffer, &uniforms_overlay, &obj, &clouds_shader);
        }

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

fn handle_input(window: &Window, translation: &mut Vec3, rotation: &mut Vec3, scale: &mut f32) {
    if window.is_key_down(Key::Right) { translation.x += 10.0; }
    if window.is_key_down(Key::Left)  { translation.x -= 10.0; }
    if window.is_key_down(Key::Up)    { translation.y -= 10.0; }
    if window.is_key_down(Key::Down)  { translation.y += 10.0; }
    if window.is_key_down(Key::S)     { *scale += 2.0; }
    if window.is_key_down(Key::A)     { *scale -= 2.0; }
    if window.is_key_down(Key::Q)     { rotation.x -= PI / 10.0; }
    if window.is_key_down(Key::W)     { rotation.x += PI / 10.0; }
    if window.is_key_down(Key::E)     { rotation.y -= PI / 10.0; }
    if window.is_key_down(Key::R)     { rotation.y += PI / 10.0; }
    if window.is_key_down(Key::T)     { rotation.z -= PI / 10.0; }
    if window.is_key_down(Key::Y)     { rotation.z += PI / 10.0; }
}
