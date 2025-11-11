use nalgebra_glm::{Vec3, Mat4};
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
use shaders::{vertex_shader, Uniforms, ElevationBandsShader, CloudsShader};
use color::Color;

pub struct ModelMatrices {
    pub base: Mat4,
    pub overlay: Mat4,
}

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

fn render_with_shader(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &dyn shaders::FragmentShader,
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

    // Escribimos fragments con alpha blending (si el shader lo desea)
    // NOTA: aquí asumimos alpha=1.0 para base. Para overlay (nubes)
    // re-renderizamos con otro shader y otro uniforms/model_matrix.
    for fragment in all_fragments {
        let x = fragment.position.x as usize;
        let y = fragment.position.y as usize;
        if x < framebuffer.width && y < framebuffer.height {
            // En este pass base, alpha=1.0 (completo). El color ya viene del shader (en triangle_with_shader se calculó).
            let color = fragment.color.to_hex();
            framebuffer.draw_rgba(x, y, fragment.depth, color, 1.0);
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
    )
    .unwrap();

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

    // ===== Shaders iniciales =====
    let terrain_shader = ElevationBandsShader {
        low_color:  Color::from_hex(0x2C7A7B),
        mid_color:  Color::from_hex(0x88B04B),
        high_color: Color::from_hex(0xD9CAB3),
        thresholds: (-0.15, 0.22),
        noise_scale: 3.0,
        noise_strength: 0.75,

        // Efectos opcionales (por defecto, terreno estático)
        animate_time: false,
        time_speed: 1.0,
        animate_spin: false,
        spin_speed: 0.0,
    };

    let clouds_shader = CloudsShader {
        cloud_color: Color::from_hex(0xFFFFFF),
        scale: 5.0,

        // Más cobertura: baja el threshold base y agrega bias
        threshold: 0.50,      // base
        coverage_bias: 0.05,  // resta al threshold efectivo → ~62% de cobertura

        sharpness: 6.0,

        // Ruido que cambia con el tiempo (true)
        animate_time: true,
        time_speed: 1.0,
        speed: 0.05,

        // Derrape longitudinal de nubes (true)
        animate_spin: true,
        spin_speed: 0.05,     // rad/s
    };

    let mut time_origin = Instant::now();

    while window.is_open() {
        if window.is_key_down(Key::Escape) { break; }

        handle_input(&window, &mut translation, &mut rotation, &mut scale);

        framebuffer.clear();

        let elapsed = time_origin.elapsed().as_secs_f32();
        let auto_spin_y = elapsed * 0.20; // rad/s (ajústalo al gusto)

        // ===== Pass base (terreno) =====
        let rotation_auto = nalgebra_glm::Vec3::new(rotation.x, rotation.y + auto_spin_y, rotation.z);
        let model_matrix = create_model_matrix(translation, scale, rotation_auto);

        let uniforms_base = Uniforms { model_matrix, time: elapsed, seed: 1337 };

        render_pass(&mut framebuffer, &uniforms_base, &obj, &terrain_shader);

        // ===== Pass overlay (nubes) =====
        let overlay_scale = scale * 1.02;
        let model_matrix_overlay = create_model_matrix(translation, overlay_scale, rotation_auto);
        let uniforms_overlay = Uniforms { model_matrix: model_matrix_overlay, time: elapsed, seed: 4242 };

        render_pass_alpha(&mut framebuffer, &uniforms_overlay, &obj, &clouds_shader);

        window
            .update_with_buffer(&framebuffer.buffer, framebuffer_width, framebuffer_height)
            .unwrap();

        std::thread::sleep(frame_delay);
    }
}

fn render_pass(
    framebuffer: &mut Framebuffer,
    uniforms: &Uniforms,
    obj: &Obj,
    shader: &impl shaders::FragmentShader
) {
    // Igual que render_with_shader pero dibujando alpha=1.0 directamente
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
    shader: &impl shaders::FragmentShader
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


fn handle_input(window: &Window, translation: &mut Vec3, rotation: &mut Vec3, scale: &mut f32) {
    if window.is_key_down(Key::Right) {
        translation.x += 10.0;
    }
    if window.is_key_down(Key::Left) {
        translation.x -= 10.0;
    }
    if window.is_key_down(Key::Up) {
        translation.y -= 10.0;
    }
    if window.is_key_down(Key::Down) {
        translation.y += 10.0;
    }
    if window.is_key_down(Key::S) {
        *scale += 2.0;
    }
    if window.is_key_down(Key::A) {
        *scale -= 2.0;
    }
    if window.is_key_down(Key::Q) {
        rotation.x -= PI / 10.0;
    }
    if window.is_key_down(Key::W) {
        rotation.x += PI / 10.0;
    }
    if window.is_key_down(Key::E) {
        rotation.y -= PI / 10.0;
    }
    if window.is_key_down(Key::R) {
        rotation.y += PI / 10.0;
    }
    if window.is_key_down(Key::T) {
        rotation.z -= PI / 10.0;
    }
    if window.is_key_down(Key::Y) {
        rotation.z += PI / 10.0;
    }
}
