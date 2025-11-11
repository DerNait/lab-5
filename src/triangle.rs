use nalgebra_glm::{Vec3, Vec2};
use crate::fragment::Fragment;
use crate::vertex::Vertex;
use crate::shaders::{FragmentShader, FragAttrs, Uniforms};

pub fn _triangle(v1: &Vertex, v2: &Vertex, v3: &Vertex) -> Vec<Fragment> {
    let mut fragments = Vec::new();
    // Modo wireframe (sigue presente si lo quieres)
    use crate::line::line;
    fragments.extend(line(v1, v2));
    fragments.extend(line(v2, v3));
    fragments.extend(line(v3, v1));
    fragments
}

pub fn triangle_with_shader(
    v1: &Vertex,
    v2: &Vertex,
    v3: &Vertex,
    shader: &dyn FragmentShader,
    uniforms: &Uniforms
) -> Vec<Fragment> {
    let mut fragments = Vec::new();
    let (a, b, c) = (v1.transformed_position, v2.transformed_position, v3.transformed_position);

    let (min_x, min_y, max_x, max_y) = calculate_bounding_box(&a, &b, &c);

    let triangle_area = edge_function(&a, &b, &c);
    if triangle_area.abs() < 1e-6 {
        return fragments; // evita degenerate
    }

    // Prep datos base para interpolación: obj pos, normal, uv
    let obj_a = v1.position;  let obj_b = v2.position;  let obj_c = v3.position;
    let n_a = v1.transformed_normal; let n_b = v2.transformed_normal; let n_c = v3.transformed_normal;
    let uv_a = v1.tex_coords; let uv_b = v2.tex_coords; let uv_c = v3.tex_coords;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);

            let (w1, w2, w3) = barycentric_coordinates(&p, &a, &b, &c, triangle_area);

            if w1 >= 0.0 && w2 >= 0.0 && w3 >= 0.0 {
                let mut normal = n_a * w1 + n_b * w2 + n_c * w3;
                if normal.magnitude() > 1e-6 { normal = normal.normalize(); }
                else { normal = Vec3::new(0.0, 0.0, 1.0); }

                let obj_pos = obj_a * w1 + obj_b * w2 + obj_c * w3;
                let uv = Vec2::new(
                    uv_a.x * w1 + uv_b.x * w2 + uv_c.x * w3,
                    uv_a.y * w1 + uv_b.y * w2 + uv_c.y * w3
                );

                let depth = a.z * w1 + b.z * w2 + c.z * w3;

                let attrs = FragAttrs { obj_pos, normal, uv, depth };
                let (color, alpha) = shader.shade(&attrs, uniforms);

                fragments.push(Fragment::with_alpha(x as f32, y as f32, color, depth, alpha));
            }
        }
    }
    fragments
}

fn calculate_bounding_box(v1: &Vec3, v2: &Vec3, v3: &Vec3) -> (i32, i32, i32, i32) {
    let min_x = v1.x.min(v2.x).min(v3.x).floor() as i32;
    let min_y = v1.y.min(v2.y).min(v3.y).floor() as i32;
    let max_x = v1.x.max(v2.x).max(v3.x).ceil() as i32;
    let max_y = v1.y.max(v2.y).max(v3.y).ceil() as i32;
    (min_x, min_y, max_x, max_y)
}

fn barycentric_coordinates(p: &Vec3, a: &Vec3, b: &Vec3, c: &Vec3, area: f32) -> (f32, f32, f32) {
    let w1 = edge_function(b, c, p) / area;
    let w2 = edge_function(c, a, p) / area;
    let w3 = edge_function(a, b, p) / area;
    (w1, w2, w3)
}

fn edge_function(a: &Vec3, b: &Vec3, c: &Vec3) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}
