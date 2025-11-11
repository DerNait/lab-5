// shaders.rs
use nalgebra_glm::{Vec2, Vec3, Mat3, Vec4, dot, normalize};
use crate::vertex::Vertex;
use crate::color::Color;
use crate::noise;

/// Uniforms compartidos
pub struct Uniforms {
    pub model_matrix: nalgebra_glm::Mat4,
    pub time: f32,          // segundos para animación
    pub seed: i32,          // semilla de ruido
}

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
    // Transform position
    let position = Vec4::new(
        vertex.position.x,
        vertex.position.y,
        vertex.position.z,
        1.0
    );
    let transformed = uniforms.model_matrix * position;

    // No hay proyección real (modo pantalla directa). Mantener w=1.
    let transformed_position = nalgebra_glm::Vec3::new(
        transformed.x,
        transformed.y,
        transformed.z
    );

    // Normal transform
    let model_mat3 = Mat3::new(
        uniforms.model_matrix[0], uniforms.model_matrix[1], uniforms.model_matrix[2],
        uniforms.model_matrix[4], uniforms.model_matrix[5], uniforms.model_matrix[6],
        uniforms.model_matrix[8], uniforms.model_matrix[9], uniforms.model_matrix[10]
    );
    let normal_matrix = model_mat3.transpose().try_inverse().unwrap_or(Mat3::identity());
    let transformed_normal = (normal_matrix * vertex.normal).normalize();

    Vertex {
        position: vertex.position,
        normal: vertex.normal,
        tex_coords: vertex.tex_coords,
        color: vertex.color,
        transformed_position,
        transformed_normal,
    }
}

/// Atributos interpolados que llegan al fragment shader
pub struct FragAttrs {
    pub obj_pos: Vec3,      // posición en espacio de OBJETO
    pub normal: Vec3,       // normal interpolada y normalizada
    pub uv: Vec2,           // UV interpolado (por si lo quieres)
    pub depth: f32,         // z en espacio pantalla actual (para zbuffer)
}

/// Trait de shaders de fragmento
pub trait FragmentShader {
    /// Devuelve (color, alpha)
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32);
}

/// ====== Helpers internos =======

#[inline]
fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
}

#[inline]
fn tri_fbm(p: Vec3, scale: f32, time: f32, seed: i32) -> f32 {
    // Desplaza el campo de ruido usando "time" con fuerza 1.0 (ya lo escalas afuera con time_speed/speed)
    let s = scale;
    let t = time;

    // Desplazamos el primer eje de cada par para que el patrón “viaje”
    let nxy = noise::fbm_2d((p.x + t) * s, p.y * s,        5, 2.0, 0.5, seed);
    let nyz = noise::fbm_2d((p.y + t) * s, p.z * s,        5, 2.0, 0.5, seed ^ 0x9E37);
    let nzx = noise::fbm_2d((p.z + t) * s, p.x * s,        5, 2.0, 0.5, seed ^ 0xC2B2);

    (nxy + nyz + nzx) / 3.0
}

#[inline]
fn lambert(normal: Vec3, light_dir: Vec3) -> f32 {
    dot(&normal, &light_dir).max(0.0)
}

/// ====== Shaders de ejemplo =======

/// Bands/elevación con ruido espacial sin costuras
pub struct ElevationBandsShader {
    pub low_color: Color,
    pub mid_color: Color,
    pub high_color: Color,
    pub thresholds: (f32, f32),  // v < t0 -> low, < t1 -> mid, else high (sobre “relieve”)
    pub noise_scale: f32,        // escala del ruido espacial
    pub noise_strength: f32,     // 0..1 (cuánto pesa el ruido vs latitud)
    pub animate_time: bool,    // ruido cambia con el tiempo
    pub time_speed: f32,       // factor del tiempo
    pub animate_spin: bool,    // gira el muestreo alrededor de Y
    pub spin_speed: f32, 
}

impl FragmentShader for ElevationBandsShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        // Posición normalizada a la esfera para evitar dependencias de escala
        let mut p = frag.obj_pos;
        let mag = p.magnitude();
        if mag > 1e-6 { p /= mag; }

        // Opcional: rotar coordenadas de muestreo (no la malla)
        let spin = if self.animate_spin { uniforms.time * self.spin_speed } else { 0.0 };
        let p = rotate_y(p, spin);

        // Opcional: tiempo en el ruido
        let t = if self.animate_time { uniforms.time * self.time_speed } else { 0.0 };

        let n  = tri_fbm(p, self.noise_scale, t, uniforms.seed); // [0,1]
        let n2 = n * 2.0 - 1.0;                                   // [-1,1]
        let lat = p.y;
        let rel = (1.0 - self.noise_strength) * lat + self.noise_strength * n2;

        // Banding por umbrales sobre el "relieve"
        let mut base = if rel < self.thresholds.0 {
            self.low_color
        } else if rel < self.thresholds.1 {
            self.mid_color
        } else {
            self.high_color
        };

        // Iluminación simple (Lambert) para dar volumen
        let light_dir = normalize(&Vec3::new(0.25, 0.6, -1.0));
        let l = lambert(frag.normal, light_dir);
        base = base * (0.35 + 0.65 * l); // evita negros totales

        (base, 1.0)
    }
}

/// Nubes sin costura (espacio de objeto), borde suave y alpha
pub struct CloudsShader {
    pub cloud_color: Color,
    pub scale: f32,
    pub threshold: f32,   // 0..1
    pub sharpness: f32,   // mayor => borde más duro
    pub speed: f32,       // animación
    pub coverage_bias: f32, // cuánta cobertura extra (resta al threshold)
    pub animate_time: bool,
    pub time_speed: f32,
    pub animate_spin: bool, // gira el muestreo -> deriva en longitud
    pub spin_speed: f32,
}

impl FragmentShader for CloudsShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        let mut p = frag.obj_pos;
        let mag = p.magnitude();
        if mag > 1e-6 { p /= mag; }

        // Spin del muestreo para “derrapar” las nubes
        let spin = if self.animate_spin { uniforms.time * self.spin_speed } else { 0.0 };
        let p = rotate_y(p, spin);

        // Tiempo del ruido (para que cambie la forma)
        let t = if self.animate_time { uniforms.time * self.time_speed } else { 0.0 };

        let n = tri_fbm(p, self.scale, t * self.speed, uniforms.seed); // [0,1]

        // Más cobertura = threshold efectivo más bajo
        let thr = (self.threshold - self.coverage_bias).clamp(0.0, 1.0);
        let k = self.sharpness.max(1.0);
        let a = ((n - thr) * k).clamp(0.0, 1.0);

        // Iluminación sutil
        let light_dir = normalize(&Vec3::new(0.25, 0.6, -1.0));
        let l = 0.5 + 0.5 * dot(&frag.normal, &light_dir).max(0.0);
        let color = self.cloud_color * (0.8 + 0.2 * l);

        (color, a)
    }
}
