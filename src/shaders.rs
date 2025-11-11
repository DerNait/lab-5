// shaders.rs
use nalgebra_glm::{Vec2, Vec3, Mat3, Vec4, dot, normalize};
use crate::vertex::Vertex;
use crate::color::Color;
use crate::noise;

/// Uniforms compartidos
pub struct Uniforms {
    pub model_matrix: nalgebra_glm::Mat4,
    pub time: f32,   // segundos para animación
    pub seed: i32,   // semilla base
}

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
    let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
    let transformed = uniforms.model_matrix * position;

    let transformed_position = nalgebra_glm::Vec3::new(transformed.x, transformed.y, transformed.z);

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
    pub depth: f32,         // z para zbuffer
}

/// Trait de shaders de fragmento
pub trait FragmentShader {
    /// Devuelve (color, alpha)
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32);
}

#[inline]
fn rotate_y(v: Vec3, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    Vec3::new(c * v.x + s * v.z, v.y, -s * v.x + c * v.z)
}

/// ==================== Colormap ====================

#[derive(Clone)]
pub struct ColorStop {
    /// Umbral en [0,1] (ordena ascendente). Se interpola entre stops contiguos.
    pub threshold: f32,
    pub color: Color,
}

/// Interpolación entre stops con “hardness” (0 = muy suave, 1 = corte duro).
fn sample_color_stops(stops: &[ColorStop], x01: f32, hardness: f32) -> Color {
    if stops.is_empty() {
        return Color::from_hex(0xFF00FF); // magenta para detectar error
    }
    let x = x01.clamp(0.0, 1.0);

    // Si hay un único color, devuélvelo tal cual.
    if stops.len() == 1 {
        return stops[0].color;
    }

    // Buscar el segmento [i, i+1] que contiene a x
    let mut i = 0usize;
    while i + 1 < stops.len() && x > stops[i + 1].threshold {
        i += 1;
    }

    if i + 1 >= stops.len() {
        return stops[stops.len() - 1].color;
    }

    let a = &stops[i];
    let b = &stops[i + 1];

    let span = (b.threshold - a.threshold).max(1e-6);
    let mut t = (x - a.threshold) / span;

    // hardness: 0 -> suave (smoothstep), 1 -> duro (step)
    let h = hardness.clamp(0.0, 1.0);
    if h >= 0.999 {
        t = if t < 0.5 { 0.0 } else { 1.0 };
    } else {
        // mezclar lineal y smoothstep según hardness
        let ss = t * t * (3.0 - 2.0 * t);
        t = ss * (1.0 - h) + t * h;
    }

    Color::lerp(a.color, b.color, t as f32)
}

/// ==================== Noise settings ====================

#[derive(Clone, Copy)]
pub enum NoiseType {
    Value,
    Perlin,
    Voronoi,
}

#[derive(Clone, Copy)]
pub enum VoronoiDistance {
    Euclidean,
    Manhattan,
    Chebyshev,
}

/// Parámetros comunes de ruido
#[derive(Clone)]
pub struct NoiseParams {
    pub kind: NoiseType,

    /// Escala espacial principal del campo
    pub scale: f32,

    /// fBm (solo Value/Perlin)
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,

    /// Voronoi
    pub cell_size: f32,
    pub w1: f32,
    pub w2: f32,
    pub w3: f32,
    pub w4: f32,
    pub dist: VoronoiDistance,

    /// Animación en el tiempo (advección del campo)
    pub animate_time: bool,
    pub time_speed: f32,

    /// Deriva longitudinal (rotación de las coords de muestreo)
    pub animate_spin: bool,
    pub spin_speed: f32,
}

/// Evalúa ruido en [0,1] en una posición de objeto normalizada a esfera.
fn eval_noise(p_obj_unit: Vec3, uniforms: &Uniforms, params: &NoiseParams) -> f32 {
    // Spin (deriva de “longitud”)
    let spin = if params.animate_spin { uniforms.time * params.spin_speed } else { 0.0 };
    let p = rotate_y(p_obj_unit, spin);

    // Advección temporal (deformar el campo con el tiempo)
    let t = if params.animate_time { uniforms.time * params.time_speed } else { 0.0 };

    match params.kind {
        NoiseType::Value => {
            // fBm sobre value noise
            let f = noise::fbm_value_3proj(p, params.scale, t, params.octaves, params.lacunarity, params.gain, uniforms.seed);
            f.clamp(0.0, 1.0)
        }
        NoiseType::Perlin => {
            let f = noise::fbm_perlin_3proj(p, params.scale, t, params.octaves, params.lacunarity, params.gain, uniforms.seed ^ 0x9E37);
            f.clamp(0.0, 1.0)
        }
        NoiseType::Voronoi => {
            let f = noise::voronoi_3proj(
                p,
                params.scale.max(1e-6),
                params.cell_size.max(1e-4),
                params.w1, params.w2, params.w3, params.w4,
                params.dist,
                uniforms.seed ^ 0xC2B2,
                t
            );
            f.clamp(0.0, 1.0)
        }
    }
}

/// ==================== Alpha modes ====================

#[derive(Clone)]
pub enum AlphaMode {
    /// Sin alpha (1.0)
    Opaque,
    /// Alpha por umbral y “sharpness”. coverage_bias resta al umbral (más cobertura)
    Threshold {
        threshold: f32,      // 0..1
        sharpness: f32,      // >=1
        coverage_bias: f32,  // >=0 (resta al threshold efectivo)
        invert: bool,
    },
    /// Alpha constante
    Constant(f32),
}

fn alpha_from_noise(n: f32, mode: &AlphaMode) -> f32 {
    match mode {
        AlphaMode::Opaque => 1.0,
        AlphaMode::Constant(a) => a.clamp(0.0, 1.0),
        AlphaMode::Threshold { threshold, sharpness, coverage_bias, invert } => {
            let thr = (threshold - coverage_bias).clamp(0.0, 1.0);
            let k = (*sharpness).max(1.0);
            let a = ((n - thr) * k).clamp(0.0, 1.0);
            if *invert { 1.0 - a } else { a }
        }
    }
}

/// ==================== Shader Genérico ====================

pub struct ProceduralLayerShader {
    /// Parámetros de ruido
    pub noise: NoiseParams,

    /// Paleta de colores (0..1 → color). Deben venir ordenados por threshold ascendente.
    pub color_stops: Vec<ColorStop>,
    /// Dureza de la transición entre stops [0..1]
    pub color_hardness: f32,

    /// Iluminación Lambert opcional sobre el color final
    pub lighting_enabled: bool,
    pub light_dir: Vec3,    // debe venir normalizado
    pub light_min: f32,     // piso (evitar negros totales)
    pub light_max: f32,     // techo multiplicador

    /// Cómo calcular alpha (para capas tipo nubes)
    pub alpha_mode: AlphaMode,
}

impl FragmentShader for ProceduralLayerShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        // Normalizar posición de objeto a la esfera para muestrear sin costuras
        let mut p = frag.obj_pos;
        let mag = p.magnitude();
        if mag > 1e-6 { p /= mag; }

        let n = eval_noise(p, uniforms, &self.noise); // [0,1]

        let mut col = sample_color_stops(&self.color_stops, n, self.color_hardness);

        if self.lighting_enabled {
            let l = dot(&frag.normal, &self.light_dir).max(0.0);
            let mul = self.light_min + (self.light_max - self.light_min) * l;
            col = col * mul;
        }

        let a = alpha_from_noise(n, &self.alpha_mode);

        (col, a)
    }
}
