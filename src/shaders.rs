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
    pub uv: Vec2,           // UV interpolado
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
    pub threshold: f32, // [0,1] ascendente
    pub color: Color,
}

fn sample_color_stops(stops: &[ColorStop], x01: f32, hardness: f32) -> Color {
    if stops.is_empty() { return Color::from_hex(0xFF00FF); }
    let x = x01.clamp(0.0, 1.0);
    if stops.len() == 1 { return stops[0].color; }

    let mut i = 0usize;
    while i + 1 < stops.len() && x > stops[i + 1].threshold { i += 1; }
    if i + 1 >= stops.len() { return stops[stops.len() - 1].color; }

    let a = &stops[i];
    let b = &stops[i + 1];
    let span = (b.threshold - a.threshold).max(1e-6);
    let mut t = (x - a.threshold) / span;

    let h = hardness.clamp(0.0, 1.0);
    if h >= 0.999 {
        t = if t < 0.5 { 0.0 } else { 1.0 };
    } else {
        let ss = t * t * (3.0 - 2.0 * t);
        t = ss * (1.0 - h) + t * h;
    }
    Color::lerp(a.color, b.color, t)
}

/// ==================== Noise settings (genérico) ====================

#[derive(Clone, Copy)]
pub enum NoiseType { Value, Perlin, Voronoi }

#[derive(Clone, Copy)]
pub enum VoronoiDistance { Euclidean, Manhattan, Chebyshev }

#[derive(Clone)]
pub struct NoiseParams {
    pub kind: NoiseType,
    pub scale: f32,
    pub octaves: u32,
    pub lacunarity: f32,
    pub gain: f32,
    pub cell_size: f32,
    pub w1: f32, pub w2: f32, pub w3: f32, pub w4: f32,
    pub dist: VoronoiDistance,
    pub animate_time: bool,
    pub time_speed: f32,
    pub animate_spin: bool,
    pub spin_speed: f32,
}

fn eval_noise(p_obj_unit: Vec3, uniforms: &Uniforms, params: &NoiseParams) -> f32 {
    let spin = if params.animate_spin { uniforms.time * params.spin_speed } else { 0.0 };
    let p = rotate_y(p_obj_unit, spin);
    let t = if params.animate_time { uniforms.time * params.time_speed } else { 0.0 };

    match params.kind {
        NoiseType::Value => {
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
    Opaque,
    Threshold { threshold: f32, sharpness: f32, coverage_bias: f32, invert: bool },
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

/// ==================== Shader Genérico: ProceduralLayerShader ====================

pub struct ProceduralLayerShader {
    pub noise: NoiseParams,
    pub color_stops: Vec<ColorStop>,
    pub color_hardness: f32,
    pub lighting_enabled: bool,
    pub light_dir: Vec3,
    pub light_min: f32,
    pub light_max: f32,
    pub alpha_mode: AlphaMode,
}

impl FragmentShader for ProceduralLayerShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        let mut p = frag.obj_pos;
        let mag = p.magnitude();
        if mag > 1e-6 { p /= mag; }

        let n = eval_noise(p, uniforms, &self.noise);
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

/// ==================== Shader: GasGiant con flow “2 fases” ====================

#[derive(Clone)]
pub struct GasFlowParams {
    pub enabled: bool,
    pub flow_scale: f32,     // frecuencia del flowmap procedural
    pub strength: f32,       // magnitud base del desplazamiento lon/lat
    pub time_speed: f32,     // velocidad del “reloj” para las fases
    pub jets_base_speed: f32,
    pub jets_frequency: f32,
    pub shear: f32,          // reservado para extensiones
    pub phase_amp: f32,      // << NUEVO: cuánto desplazan lon/lat por ciclo (gain visual)
}

/// Esféricas
#[inline] fn to_spherical(p: Vec3) -> (f32, f32) {
    let lon = p.z.atan2(p.x);
    let lat = p.y.asin();
    (lon, lat)
}
#[inline] fn wrap_pi(a: f32) -> f32 {
    let mut x = a;
    while x <= -std::f32::consts::PI { x += 2.0 * std::f32::consts::PI; }
    while x >  std::f32::consts::PI  { x -= 2.0 * std::f32::consts::PI; }
    x
}

pub struct GasGiantShader {
    pub color_stops: Vec<ColorStop>,
    pub color_hardness: f32,
    pub band_frequency: f32,
    pub band_contrast: f32,
    pub lat_shear: f32,
    pub turb_scale: f32,
    pub turb_octaves: u32,
    pub turb_lacunarity: f32,
    pub turb_gain: f32,
    pub flow: GasFlowParams,
    pub lighting_enabled: bool,
    pub light_dir: Vec3,
    pub light_min: f32,
    pub light_max: f32,
}

impl GasGiantShader {
    fn flow_vector_static(&self, lon: f32, lat: f32, seed: i32) -> (f32, f32) {
        if !self.flow.enabled { return (0.0, 0.0); }
        let u = lon * self.flow.flow_scale;
        let v = lat * self.flow.flow_scale;
        let a = 2.0 * std::f32::consts::PI * noise::value_noise_2d(u, v, seed ^ 0xABCD);
        let (sa, ca) = a.sin_cos();
        let lat_cos = lat.cos().abs().max(0.15);
        let dlon = ca * self.flow.strength * lat_cos;
        let dlat = sa * self.flow.strength * 0.5;
        (dlon, dlat)
    }

    #[inline]
    fn band_func(&self, x: f32, contrast: f32) -> f32 {
        let s = 0.5 + 0.5 * x.sin();
        let c = contrast.clamp(0.0, 1.0);
        let sc = s * s * (3.0 - 2.0 * s);
        sc * c + s * (1.0 - c)
    }

    fn eval_bands(&self, lon: f32, lat: f32, uniforms: &Uniforms) -> f32 {
        let turb = {
            let u = lon * self.turb_scale;
            let v = lat * self.turb_scale;
            let n = noise::perlin_2d(u, v, uniforms.seed ^ 0x2222);
            let f = noise::fbm_perlin_3proj(
                Vec3::new(lon.cos(), lat, lon.sin()),
                1.0, 0.0,
                self.turb_octaves, self.turb_lacunarity, self.turb_gain,
                uniforms.seed ^ 0x4444
            );
            (n * 2.0 - 1.0) * 0.5 + (f * 2.0 - 1.0) * 0.5
        };
        let shear_term = self.lat_shear * lon;
        let phase = self.band_frequency * (lat + shear_term + turb);
        self.band_func(phase, self.band_contrast)
    }
}

impl FragmentShader for GasGiantShader {
    fn shade(&self, frag: &FragAttrs, uniforms: &Uniforms) -> (Color, f32) {
        // Normalizar a esfera
        let mut p = frag.obj_pos;
        let mag = p.magnitude();
        if mag > 1e-6 { p /= mag; }
        let (mut lon, mut lat) = to_spherical(p);

        // Deriva zonal: mantenla acotada (sin acumulación infinita)
        // (Si quisieras reactivar el drift continuo, aquí sumarías + uniforms.time * jets)
        let jets = self.flow.jets_base_speed * (self.flow.jets_frequency * lat).sin();

        // Fases 0..1 y mezcla triangular
        let t = uniforms.time * self.flow.time_speed;
        let phase1 = t.fract();
        let phase2 = (phase1 + 0.5).fract();
        let flow_mix = ((phase1 - 0.5).abs() * 2.0).clamp(0.0, 1.0);

        // Vector de flujo estático + componente zonal
        let (vx, vy) = self.flow_vector_static(lon, lat, uniforms.seed ^ 0x1357);
        let vlon = vx + jets;

        // AMPLIFICADOR VISUAL de la fase (para que se note)
        let amp = self.flow.phase_amp.max(0.0);

        // Dos muestreos faseados y mezcla sin “pop”
        let lon_a = wrap_pi(lon + vlon * phase1 * amp);
        let lat_a = (lat +  vy   * phase1 * amp)
            .clamp(-std::f32::consts::FRAC_PI_2 + 1e-3, std::f32::consts::FRAC_PI_2 - 1e-3);

        let lon_b = wrap_pi(lon + vlon * phase2 * amp);
        let lat_b = (lat +  vy   * phase2 * amp)
            .clamp(-std::f32::consts::FRAC_PI_2 + 1e-3, std::f32::consts::FRAC_PI_2 - 1e-3);

        let bands_a = self.eval_bands(lon_a, lat_a, uniforms);
        let bands_b = self.eval_bands(lon_b, lat_b, uniforms);
        let bands   = bands_a * (1.0 - flow_mix) + bands_b * flow_mix;

        let mut col = sample_color_stops(&self.color_stops, bands, self.color_hardness);

        if self.lighting_enabled {
            let l = dot(&frag.normal, &self.light_dir).max(0.0);
            let mul = self.light_min + (self.light_max - self.light_min) * l;
            col = col * mul;
        }
        (col, 1.0)
    }
}
