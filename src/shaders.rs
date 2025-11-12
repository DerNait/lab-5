// shaders.rs
use nalgebra_glm::{Vec2, Vec3, Mat3, Vec4, dot};
use crate::vertex::Vertex;
use crate::color::Color;
use crate::noise;

/// Uniforms compartidos
pub struct Uniforms {
    pub model_matrix: nalgebra_glm::Mat4,
    pub view_matrix:  nalgebra_glm::Mat4,
    pub time: f32,   // segundos para animación
    pub seed: i32,   // semilla base
    pub ring_a: f32,      // semieje X (o el del primer eje del plano elegido)
    pub ring_b: f32,      // semieje Y (o segundo eje)
    pub ring_plane_xy: bool,
}

pub fn vertex_shader(vertex: &Vertex, uniforms: &Uniforms) -> Vertex {
    let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);

    // Aplicamos View * Model
    let vm = uniforms.view_matrix * uniforms.model_matrix;
    let transformed = vm * position;

    let transformed_position = nalgebra_glm::Vec3::new(transformed.x, transformed.y, transformed.z);

    // normales con (View*Model) sin traslación
    let m = vm;
    let model_mat3 = Mat3::new(
        m[0],  m[1],  m[2],
        m[4],  m[5],  m[6],
        m[8],  m[9],  m[10]
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

/// Plano del anillo para proyectar posición OBJ → UV sintéticos
#[derive(Clone, Copy)]
pub enum RingPlane { XY, XZ, YZ }

#[derive(Clone, Copy)]
pub enum NoiseType {
    Value,
    Perlin,
    Voronoi,
    BandedGas,
    RadialGradient {
        inner: f32,
        outer: f32,
        invert: bool,
        bias: f32,
        gamma: f32,
    },
    /// Degradé radial en espacio UV (0..1), centro por defecto (0.5,0.5)
    UVRadialGradient {
        center: nalgebra_glm::Vec2, // típicamente Vec2::new(0.5, 0.5)
        invert: bool,
        bias: f32,
        gamma: f32,
    },
    /// NUEVO: radial en UV generado desde la POSICIÓN en OBJ (centrado en el origen del modelo)
    UVRadialGradientObj {
        plane: RingPlane,   // XY, XZ o YZ
        radius_max: f32,    // radio externo del anillo en unidades del OBJ
        invert: bool,
        bias: f32,
        gamma: f32,
    },
}

#[derive(Clone, Copy)]
pub enum VoronoiDistance { Euclidean, Manhattan, Chebyshev }

/// Flowmap genérico: lo puede usar cualquier NoiseType (lo activas con `flow.enabled`)
#[derive(Clone)]
pub struct FlowParams {
    pub enabled: bool,
    pub flow_scale: f32,   // frecuencia del flowmap procedural
    pub strength: f32,     // magnitud base del desplazamiento lon/lat
    pub time_speed: f32,   // velocidad del “reloj” para las fases
    pub jets_base_speed: f32,
    pub jets_frequency: f32,
    pub phase_amp: f32,    // ganancia visual por ciclo (0..+)
}

impl Default for FlowParams {
    fn default() -> Self {
        Self {
            enabled: false,
            flow_scale: 2.0,
            strength: 0.0,
            time_speed: 0.0,
            jets_base_speed: 0.0,
            jets_frequency: 1.0,
            phase_amp: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct NoiseParams {
    pub kind: NoiseType,

    // Parámetros comunes (Value/Perlin/Voronoi)
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

    // --------- NUEVOS: usados por BandedGas (ignorados por otros tipos) ----------
    pub band_frequency: f32,   // densidad de bandas por radian
    pub band_contrast: f32,    // 0..1, forma de onda más dura
    pub lat_shear: f32,        // inclinación longitudinal dependiente de lon
    pub turb_scale: f32,       // turbulencia (Perlin/FBM) sobre lon/lat
    pub turb_octaves: u32,
    pub turb_lacunarity: f32,
    pub turb_gain: f32,
    pub turb_amp: f32,

    /// Flowmap opcional (para BandedGas o incluso para distorsionar Value/Perlin/Voronoi al mapear en esfera)
    pub flow: FlowParams,
}

impl NoiseParams {
    /// Ayuda para “defaults” de BandedGas (si no quieres rellenar todo a mano)
    pub fn banded_defaults(kind: NoiseType) -> Self {
        Self {
            kind,
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

            band_frequency: 4.0,
            band_contrast: 0.85,
            lat_shear: 0.2,
            turb_scale: 10.0,
            turb_octaves: 4,
            turb_lacunarity: 2.0,
            turb_gain: 0.55,
            turb_amp: 0.5,

            flow: FlowParams::default(),
        }
    }
}

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

/// Curvatura controlada del gradiente (mezcla bias y potencia)
#[inline]
fn shape_bias_gamma(mut x: f32, bias: f32, gamma: f32) -> f32 {
    let b = bias.clamp(0.0, 1.0);
    // mezcla entre lineal y cuadrática hacia el borde
    x = x * (1.0 - b) + x * x * b;
    let g = gamma.max(1e-6);
    x.powf(g)
}

fn eval_noise(p_obj_unit_in: Vec3, uniforms: &Uniforms, params: &NoiseParams) -> f32 {
    // Spin (rotación del patrón en espacio objeto)
    let spin = if params.animate_spin { uniforms.time * params.spin_speed } else { 0.0 };
    let mut p = rotate_y(p_obj_unit_in, spin);

    // === Desplazamiento por FLOWMAP (lon/lat), si corresponde ===
    let (mut lon, mut lat) = to_spherical(p);
    if params.flow.enabled {
        let u = lon * params.flow.flow_scale;
        let v = lat * params.flow.flow_scale;
        let a = 2.0 * std::f32::consts::PI * noise::value_noise_2d(u, v, uniforms.seed ^ 0xABCD);
        let (sa, ca) = a.sin_cos();

        // vector base a partir del valor de ruido
        let lat_cos = lat.cos().abs().max(0.15);
        let dlon = ca * params.flow.strength * lat_cos;
        let dlat = sa * params.flow.strength * 0.5;

        // deriva zonal acotada (no acumula infinito)
        let jets = params.flow.jets_base_speed * (params.flow.jets_frequency * lat).sin();

        // fases acotadas 0..1 + mezcla triangular para evitar “pops”
        let t = uniforms.time * params.flow.time_speed;
        let phase1 = t.fract();
        let phase2 = (phase1 + 0.5).fract();
        let flow_mix = ((phase1 - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let amp = params.flow.phase_amp.max(0.0);

        let lon_a = wrap_pi(lon + (dlon + jets) * phase1 * amp);
        let lat_a = (lat + dlat * phase1 * amp)
            .clamp(-std::f32::consts::FRAC_PI_2 + 1e-3, std::f32::consts::FRAC_PI_2 - 1e-3);

        let lon_b = wrap_pi(lon + (dlon + jets) * phase2 * amp);
        let lat_b = (lat + dlat * phase2 * amp)
            .clamp(-std::f32::consts::FRAC_PI_2 + 1e-3, std::f32::consts::FRAC_PI_2 - 1e-3);

        // remuestrea p sobre la esfera desplazada
        let bands_a = eval_bands_core(lon_a, lat_a, uniforms, params);
        let bands_b = eval_bands_core(lon_b, lat_b, uniforms, params);
        // si estamos en BandedGas y flow activo, devolvemos directamente las bandas mezcladas
        if let NoiseType::BandedGas = params.kind {
            return bands_a * (1.0 - flow_mix) + bands_b * flow_mix;
        } else {
            // para otros tipos, actualizamos (lon,lat) mezclado y reconstruimos p
            let lon_m = wrap_pi(lon_a * (1.0 - flow_mix) + lon_b * flow_mix);
            let lat_m =      lat_a * (1.0 - flow_mix) + lat_b * flow_mix;
            let cl = lat_m.cos();
            p = Vec3::new(cl * lon_m.cos(), lat_m.sin(), cl * lon_m.sin());
            lon = lon_m; lat = lat_m;
        }
    }

    // === Evaluación por tipo de ruido ===
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
        NoiseType::BandedGas => {
            // sin flow (o flow ya manejado arriba): bandas “tipo Júpiter”
            eval_bands_core(lon, lat, uniforms, params)
        }
        NoiseType::RadialGradient { inner, outer, invert, bias, gamma } => {
            // p llega normalizado a la esfera del objeto. Para anillo, usa radio en XZ.
            let r = (p.x * p.x + p.z * p.z).sqrt();
            let i = (inner).max(1e-6);
            let o = (outer).max(i + 1e-6);
            let mut rn = ((r - i) / (o - i)).clamp(0.0, 1.0); // 0 en inner, 1 en outer
            rn = shape_bias_gamma(rn, bias, gamma);
            let v = if invert { rn } else { 1.0 - rn }; // por defecto: centro claro → borde oscuro
            v.clamp(0.0, 1.0)
        }
        NoiseType::UVRadialGradient { center, invert, bias, gamma } => {
            // Radial en UV nativos (mapeo lon/lat → uv)
            let u_coord = 0.5 + lon / (2.0 * std::f32::consts::PI);
            let v_coord = 0.5 + lat / std::f32::consts::PI;
            let du = u_coord - center.x;
            let dv = v_coord - center.y;
            let max_r = 0.5_f32; // distancia desde centro al borde en UV
            let mut rn = ((du * du + dv * dv).sqrt() / max_r).clamp(0.0, 1.0);
            rn = shape_bias_gamma(rn, bias, gamma);
            let val = if invert { rn } else { 1.0 - rn };
            val.clamp(0.0, 1.0)
        }
        // La variante UVRadialGradientObj se implementa en shade() porque requiere p_obj crudo (no unit).
        _ => 0.0, // aquí no se usa; se maneja en shade()
    }
}

// Núcleo de bandas (reutilizable para flow o no-flow)
fn eval_bands_core(lon: f32, lat: f32, uniforms: &Uniforms, p: &NoiseParams) -> f32 {
    // Representación periódica de la longitud
    let (sln, cln) = lon.sin_cos();

    // Turbulencia SEAMLESS: todo en espacio periódico (cos(lon), sin(lon), lat)
    let turb = {
        // Solo fBm 3-proyecciones (ya libre de seams); quitamos el perlin_2d(u=lon, v=lat)
        let f = noise::fbm_perlin_3proj(
            Vec3::new(cln, lat, sln), // coords periódicas
            p.turb_scale.max(1e-6),   // usamos turb_scale como frecuencia
            0.0,
            p.turb_octaves, p.turb_lacunarity, p.turb_gain,
            uniforms.seed ^ 0x4444
        );
        (f * 2.0 - 1.0) // [-1,1]
    };

    // Cizalla latitudinal periódica (en vez de ∝ lon)
    let shear_term = p.lat_shear * sln;

    // Bandas
    let phase = p.band_frequency * (lat + shear_term + p.turb_amp * turb);
    band_func(phase, p.band_contrast)
}

#[inline]
fn band_func(x: f32, contrast: f32) -> f32 {
    let s = 0.5 + 0.5 * x.sin();      // [0,1]
    let c = contrast.clamp(0.0, 1.0); // mezcla soft/hard
    let sc = s * s * (3.0 - 2.0 * s); // smoothstep(s)
    sc * c + s * (1.0 - c)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
        // Mantén ambas: posición de objeto cruda (para anillos/planos)
        // y posición unitaria (para ruidos esféricos en planetas).
        let p_obj = frag.obj_pos;
        let p_unit = {
            let mut q = p_obj;
            let m = q.magnitude();
            if m > 1e-6 { q /= m; }
            q
        };

        // Elegimos cómo evaluar 'n' según el tipo de ruido
        let n = match self.noise.kind {
            // Radial en OBJ: usa radio en el plano XZ *sin normalizar*
            NoiseType::RadialGradient { inner, outer, invert, bias, gamma } => {
                // Elige el plano correcto para el anillo
                let (px, py, a, b) = if uniforms.ring_plane_xy {
                    (p_obj.x, p_obj.y, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
                } else {
                    // si tu malla estuviera en XZ, pon ring_plane_xy = false
                    (p_obj.x, p_obj.z, uniforms.ring_a.max(1e-6), uniforms.ring_b.max(1e-6))
                };

                // radio elíptico normalizado: (x/a)^2 + (y/b)^2
                let r_ell = ((px / a).powi(2) + (py / b).powi(2)).sqrt();

                // inner/outer son fracciones del borde exterior (0..1)
                let i = inner.clamp(0.0, 0.999);
                let o = outer.clamp(i + 1e-6, 1.0);
                let mut rn = ((r_ell - i) / (o - i)).clamp(0.0, 1.0);

                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }

            // Radial en UV nativos del mesh
            NoiseType::UVRadialGradient { center, invert, bias, gamma } => {
                let du = frag.uv.x - center.x;
                let dv = frag.uv.y - center.y;
                let max_r = 0.5_f32; // desde (0.5,0.5) al borde de la textura
                let mut rn = ((du*du + dv*dv).sqrt() / max_r).clamp(0.0, 1.0);
                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }
            // NUEVO: Radial en UV "sintéticos" generados desde OBJ (centrado en el planeta)
            NoiseType::UVRadialGradientObj { plane, radius_max, invert, bias, gamma } => {
                let (px, py) = match plane {
                    RingPlane::XY => (p_obj.x, p_obj.y),
                    RingPlane::XZ => (p_obj.x, p_obj.z),
                    RingPlane::YZ => (p_obj.y, p_obj.z),
                };
                let r = (px*px + py*py).sqrt();
                let mut rn = (r / radius_max.max(1e-6)).clamp(0.0, 1.0); // 0 centro, 1 borde
                rn = shape_bias_gamma(rn, bias, gamma);
                let v = if invert { rn } else { 1.0 - rn };
                v.clamp(0.0, 1.0)
            }
            // El resto de tipos se evalúan como antes (usando p_unit)
            _ => eval_noise(p_unit, uniforms, &self.noise),
        };

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
