// noise.rs
// Conjunto de ruidos 2D/3D proyectados para superficie de esfera, más fBm.
// Incluye: Value, Perlin, Voronoi (F1) y utilidades.

use crate::shaders::VoronoiDistance;
use nalgebra_glm::Vec3;

pub fn hash(mut x: i32) -> i32 {
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4eb2d);
    x ^= x >> 15;
    x
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// ==================== VALUE NOISE ====================

pub fn value_noise_2d(x: f32, y: f32, seed: i32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;

    let h00 = hash(seed ^ (xi.wrapping_mul(374761393) ^ yi.wrapping_mul(668265263)));
    let h10 = hash(seed ^ ((xi + 1).wrapping_mul(374761393) ^ yi.wrapping_mul(668265263)));
    let h01 = hash(seed ^ (xi.wrapping_mul(374761393) ^ (yi + 1).wrapping_mul(668265263)));
    let h11 = hash(seed ^ ((xi + 1).wrapping_mul(374761393) ^ (yi + 1).wrapping_mul(668265263)));

    let v00 = (h00 & 0xffff) as f32 / 65535.0;
    let v10 = (h10 & 0xffff) as f32 / 65535.0;
    let v01 = (h01 & 0xffff) as f32 / 65535.0;
    let v11 = (h11 & 0xffff) as f32 / 65535.0;

    let u = fade(xf);
    let v = fade(yf);

    let x1 = lerp(v00, v10, u);
    let x2 = lerp(v01, v11, u);
    lerp(x1, x2, v)
}

/// fBm sobre value noise, combinando tres proyecciones (XY, YZ, ZX) para evitar seams.
pub fn fbm_value_3proj(p: Vec3, scale: f32, time: f32, oct: u32, lac: f32, gain: f32, seed: i32) -> f32 {
    let s = scale;
    let t = time;
    let nxy = fbm_2d((p.x + t) * s, p.y * s, oct, lac, gain, seed);
    let nyz = fbm_2d((p.y + t) * s, p.z * s, oct, lac, gain, seed ^ 0x9E37);
    let nzx = fbm_2d((p.z + t) * s, p.x * s, oct, lac, gain, seed ^ 0xC2B2);
    (nxy + nyz + nzx) / 3.0
}

pub fn fbm_2d(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32, seed: i32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for i in 0..octaves {
        sum += amp * value_noise_2d(x * freq, y * freq, seed ^ (i as i32 * 1013));
        freq *= lacunarity;
        amp *= gain;
    }
    sum.clamp(0.0, 1.0)
}

/// ==================== PERLIN NOISE ====================

fn grad_from_hash(h: i32) -> (f32, f32) {
    // 8 gradientes
    match (h & 7) {
        0 => ( 1.0,  0.0),
        1 => (-1.0,  0.0),
        2 => ( 0.0,  1.0),
        3 => ( 0.0, -1.0),
        4 => ( 0.7071,  0.7071),
        5 => (-0.7071,  0.7071),
        6 => ( 0.7071, -0.7071),
        _ => (-0.7071, -0.7071),
    }
}

pub fn perlin_2d(x: f32, y: f32, seed: i32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let xf = x - xi as f32;
    let yf = y - yi as f32;

    let h00 = hash(seed ^ (xi.wrapping_mul(374761393) ^ yi.wrapping_mul(668265263)));
    let h10 = hash(seed ^ ((xi + 1).wrapping_mul(374761393) ^ yi.wrapping_mul(668265263)));
    let h01 = hash(seed ^ (xi.wrapping_mul(374761393) ^ (yi + 1).wrapping_mul(668265263)));
    let h11 = hash(seed ^ ((xi + 1).wrapping_mul(374761393) ^ (yi + 1).wrapping_mul(668265263)));

    let (gx00, gy00) = grad_from_hash(h00);
    let (gx10, gy10) = grad_from_hash(h10);
    let (gx01, gy01) = grad_from_hash(h01);
    let (gx11, gy11) = grad_from_hash(h11);

    let d00 = gx00 * xf       + gy00 * yf;
    let d10 = gx10 * (xf - 1.0) + gy10 * yf;
    let d01 = gx01 * xf       + gy01 * (yf - 1.0);
    let d11 = gx11 * (xf - 1.0) + gy11 * (yf - 1.0);

    let u = fade(xf);
    let v = fade(yf);

    // Interpolación bilineal suavizada
    let x1 = lerp(d00, d10, u);
    let x2 = lerp(d01, d11, u);
    // Perlin clásico produce [-1,1], remap a [0,1]
    (lerp(x1, x2, v) * 0.5 + 0.5).clamp(0.0, 1.0)
}

pub fn fbm_perlin_3proj(p: Vec3, scale: f32, time: f32, oct: u32, lac: f32, gain: f32, seed: i32) -> f32 {
    let s = scale;
    let t = time;
    let nxy = fbm_perlin_2d((p.x + t) * s, p.y * s, oct, lac, gain, seed);
    let nyz = fbm_perlin_2d((p.y + t) * s, p.z * s, oct, lac, gain, seed ^ 0x9E37);
    let nzx = fbm_perlin_2d((p.z + t) * s, p.x * s, oct, lac, gain, seed ^ 0xC2B2);
    (nxy + nyz + nzx) / 3.0
}

fn fbm_perlin_2d(x: f32, y: f32, octaves: u32, lacunarity: f32, gain: f32, seed: i32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for i in 0..octaves {
        sum += amp * perlin_2d(x * freq, y * freq, seed ^ (i as i32 * 733));
        freq *= lacunarity;
        amp *= gain;
    }
    sum.clamp(0.0, 1.0)
}

/// ==================== VORONOI ====================

fn jitter(seed: i32, x: i32, y: i32) -> (f32, f32) {
    let h = hash(seed ^ (x.wrapping_mul(1103515245) ^ y.wrapping_mul(12345)));
    let a = ((h & 0xffff) as f32) / 65535.0;
    let b = ((h >> 16) as f32) / 65535.0;
    (a, b)
}

/// Voronoi F1 normalizado a [0,1] sobre tres proyecciones XY,YZ,ZX con pesos opcionales.
pub fn voronoi_3proj(
    p: Vec3,
    scale: f32,
    cell_size: f32,
    w1: f32, w2: f32, w3: f32, _w4: f32, // 4º peso reservado
    dist_ty: VoronoiDistance,
    seed: i32,
    time: f32,
) -> f32 {
    fn voronoi2d(mut x: f32, mut y: f32, cell: f32, seed: i32, ty: VoronoiDistance) -> f32 {
        x /= cell; y /= cell;

        let xi = x.floor() as i32;
        let yi = y.floor() as i32;

        let mut dmin = f32::INFINITY;

        for yy in (yi-1)..=(yi+1) {
            for xx in (xi-1)..=(xi+1) {
                let (jx, jy) = jitter(seed, xx, yy);
                let fx = xx as f32 + jx;
                let fy = yy as f32 + jy;

                let dx = fx - x;
                let dy = fy - y;

                let d = match ty {
                    VoronoiDistance::Euclidean => (dx*dx + dy*dy).sqrt(),
                    VoronoiDistance::Manhattan => dx.abs() + dy.abs(),
                    VoronoiDistance::Chebyshev => dx.abs().max(dy.abs()),
                };

                if d < dmin { dmin = d; }
            }
        }
        // Normalización aproximada a [0,1] (depende del tipo de distancia)
        let norm = match ty {
            VoronoiDistance::Euclidean => (dmin / 1.5).min(1.0),
            VoronoiDistance::Manhattan => (dmin / 2.5).min(1.0),
            VoronoiDistance::Chebyshev => (dmin / 1.5).min(1.0),
        };
        1.0 - norm // células blancas, bordes oscuros
    }

    let t = time; // puedes sumar t a un eje si quieres “derrapar” ligeramente
    let v1 = voronoi2d((p.x + t) * scale, p.y * scale, cell_size, seed, dist_ty);
    let v2 = voronoi2d((p.y + t) * scale, p.z * scale, cell_size, seed ^ 0x55AA, dist_ty);
    let v3 = voronoi2d((p.z + t) * scale, p.x * scale, cell_size, seed ^ 0xA55A, dist_ty);

    let wsum = (w1 + w2 + w3).max(1e-6);
    (v1 * w1 + v2 * w2 + v3 * w3) / wsum
}
