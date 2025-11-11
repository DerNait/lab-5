// noise.rs
// Value noise 2D + fBm para shading procedural (CPU)

pub fn hash(mut x: i32) -> i32 {
    // Pequeño hash determinista
    x = (x ^ 61) ^ (x >> 16);
    x = x.wrapping_add(x << 3);
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4eb2d);
    x ^= x >> 15;
    x
}

#[inline]
fn fade(t: f32) -> f32 {
    // Suaviza (curva tipo Perlin)
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Value noise 2D en [0,1]
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

// fBm (fractal Brownian motion) combinando value noise
pub fn fbm_2d(mut x: f32, mut y: f32, octaves: u32, lacunarity: f32, gain: f32, seed: i32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    for _ in 0..octaves {
        sum += amp * value_noise_2d(x * freq, y * freq, seed);
        freq *= lacunarity;
        amp *= gain;
    }
    // Normaliza a [0,1] aprox
    sum.clamp(0.0, 1.0)
}
