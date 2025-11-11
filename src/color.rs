use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Color {
  r: u8,
  g: u8,
  b: u8,
}

impl Color {
  pub fn new(r: u8, g: u8, b: u8) -> Self {
    Color { r, g, b }
  }

  pub fn black() -> Self {
    Color { r: 0, g: 0, b: 0 }
  }

  pub fn from_float(r: f32, g: f32, b: f32) -> Self {
    Color {
      r: (r.clamp(0.0, 1.0) * 255.0) as u8,
      g: (g.clamp(0.0, 1.0) * 255.0) as u8,
      b: (b.clamp(0.0, 1.0) * 255.0) as u8,
    }
  }

  pub fn from_hex(hex: u32) -> Self {
    let r = ((hex >> 16) & 0xFF) as u8;
    let g = ((hex >> 8) & 0xFF) as u8;
    let b = (hex & 0xFF) as u8;
    Color { r, g, b }
  }

  pub fn to_hex(&self) -> u32 {
    ((self.r as u32) << 16) | ((self.g as u32) << 8) | (self.b as u32)
  }

  pub fn lerp(a: Color, b: Color, t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    let rf = a.r as f32 + (b.r as f32 - a.r as f32) * t;
    let gf = a.g as f32 + (b.g as f32 - a.g as f32) * t;
    let bf = a.b as f32 + (b.b as f32 - a.b as f32) * t;
    Color::new(rf.round() as u8, gf.round() as u8, bf.round() as u8)
  }
}

use std::ops::Add;
impl Add for Color {
  type Output = Color;
  fn add(self, other: Color) -> Color {
    Color {
      r: self.r.saturating_add(other.r),
      g: self.g.saturating_add(other.g),
      b: self.b.saturating_add(other.b),
    }
  }
}

use std::ops::Mul;
impl Mul<f32> for Color {
  type Output = Color;
  fn mul(self, scalar: f32) -> Color {
    let s = scalar.clamp(0.0, 10.0);
    Color {
      r: (self.r as f32 * s).clamp(0.0, 255.0) as u8,
      g: (self.g as f32 * s).clamp(0.0, 255.0) as u8,
      b: (self.b as f32 * s).clamp(0.0, 255.0) as u8,
    }
  }
}

impl fmt::Display for Color {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Color(r: {}, g: {}, b: {})", self.r, self.g, self.b)
  }
}
