// framebuffer.rs

pub struct Framebuffer {
    pub width: usize,
    pub height: usize,
    pub buffer: Vec<u32>,
    pub zbuffer: Vec<f32>,
    background_color: u32,
    current_color: u32,
}

impl Framebuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Framebuffer {
            width,
            height,
            buffer: vec![0; width * height],
            zbuffer: vec![f32::INFINITY; width * height],
            background_color: 0x000000,
            current_color: 0xFFFFFF,
        }
    }

    pub fn clear(&mut self) {
        for pixel in self.buffer.iter_mut() {
            *pixel = self.background_color;
        }
        for depth in self.zbuffer.iter_mut() {
            *depth = f32::INFINITY;
        }
    }

    pub fn point(&mut self, x: usize, y: usize, depth: f32) {
        if x < self.width && y < self.height {
            let index = y * self.width + x;
            if self.zbuffer[index] > depth {
                self.buffer[index] = self.current_color;
                self.zbuffer[index] = depth;
            }
        }
    }

    pub fn set_background_color(&mut self, color: u32) {
        self.background_color = color;
    }

    pub fn set_current_color(&mut self, color: u32) {
        self.current_color = color;
    }

    /// Nuevo: dibuja un píxel con color directo y alpha (0..1). Respeta zbuffer.
    pub fn draw_rgba(&mut self, x: usize, y: usize, depth: f32, color: u32, alpha: f32) {
        if x >= self.width || y >= self.height { return; }
        let idx = y * self.width + x;

        if self.zbuffer[idx] > depth {
            if alpha >= 0.999 {
                // opaco
                self.buffer[idx] = color;
            } else if alpha > 0.0 {
                // semi-transparente: componemos
                let dst = self.buffer[idx];

                let sr = ((color >> 16) & 0xFF) as f32;
                let sg = ((color >> 8)  & 0xFF) as f32;
                let sb = ( color        & 0xFF) as f32;

                let dr = ((dst >> 16) & 0xFF) as f32;
                let dg = ((dst >> 8)  & 0xFF) as f32;
                let db = ( dst        & 0xFF) as f32;

                let a = alpha.clamp(0.0, 1.0);
                let rr = (sr * a + dr * (1.0 - a)).clamp(0.0, 255.0) as u32;
                let rg = (sg * a + dg * (1.0 - a)).clamp(0.0, 255.0) as u32;
                let rb = (sb * a + db * (1.0 - a)).clamp(0.0, 255.0) as u32;

                self.buffer[idx] = (rr << 16) | (rg << 8) | rb;
            }

            // IMPORTANTE: ahora también escribimos z aun con alpha,
            // para que el propio anillo se ocluya correctamente y no quede “cortado”.
            self.zbuffer[idx] = depth;
        }
    }
}
