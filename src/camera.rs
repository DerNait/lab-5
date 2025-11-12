use nalgebra_glm::{Mat4, Vec3};

pub struct Camera {
    pub position: Vec3,   // en las mismas unidades “de pantalla” que usas
    pub rotation: Vec3,   // pitch (x), yaw (y), roll (z)
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        let (sx, cx) = self.rotation.x.sin_cos();
        let (sy, cy) = self.rotation.y.sin_cos();
        let (sz, cz) = self.rotation.z.sin_cos();

        let rx = Mat4::new(
            1.0, 0.0,  0.0, 0.0,
            0.0, cx,  -sx, 0.0,
            0.0, sx,   cx, 0.0,
            0.0, 0.0,  0.0, 1.0,
        );
        let ry = Mat4::new(
             cy, 0.0, sy, 0.0,
             0.0, 1.0, 0.0, 0.0,
            -sy, 0.0, cy, 0.0,
             0.0, 0.0, 0.0, 1.0,
        );
        let rz = Mat4::new(
            cz, -sz, 0.0, 0.0,
            sz,  cz, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // La vista es la inversa de la pose de la cámara: R^T y -R^T * t
        let r = rz * ry * rx;
        let t = Mat4::new(
            1.0, 0.0, 0.0, -self.position.x,
            0.0, 1.0, 0.0, -self.position.y,
            0.0, 0.0, 1.0, -self.position.z,
            0.0, 0.0, 0.0,  1.0,
        );

        r.transpose() * t
    }
}
