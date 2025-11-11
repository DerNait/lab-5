use tobj;
use nalgebra_glm::{Vec2, Vec3};
use crate::vertex::Vertex;

pub struct Obj {
    meshes: Vec<Mesh>,
}

struct Mesh {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    texcoords: Vec<Vec2>,
    indices: Vec<u32>,
}

impl Obj {
    pub fn load(filename: &str) -> Result<Self, tobj::LoadError> {
        let (models, _) = tobj::load_obj(filename, &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        })?;

        let meshes = models.into_iter().map(|model| {
            let mesh = model.mesh;
            Mesh {
                vertices: mesh.positions.chunks(3)
                    .map(|v| Vec3::new(v[0], -v[1], -v[2]))
                    .collect(),
                normals: mesh.normals.chunks(3)
                    .map(|n| Vec3::new(n[0], -n[1], -n[2]))
                    .collect(),
                texcoords: mesh.texcoords.chunks(2)
                    .map(|t| Vec2::new(t[0], 1.0 - t[1]))
                    .collect(),
                indices: mesh.indices,
            }
        }).collect();

        Ok(Obj { meshes })
    }

    /// Recorre todas las mallas y llama al callback con cada triángulo (índices).
    pub fn for_each_face<F: FnMut(usize, usize, usize)>(&self, mut f: F) {
        for mesh in &self.meshes {
            for idx in mesh.indices.chunks(3) {
                if let [i0, i1, i2] = *idx {
                    f(i0 as usize, i1 as usize, i2 as usize);
                }
            }
        }
    }

    /// Devuelve vistas a los buffers (posición, normal, uv) de la primera malla.
    /// Si manejas múltiples mallas, puedes adaptar el render para recorrerlas todas.
    pub fn mesh_buffers(&self) -> (&[Vec3], &[Vec3], &[Vec2]) {
        let m = &self.meshes[0];
        (&m.vertices, &m.normals, &m.texcoords)
    }

    /// (Opcional) Si quieres vértices expandidos como antes.
    pub fn get_vertex_array(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();

        for mesh in &self.meshes {
            for &index in &mesh.indices {
                let position = mesh.vertices[index as usize];
                let normal = mesh.normals.get(index as usize)
                    .cloned()
                    .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
                let tex_coords = mesh.texcoords.get(index as usize)
                    .cloned()
                    .unwrap_or(Vec2::new(0.0, 0.0));

                vertices.push(Vertex::new(position, normal, tex_coords));
            }
        }

        vertices
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut min_v = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max_v = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);

        for mesh in &self.meshes {
            for p in &mesh.vertices {
                min_v.x = min_v.x.min(p.x);  min_v.y = min_v.y.min(p.y);  min_v.z = min_v.z.min(p.z);
                max_v.x = max_v.x.max(p.x);  max_v.y = max_v.y.max(p.y);  max_v.z = max_v.z.max(p.z);
            }
        }
        (min_v, max_v)
    }
}
