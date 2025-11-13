# Lab 5: Shaders en Planetas 🌌

Mini renderizador 3D en software que muestra distintos planetas y una estrella usando **shaders procedurales** (Perlin, Voronoi, BandedGas, Flow, etc.) sobre modelos OBJ.

- Carga de modelo OBJ de esfera y anillos.
- Sombreado procedural para:
  - Tierra con océanos, continentes y nubes animadas.
  - Gigante gaseoso tipo Júpiter.
  - Planeta rocoso tipo Marte con tormentas de polvo.
  - Urano pastel con bandas suaves.
  - Saturno con bandas amarillas y anillos.
  - Estrella / sol con “lava” en movimiento.
- Z-buffer, iluminación difusa y capas con alpha para nubes/anillos.
- Lunas orbitando el planeta con shaders propios.

---

## 🎥 Video de demostración

[![Demo - Lab 5 Shaders en Planetas](https://img.youtube.com/vi/8V3RQKlX4dk/0.jpg)](https://www.youtube.com/watch?v=8V3RQKlX4dk)

---

## 📸 Capturas

![Render](captura%201.png)
![Render](captura%202.png)
![Render](captura%203.png)
![Render](captura%204.png)
![Render](captura%205.png)
![Render](captura%206.png)

---

## 🎮 Controles

### Cámara / zoom
- **Flechas**: mover la cámara en X/Y.
- **A / S**: alejar / acercar (zoom vía escala del modelo).
- **Q / W**: rotar cámara en eje **X** (pitch).
- **E / R**: rotar cámara en eje **Y** (yaw).
- **T / Y**: rotar cámara en eje **Z** (roll).

### Selección de planeta / estrella
- **1** – Tierra con nubes.
- **2** – Gigante gaseoso tipo Júpiter.
- **3** – Marte rocoso con nubes finas.
- **4** – Urano (tonos azulados pastel).
- **5** – Saturno (bandas amarillas + anillos).
- **6** – Estrella / sol con superficie de “lava”.

### Anillos y lunas
- **Z**: activar/desactivar anillos.
- **X**: activar/desactivar luna 1.
- **C**: activar/desactivar luna 2.

---

## 🛠 Detalles técnicos

- Rasterización por triángulos en CPU con **z-buffer**.
- Shaders procedurales basados en:
  - Ruido Perlin / Value / Voronoi.
  - Shaders tipo **BandedGas** para planetas gaseosos.
  - Flow maps para animar bandas y “lava”.
  - Gradientes radiales para anillos.
- Iluminación difusa simple con vector de luz configurable.
- Soporte de múltiples capas con alpha (nubes, atmósferas, anillos).
