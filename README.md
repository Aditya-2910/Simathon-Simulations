# Simathon-Simulations

This is the first trial of the simulation of the blackhole, please run it using 

if python version>3.12 -> run

pip install uv
uv run --python 3.12 --with taichi sim.py 2>&1


# Black Hole Simulation Suite (Schwarzschild GR Ray-Marchers)

A series of real-time particle and ray-marched Schwarzschild black hole simulations implemented in Python using the high-performance **Taichi** graphics library. This suite demonstrates the technical evolution from a basic 2D particle simulation to a physical, volumetric geodesic ray-marcher with relativistic lensing, rendering a cinematic "Gargantua"-style black hole.

---

## 🚀 Evolution of Simulation Versions

### 🪐 Simulation 1: 2D Particle Accretion Disk (`sim.py` / Sim 1)
* **Visual Mode**: 2D flat face-on orthographic view.
* **Technique**: GPU particle rendering (180,000 accretion disk, 14,000 jet particles) splatted onto a pre-computed raster background.
* **Physics Covered**: Newtonian gravity + 1st-order Post-Newtonian (1PN) Schwarzschild precession term, relativistic Doppler beaming, gravitational redshift, and weak-field Einstein lensing on background stars.
* **Interactivity**: Left-click to inject gas clumps, right-click and drag to scale the black hole mass.

### 🪐 Simulation 2: Tilted Pseudo-3D & Motion Blur (`sim2.py` / Sim 2)
* **Visual Mode**: Tilted 3D perspective ($20^\circ$ elevation above the disk plane) resembling an *Interstellar* edge-on view.
* **Technique**: Precomputed 3D coordinate rotations projected onto screen coordinates. Relativistic light deflection from the back side of the disk is approximated via a rasterized wraparound lensing arc.
* **Key Additions**: 
  * Accretion disk particles store history to render smooth, motion-blurred orbital trails.
  * A spacecraft pod is introduced on a stable orbit near the photon sphere.

### 🪐 Simulation 3: Two-Pass Image-Space Lensing (`sim3.py` / Sim 3)
* **Technique**: Two-pass offscreen renderer.
  * **Pass A**: Offscreen buffer (`buffer_a`) renders the undistorted scene elements (particles, stars, jets, ship).
  * **Pass B**: A post-process pixel shader warps the offscreen image using Schwarzschild light deflection formulas, physically bending background stars and wrapping back-disk elements around the event horizon.
* **Key Advantage**: First version to apply uniform physical lensing distortion to both particle assets and background starfields.

### 🪐 Simulation 4: Geodesic Ray-Marching & Volumetric Disk (`sim4.py` / Sim 4)
* **Technique**: Real-time ray-tracing engine. Integrates light rays backwards from the camera using the Schwarzschild geodesic integration equations.
* **Accretion Disk**: Modeled as a true 3D volumetric density field. Density is procedurally sampled and modulated with 3D noise (Fractal Brownian Motion) to create realistic, granular, and dusty structures.
* **Key Additions**: Cinematic post-processing with multi-pass Bloom and exposure-based tone-mapping.

### 🪐 Simulation 5: Interactive Ray-Marching & Spacecraft (`sim5.py` / Sim 5)
* **Technique**: Interactive ray-marched renderer.
* **Key Additions**:
  * **Real-time Navigation**: Click and drag mouse to orbit the camera, zoom in/out with W/S, and adjust camera elevation with A/D.
  * **3D Spacecraft Pod**: Renders a physical spacecraft sphere orbiting outside the ISCO, with ray-marching registering ray-pod intersection and rendering a lensed, glowing blue silhouette.

  CONTROLS:
    MOUSE DRAG   - Orbit Black Hole
    W / S        - Zoom In / Out
    A / D        - Move Camera Up / Down
    SPACE        - Pause
    R            - Reset
---
