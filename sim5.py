"""
sim5.py -- Schwarzschild Black Hole: Version 5 (Interactive Ray-Marcher)
Based on Version 4, adding interactive camera controls and a spacecraft pod.

Interaction:
  - Mouse Click + Drag: Orbit the camera around the Black Hole.
  - W / S Keys: Zoom in and out.
  - A / D Keys: Adjust camera height.
  - SPACE: Toggle Pause.
  - R: Reset simulation state.

Features:
  - Geodesic Ray-Marching (Physical lensing).
  - Volumetric Accretion Disk (Procedural textures).
  - Spacecraft Pod: A visible ship on a stable orbit near the ISCO.
  - Cinematic Bloom & Tone-mapping.

Run:
    uv run --python 3.12 --with taichi sim5.py
"""

import math
import numpy as np
import taichi as ti

# =============================================================================
# 1. TAICHI INIT
# =============================================================================
try:
    ti.init(arch=ti.gpu)
    print("[Taichi] GPU backend initialized.")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Running on CPU. Rendering will be slow.")

# =============================================================================
# 2. CONSTANTS & PARAMETERS
# =============================================================================
RES_W, RES_H = 1280, 800
MAX_STEPS = 250         # Ray integration steps
STEP_SIZE = 0.14
INFINITY  = 60.0

# Physical Radii (Normalized units where GM = 1, c = 1 => Rs = 2.0)
R_S    = 2.0
R_ISCO = 6.0
ADISK_INNER  = 2.6      # Inner disk near shadow
ADISK_OUTER  = 20.0
ADISK_HEIGHT = 0.20

# Colors
COLOR_HOT   = ti.Vector([1.0, 0.9, 0.75])
COLOR_COLD  = ti.Vector([1.0, 0.4, 0.1])
COLOR_SHIP  = ti.Vector([0.3, 0.7, 1.0])  # Blue-ish glowing ship

# Fields
canvas       = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))
hdr_buffer   = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))
bloom_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))

# Interactive Camera State
# cam_angle: [yaw, pitch, distance]
cam_state = ti.Vector.field(3, dtype=ti.f32, shape=())
# ship_pos: [x, y, z]
ship_pos  = ti.Vector.field(3, dtype=ti.f32, shape=())

# =============================================================================
# 3. NOISE & MATH HELPERS
# =============================================================================
@ti.func
def fract(x): return x - ti.floor(x)

@ti.func
def smoothstep(edge0, edge1, x):
    t = ti.max(0.0, ti.min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)

@ti.func
def hash33(p):
    p = ti.Vector([
        ti.sin(p[0] * 127.1 + p[1] * 311.7 + p[2] * 74.7),
        ti.sin(p[0] * 269.5 + p[1] * 183.3 + p[2] * 246.1),
        ti.sin(p[0] * 113.5 + p[1] * 271.9 + p[2] * 124.6)
    ])
    return -1.0 + 2.0 * fract(p * 43758.5453123)

@ti.func
def noise3d(p):
    i, f = ti.floor(p), fract(p)
    u = f * f * (3.0 - 2.0 * f)
    res = 0.0
    for di in ti.static(range(2)):
        for dj in ti.static(range(2)):
            for dk in ti.static(range(2)):
                corner = ti.Vector([float(di), float(dj), float(dk)])
                grad = hash33(i + corner)
                disp = f - corner
                weight = 1.0
                if di == 1: weight *= u.x
                else: weight *= (1.0 - u.x)
                if dj == 1: weight *= u.y
                else: weight *= (1.0 - u.y)
                if dk == 1: weight *= u.z
                else: weight *= (1.0 - u.z)
                res += weight * grad.dot(disp)
    return res

@ti.func
def fbm(p, octaves=4):
    res, amp, freq = 0.0, 0.5, 1.0
    for _ in range(octaves):
        res += amp * noise3d(p * freq)
        amp *= 0.5
        freq *= 2.0
    return res

# =============================================================================
# 4. VOLUMETRIC RAY-MARCHING
# =============================================================================
@ti.func
def accel(pos, h2):
    """Geodesic curvature term."""
    r2 = pos.norm_sqr()
    return -1.5 * h2 * pos / (r2 * r2 * ti.sqrt(r2) + 1e-6)

@ti.func
def get_disk_density(pos, time):
    r = ti.sqrt(pos.x**2 + pos.z**2)
    density = 0.0
    if ADISK_INNER < r < ADISK_OUTER:
        # Volumetric band
        if ti.abs(pos.y) < ADISK_HEIGHT:
            h_fac = 1.0 - ti.abs(pos.y) / ADISK_HEIGHT
            r_fac = (ADISK_OUTER - r) / (ADISK_OUTER - ADISK_INNER)
            density = h_fac * r_fac * 1.5
            # Rotational noise
            ang = ti.atan2(pos.z, pos.x) + time * 0.4 * (1.0 / (r + 1.0))
            n = fbm(ti.Vector([r * 0.4, pos.y * 4.0, ang * r * 0.7]))
            density *= (0.7 + 0.3 * n)
            # Edge smoothing
            density *= smoothstep(ADISK_INNER, ADISK_INNER * 1.2, r)
    return ti.max(0.0, density)

@ti.func
def get_ship_hit(pos, s_pos):
    """Check if the ray hits the small pod silhouette."""
    dist = (pos - s_pos).norm()
    res = 0.0
    if dist < 0.25: # Pod radius
        res = 1.0 - dist / 0.25
    return res

@ti.func
def get_doppler(pos, dir):
    r = ti.sqrt(pos.x**2 + pos.z**2)
    v_tang = ti.Vector([-pos.z, 0.0, pos.x]) / r
    beta = 0.4 / ti.sqrt(r)
    cos_theta = dir.normalized().dot(v_tang)
    return 1.0 / (1.0 - beta * cos_theta + 1e-4)

@ti.kernel
def render(time: ti.f32):
    yaw, pitch, dist = cam_state[None]
    
    # Camera Cartesian Conversion
    cam_pos = ti.Vector([
        dist * ti.cos(yaw) * ti.cos(pitch),
        dist * ti.sin(pitch),
        dist * ti.sin(yaw) * ti.cos(pitch)
    ])
    
    target = ti.Vector([0.0, 0.0, 0.0])
    fov = 0.7
    
    # Orbiting ship position
    sr = 9.0 # Orbit radius
    s_ang = time * 0.3 # Orbit speed
    s_pos = ti.Vector([sr * ti.cos(s_ang), 0.2 * ti.sin(s_ang*2), sr * ti.sin(s_ang)])
    ship_pos[None] = s_pos # Sync to CPU later maybe

    # Standard LookAt
    fwd = (target - cam_pos).normalized()
    rit = ti.Vector([0, 1, 0]).cross(fwd).normalized()
    upw = fwd.cross(rit).normalized()

    for i, j in canvas:
        uv = ti.Vector([(i - RES_W * 0.5) / RES_H, (j - RES_H * 0.5) / RES_H])
        
        pos = cam_pos
        dir = (fwd + uv.x * rit * fov + uv.y * upw * fov).normalized()
        
        h2 = (pos.cross(dir)).norm_sqr()
        accum_color = ti.Vector([0.0, 0.0, 0.0])
        opacity     = 1.0
        
        # Ray Integration Loop
        for step in range(MAX_STEPS):
            # 1. Physics update
            dir += accel(pos, h2) * STEP_SIZE
            pos += dir * STEP_SIZE
            rSq  = pos.norm_sqr()
            
            # 2. Terminal Checks
            if rSq < (R_S * 0.48)**2: # Sucked in
                opacity = 0.0
                break
            if rSq > INFINITY**2: # Escaped to stars
                bg_n = fbm(dir * 120.0, 1)
                if bg_n > 0.94:
                    accum_color += ti.Vector([1, 1, 1]) * opacity * ((bg_n - 0.94) * 25.0)
                break
            
            # 3. Object checks: Ship/Pod
            s_hit = get_ship_hit(pos, s_pos)
            if s_hit > 0.001:
                accum_color += COLOR_SHIP * (s_hit * 4.0) * opacity
                opacity *= (1.0 - s_hit)
            
            # 4. Volumetric Dish Sampling
            if ti.abs(pos.y) < ADISK_HEIGHT * 1.5:
                dens = get_disk_density(pos, time)
                if dens > 0.001:
                    r = ti.sqrt(pos.x**2 + pos.z**2)
                    r_t = (r - ADISK_INNER) / (ADISK_OUTER - ADISK_INNER)
                    col = COLOR_HOT * (1.0 - r_t) + COLOR_COLD * r_t
                    
                    # Beaming & Redshift
                    col *= get_doppler(pos, dir)**3
                    col *= ti.sqrt(ti.max(0.0, 1.0 - R_S / r))
                    
                    # Accumulate (standard scattering approx)
                    alpha = ti.min(1.0, dens * STEP_SIZE * 0.6)
                    accum_color += col * dens * STEP_SIZE * opacity
                    opacity *= (1.0 - alpha)
                    
            if opacity < 0.01: break

        hdr_buffer[i, j] = accum_color

# =============================================================================
# 5. POST-PROCESSING (Bloom & Mix)
# =============================================================================
@ti.kernel
def bloom_extract():
    for i, j in hdr_buffer:
        val = hdr_buffer[i, j]
        if val.norm() > 0.8:
            bloom_buffer[i, j] = val
        else:
            bloom_buffer[i, j] = ti.Vector([0, 0, 0])

@ti.kernel
def composite(exposure: ti.f32):
    for i, j in canvas:
        col = hdr_buffer[i, j] + bloom_buffer[i, j] * 0.45
        canvas[i, j] = 1.0 - ti.exp(-col * exposure)

@ti.kernel
def bloom_blur(horizontal: ti.template()):
    for i, j in canvas:
        res = ti.Vector([0.0, 0.0, 0.0])
        w = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216]
        if ti.static(horizontal):
            res += bloom_buffer[i, j] * w[0]
            for k in ti.static(range(1, 5)):
                if i + k < RES_W: res += bloom_buffer[i+k, j] * w[k]
                if i - k >= 0:    res += bloom_buffer[i-k, j] * w[k]
        else:
            res += bloom_buffer[i, j] * w[0]
            for k in ti.static(range(1, 5)):
                if j + k < RES_H: res += bloom_buffer[i, j+k] * w[k]
                if j - k >= 0:    res += bloom_buffer[i, j-k] * w[k]
        bloom_buffer[i, j] = res

# =============================================================================
# 6. APP CONTROLLER
# =============================================================================
def main():
    print("="*65)
    print("  SCHWARZSCHILD V5: INTERACTIVE RAY-MARCHER")
    print("="*65)
    print("  CONTROLS:")
    print("    MOUSE DRAG   - Orbit Black Hole")
    print("    W / S        - Zoom In / Out")
    print("    A / D        - Move Camera Up / Down")
    print("    SPACE        - Pause")
    print("    R            - Reset")
    print("="*65)

    gui = ti.GUI("Blackhole V5 - Interactive", (RES_W, RES_H), fast_gui=True)
    
    # Initial Camera: [yaw, pitch, distance]
    cam_angle = [0.0, 0.2, 18.0]
    cam_state[None] = cam_angle
    
    time = 0.0
    paused = False
    
    while gui.running:
        # Handle Input
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.SPACE: paused = not paused
            elif e.key == 'r': 
                cam_angle = [0.0, 0.2, 18.0]
                time = 0.0

        if gui.is_pressed(ti.GUI.LMB):
            curr_mouse = gui.get_cursor_pos()
            # Yaw (horizontal orbit) - simple mapping
            cam_angle[0] += (curr_mouse[0] - 0.5) * 0.1
            # Pitch (vertical orbit)
            cam_angle[1] = np.clip(cam_angle[1] + (curr_mouse[1] - 0.5) * 0.05, -1.5, 1.5)

        if gui.is_pressed('w'): cam_angle[2] = max(5.0, cam_angle[2] - 0.2)
        if gui.is_pressed('s'): cam_angle[2] = min(50.0, cam_angle[2] + 0.2)
        if gui.is_pressed('a'): cam_angle[1] = min(1.5, cam_angle[1] + 0.02)
        if gui.is_pressed('d'): cam_angle[1] = max(-1.5, cam_angle[1] - 0.02)

        cam_state[None] = cam_angle
        
        # Simulation Step
        if not paused:
            time += 0.04
        
        # Render Pipeline
        render(time)
        bloom_extract()
        for _ in range(2):
            bloom_blur(True)
            bloom_blur(False)
        composite(1.25)
        
        gui.set_image(canvas)
        gui.show()

if __name__ == "__main__":
    main()
