"""
sim4.py -- Schwarzschild Black Hole: Version 4
Inspired by rossning92/Blackhole (Real-time Ray-Marching in Curved Spacetime)

Technique:
  1. Geodesic Integration: Rays are integrated using the Schwarzschild geodesic 
     approximation, solving for the path of light in curved spacetime.
  2. Volumetric Ray-Marching: The accretion disk is rendered as a volumetric 
     entity. We sample density and lighting along the ray path.
  3. Procedural Texturing: Use 3D Simplex-like noise to create the "dusty" 
     and "granular" look of the disk.
  4. Post-processing: Multi-pass Bloom and Tonemapping for cinematic output.

Run:
    uv run --python 3.12 --with taichi sim4.py
"""

import math
import numpy as np
import taichi as ti

# =============================================================================
# 1. TAICHI INIT
# =============================================================================
try:
    ti.init(arch=ti.gpu, kernel_profiler=True)
    print("[Taichi] GPU backend initialized.")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Running on CPU. This will be SLOW. Reduce resolution.")

# =============================================================================
# 2. CONSTANTS & PARAMETERS
# =============================================================================
RES_W, RES_H = 1280, 800
MAX_STEPS = 240         # Integration steps per ray
STEP_SIZE = 0.15        # Integration step size
INFINITY  = 60.0        # Max distance for rays

# Schwarzschild Radii (Normalized units where R_s = 2.0)
R_S    = 2.0
R_ISCO = 6.0            # Inner-most stable circular orbit
ADISK_HEIGHT = 0.18     # Vertical thickness of the disk
ADISK_INNER  = 2.6
ADISK_OUTER  = 18.0

# Colors (Blackbody approximation)
COLOR_HOT  = ti.Vector([1.0, 0.9, 0.7])  # Near ISCO
COLOR_COLD = ti.Vector([1.0, 0.3, 0.05]) # Outer edge

# Fields
# Final image to display
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))
# Intermediate buffer for bloom
hdr_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))
bloom_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(RES_W, RES_H))

# =============================================================================
# 3. NOISE FUNCTIONS (Simplex approximation in Taichi)
# =============================================================================
@ti.func
def fract(x):
    return x - ti.floor(x)

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
    """3D Perlin-like noise."""
    i = ti.floor(p)
    f = fract(p)
    
    # Smoothstep interpolation
    u = f * f * (3.0 - 2.0 * f)
    
    # 8 corners of the cube
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
                res += weight * dot(grad, disp)
    
    return res

@ti.func
def fbm(p, octaves=4):
    """Fractal Brownian Motion."""
    res = 0.0
    amp = 0.5
    freq = 1.0
    for _ in range(octaves):
        res += amp * noise3d(p * freq)
        amp *= 0.5
        freq *= 2.0
    return res

@ti.func
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# =============================================================================
# 4. PHYSICS & RENDERING
# =============================================================================
@ti.func
def accel(pos, h2):
    """Schwarzschild geodesic acceleration approximation."""
    r2 = dot(pos, pos)
    r = ti.sqrt(r2)
    # The term -1.5 * h^2 / r^5 * pos (or similar depending on derivation)
    # This is from the paper: "Fast Approximation of Schwarzschild Geodesics"
    # We use a simplified form that gives the correct visual lensing.
    return -1.5 * h2 * pos / (r2 * r2 * r + 1e-6)

@ti.func
def get_disk_density(pos, time):
    """Sample the volumetric density of the accretion disk."""
    r = ti.sqrt(pos.x*pos.x + pos.z*pos.z)
    density = 0.0
    
    if ADISK_INNER < r < ADISK_OUTER:
        # Vertical Gaussian-like profile
        y_dist = ti.abs(pos.y)
        if y_dist < ADISK_HEIGHT:
            h_fac = 1.0 - y_dist / ADISK_HEIGHT
            # Radial profile: denser towards center
            r_fac = (ADISK_OUTER - r) / (ADISK_OUTER - ADISK_INNER)
            density = h_fac * r_fac * 1.5
            
            # Add procedural noise
            # Map Cartesian to simple cylindrical coords for noise rotation
            angle = ti.atan2(pos.z, pos.x)
            # Adjust angle by time for rotation effect
            rot_angle = angle + time * 0.5 * (1.0 / (r + 1.0))
            noise_pos = ti.Vector([r * 0.5, pos.y * 3.0, rot_angle * r * 0.8])
            n = fbm(noise_pos)
            density *= (0.6 + 0.4 * n)
            
            # Smoothstep near inner edge
            density *= smoothstep(ADISK_INNER, ADISK_INNER * 1.5, r)
            
    return ti.max(0.0, density)

@ti.func
def get_doppler_factor(pos, dir, time):
    """Simple Doppler beaming approximation."""
    # Tangential velocity vector at pos
    r = ti.sqrt(pos.x**2 + pos.z**2)
    v_tang = ti.Vector([-pos.z, 0.0, pos.x]) / r
    # beta proportional to sqrt(1/r)
    beta = 0.4 / ti.sqrt(r)
    # Cosine of angle between ray direction and velocity
    cos_theta = dot(dir.normalized(), v_tang)
    # delta = 1 / (gamma * (1 - beta * cos_theta))
    # We'll use a power to make it obvious
    return 1.0 / (1.0 - beta * cos_theta + 1e-4)

@ti.func
def look_at(cam_pos, target, uv, fov):
    forward = (target - cam_pos).normalized()
    right = ti.Vector([0.0, 1.0, 0.0]).cross(forward).normalized()
    up = forward.cross(right).normalized()
    return (forward + uv.x * right * fov + uv.y * up * fov).normalized()

@ti.kernel
def render(time: ti.f32):
    # Camera setup (Orbiting slow)
    cam_dist = 18.0
    cam_h = 4.0 * ti.sin(time * 0.15)
    cam_pos = ti.Vector([
        cam_dist * ti.cos(time * 0.2),
        cam_h,
        cam_dist * ti.sin(time * 0.2)
    ])
    target = ti.Vector([0.0, 0.0, 0.0])
    fov = 0.6

    for i, j in canvas:
        uv = ti.Vector([(i - RES_W * 0.5) / RES_H, (j - RES_H * 0.5) / RES_H])
        
        # Ray Setup
        pos = cam_pos
        dir = look_at(cam_pos, target, uv, fov)
        
        # Conserved angular momentum h = r x v_initial
        # Here v_initial is the ray direction
        h = pos.cross(dir)
        h2 = h.norm_sqr()
        
        # Accumulators
        accum_color = ti.Vector([0.0, 0.0, 0.0])
        opacity = 1.0
        
        for _ in range(MAX_STEPS):
            # 1. Geodesic update
            acc = accel(pos, h2)
            dir += acc * STEP_SIZE
            pos += dir * STEP_SIZE
            
            rSq = dot(pos, pos)
            
            # Hit check: Event Horizon
            if rSq < (R_S * 0.5)**2: # Horizon in ray tracing usually sits at Rs
                opacity = 0.0
                break
            
            # Exit check: Deep Space
            if rSq > INFINITY**2:
                # Background Starfield (Procedural)
                # Use dir as direction to starfield
                n = fbm(dir * 100.0, 1)
                if n > 0.95:
                    accum_color += ti.Vector([1.0, 1.0, 1.0]) * opacity * ((n-0.95)*20.0)
                break
            
            # 2. Disc Interior Sampling
            # We check if we are within the vertical bounds of the disk
            if ti.abs(pos.y) < ADISK_HEIGHT:
                density = get_disk_density(pos, time)
                if density > 0.001:
                    # Color based on radius
                    r = ti.sqrt(pos.x*pos.x + pos.z*pos.z)
                    r_t = (r - ADISK_INNER) / (ADISK_OUTER - ADISK_INNER)
                    col = COLOR_HOT * (1.0 - r_t) + COLOR_COLD * r_t
                    
                    # Doppler beaming
                    doppler = get_doppler_factor(pos, dir, time)
                    col *= doppler**3
                    
                    # Gravitational redshift sqrt(1 - Rs/r)
                    redshift = ti.sqrt(ti.max(0.0, 1.0 - R_S / r))
                    col *= redshift
                    
                    # Accumulate
                    sample_alpha = ti.min(1.0, density * STEP_SIZE * 0.5)
                    accum_color += col * density * STEP_SIZE * opacity
                    opacity *= (1.0 - sample_alpha)
            
            # Early exit if opaque
            if opacity < 0.01:
                break

        # Tonemapping (Simple Exposure)
        hdr_buffer[i, j] = accum_color

# =============================================================================
# 5. POST-PROCESSING (Bloom)
# =============================================================================
@ti.kernel
def bloom_extract():
    for i, j in hdr_buffer:
        val = hdr_buffer[i, j]
        bright = dot(val, ti.Vector([0.299, 0.587, 0.114]))
        if bright > 0.8:
            bloom_buffer[i, j] = val
        else:
            bloom_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def bloom_blur(horizontal: ti.template()):
    for i, j in canvas:
        res = ti.Vector([0.0, 0.0, 0.0])
        weights = [0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216]
        if ti.static(horizontal):
            res += bloom_buffer[i, j] * weights[0]
            for k in ti.static(range(1, 5)):
                if i + k < RES_W: res += bloom_buffer[i + k, j] * weights[k]
                if i - k >= 0:    res += bloom_buffer[i - k, j] * weights[k]
        else:
            res += bloom_buffer[i, j] * weights[0]
            for k in ti.static(range(1, 5)):
                if j + k < RES_H: res += bloom_buffer[i, j + k] * weights[k]
                if j - k >= 0:    res += bloom_buffer[i, j - k] * weights[k]
        bloom_buffer[i, j] = res

@ti.kernel
def composite(exposure: ti.f32):
    for i, j in canvas:
        hdr = hdr_buffer[i, j]
        bloom = bloom_buffer[i, j]
        # Mix
        color = hdr + bloom * 0.4
        # Exposure tonemapping
        color = 1.0 - ti.exp(-color * exposure)
        canvas[i, j] = color

# =============================================================================
# 6. MAIN LOOP
# =============================================================================
def main():
    print("=" * 60)
    print("  SCHWARZSCHILD RAY-MARCHER - Cinematic Simulation")
    print("=" * 60)
    print("  Technique: Geodesic integration + Volumetric Sampling")
    print("  Reference: Gargantua (Interstellar)")
    print("=" * 60)
    
    gui = ti.GUI("Blackhole Raymarcher", (RES_W, RES_H), fast_gui=True)
    
    counter = 0
    while gui.running:
        time = counter * 0.03
        
        # 1. Main Render Pass
        render(time)
        
        # 2. Bloom Passes
        bloom_extract()
        for _ in range(2):
            bloom_blur(True)
            bloom_blur(False)
        
        # 3. Composite
        composite(1.4)
        
        gui.set_image(canvas)
        gui.show()
        counter += 1

if __name__ == "__main__":
    main()
