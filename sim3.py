"""
sim3.py -- Schwarzschild Black Hole: Version 3 (Gargantua Style)
  ● Two-pass Image-Space Lensing (Post-process warp)
  ● Tilted Thin Accretion Disk (20 degrees)
  ● Relativistic Doppler Beaming & Gravitational Redshift
  ● Twin Helical Jets & Spacecraft Pod
  ● Authentic "Interstellar" wraparound lensing look

Run with:
    uv run --python 3.12 --with taichi sim3.py
"""

import math
import numpy as np
import taichi as ti

# -----------------------------------------------------------------------------
# 1. TAICHI INIT
# -----------------------------------------------------------------------------
try:
    ti.init(arch=ti.gpu)
    print("[Taichi] GPU backend initialised.")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Running on CPU. Lower N_DISK if FPS < 30.")

# -----------------------------------------------------------------------------
# 2. PHYSICAL CONSTANTS & UNIT SYSTEM
# -----------------------------------------------------------------------------
GM      = 1.0          # G * M_bh
C_LIGHT = 1.0          # Speed of light (dimensionless units)

# Characteristic radii
R_S    = 2.0 * GM / (C_LIGHT**2)  # Schwarzschild Radius = 2.0
R_PH   = 1.5 * R_S                 # Photon Sphere = 3.0
R_ISCO = 3.0 * R_S                 # ISCO = 6.0

# Canvas & World units
CANVAS_W, CANVAS_H = 1280, 800
SCALE = 25.0 / R_S            # 25 pixels per R_S (event horizon ~ 50px wide)

# Disk parameters
R_DISK_IN  = R_ISCO
R_DISK_OUT = 35.0 * R_S

# Camera Tilt (Interstellar-style ~20 degrees off edge-on)
THETA_TILT = math.radians(20.0)
COS_T      = math.cos(THETA_TILT)
SIN_T      = math.sin(THETA_TILT)

# Lensing Post-Process parameters
# We use a deflection approximation based on the impact parameter b
# alpha(b) = 4GM / (c^2 b)
LENS_STRENGTH = 4.0 * GM / (C_LIGHT**2)

EPS = 1e-6

# -----------------------------------------------------------------------------
# 3. PARTICLE COUNTS
# -----------------------------------------------------------------------------
N_DISK  = 200_000
N_JET   =  12_000
N_STARS =   3_000

# -----------------------------------------------------------------------------
# 4. TAICHI FIELDS
# -----------------------------------------------------------------------------
# Disk (3D particles in x-y plane, projected in Pass A)
pos_d   = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)
vel_d   = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)
col_d   = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)
life_d  = ti.field(dtype=ti.f32, shape=N_DISK)

# Jets (3D)
pos_j   = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
vel_j   = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
col_j   = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
life_j  = ti.field(dtype=ti.f32, shape=N_JET)

# Background Stars (2D coords)
star_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)
star_col = ti.Vector.field(3, dtype=ti.f32, shape=N_STARS)

# Spacecraft (3D)
ship_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
ship_vel = ti.Vector.field(3, dtype=ti.f32, shape=1)

# Buffer A: Offscreen render (Undistorted Scene)
buffer_a = ti.Vector.field(3, dtype=ti.f32, shape=(CANVAS_W, CANVAS_H))
# Final Image: Lensed Scene
canvas_img = ti.Vector.field(3, dtype=ti.f32, shape=(CANVAS_W, CANVAS_H))

# Interactive
bh_mass_mul = ti.field(dtype=ti.f32, shape=())
inject_coord = ti.Vector.field(2, dtype=ti.f32, shape=1)

# -----------------------------------------------------------------------------
# 5. COLOR PALETTE (Blackbody approximation)
# -----------------------------------------------------------------------------
PALETTE_N = 512
palette = ti.Vector.field(3, dtype=ti.f32, shape=PALETTE_N)

def build_palette_np():
    pal = np.zeros((PALETTE_N, 3), dtype=np.float32)
    for i in range(PALETTE_N):
        t = i / (PALETTE_N - 1.0)
        if t < 0.2:
            u = t / 0.2
            r, g, b = 0.4 + 0.5*u, 0.05*u, 0.0
        elif t < 0.5:
            u = (t - 0.2) / 0.3
            r, g, b = 0.9 + 0.1*u, 0.1 + 0.6*u, 0.02*u
        elif t < 0.8:
            u = (t - 0.5) / 0.3
            r, g, b = 1.0, 0.7 + 0.3*u, 0.05 + 0.4*u
        else:
            u = (t - 0.8) / 0.2
            r, g, b = 1.0 - 0.1*u, 1.0, 0.5 + 0.5*u
        pal[i] = np.clip([r, g, b], 0.0, 1.0)
    return pal

@ti.kernel
def upload_palette(pal_np: ti.types.ndarray()):
    for i in range(PALETTE_N):
        palette[i] = ti.Vector([pal_np[i, 0], pal_np[i, 1], pal_np[i, 2]])

@ti.func
def sample_palette(t: ti.f32):
    i = ti.cast(ti.max(0.0, ti.min(1.0, t)) * (PALETTE_N - 1), ti.i32)
    return palette[i]

# -----------------------------------------------------------------------------
# 6. PHYSICS HELPERS
# -----------------------------------------------------------------------------
@ti.func
def get_v_circ(r: ti.f32, gm: ti.f32):
    return ti.sqrt(gm / (r + EPS))

@ti.func
def get_temp(r: ti.f32):
    return ti.min(1.0, (R_ISCO / r) ** 0.75)

@ti.func
def get_doppler(vx: ti.f32, vy: ti.f32):
    # Observer is at some angle. Simplified asymmetry: boost positive X velocity
    # delta = 1 / (gamma * (1 - beta * cos_theta))
    # Approximation for asymmetry:
    v_mag = ti.sqrt(vx*vx + vy*vy)
    beta = ti.min(v_mag / C_LIGHT, 0.99)
    gamma = 1.0 / ti.sqrt(1.0 - beta*beta)
    # Observer projected toward -y? No, fromInterstellar angle, 
    # one side flows toward us (left in Gargantua shot usually, or right)
    # We'll boost vx > 0.
    cos_theta = vx / (v_mag + EPS)
    delta = 1.0 / (gamma * (1.0 - beta * cos_theta) + EPS)
    return ti.pow(delta, 3.0)

@ti.func
def get_redshift(r: ti.f32):
    return ti.sqrt(ti.max(0.0, 1.0 - R_S / r))

@ti.func
def project_3d_to_2d(p):
    # World: Z is up, Disk is X-Y.
    # Tilt about X axis by THETA_TILT
    # y' = y * cos + z * sin
    # z' = -y * sin + z * cos
    # We discard z' for ortho projection (deep space)
    x = p[0]
    y = p[1] * COS_T + p[2] * SIN_T
    return ti.Vector([x, y])

# -----------------------------------------------------------------------------
# 7. INITIALIZATION
# -----------------------------------------------------------------------------
@ti.kernel
def init_all():
    gm = GM * bh_mass_mul[None]
    TPI = 2.0 * math.pi
    # Disk
    for i in range(N_DISK):
        u = ti.random()
        r = R_DISK_IN + (u * u) * (R_DISK_OUT - R_DISK_IN)
        ang = ti.random() * TPI
        vc = get_v_circ(r, gm)
        dv = (ti.random() - 0.5) * 0.05 * vc
        pos_d[i] = ti.Vector([r * ti.cos(ang), r * ti.sin(ang), (ti.random()-0.5)*0.1])
        vel_d[i] = ti.Vector([-(vc+dv) * ti.sin(ang), (vc+dv) * ti.cos(ang), 0.0])
        life_d[i] = ti.random() * 10.0
    # Jets
    for i in range(N_JET):
        side = 1.0 if i < N_JET // 2 else -1.0
        pos_j[i] = ti.Vector([0.0, 0.0, 0.0])
        vel_j[i] = ti.Vector([ti.random()-0.5, ti.random()-0.5, side * 1.5]).normalized() * C_LIGHT * 0.95
        life_j[i] = ti.random() * 5.0
    # Stars
    for i in range(N_STARS):
        star_pos[i] = ti.Vector([ti.random(), ti.random()])
        star_col[i] = ti.Vector([0.8, 0.8, 1.0]) * (0.5 + 0.5*ti.random())
    # Ship
    r_ship = R_PH * 1.8
    ship_pos[0] = ti.Vector([r_ship, 0.0, 0.0])
    ship_vel[0] = ti.Vector([0.0, get_v_circ(r_ship, gm), 0.0])

# -----------------------------------------------------------------------------
# 8. UPDATE KERNELS
# -----------------------------------------------------------------------------
@ti.kernel
def update_physics(dt: ti.f32):
    gm = GM * bh_mass_mul[None]
    TPI = 2.0 * math.pi
    # Disk
    for i in range(N_DISK):
        p, v = pos_d[i], vel_d[i]
        r = p.norm()
        # Simple Keplerian update
        acc = -gm / (r*r*r + EPS) * p
        v += acc * dt
        p += v * dt
        # Boundary / Absorption
        if r < R_S * 1.05 or r > R_DISK_OUT * 1.2:
            u = ti.random()
            r_new = R_DISK_IN + (u * u) * (R_DISK_OUT - R_DISK_IN)
            ang = ti.random() * TPI
            vc = get_v_circ(r_new, gm)
            p = ti.Vector([r_new * ti.cos(ang), r_new * ti.sin(ang), (ti.random()-0.5)*0.1])
            v = ti.Vector([-vc * ti.sin(ang), vc * ti.cos(ang), 0.0])
        pos_d[i], vel_d[i] = p, v
        # Color
        temp = get_temp(r)
        base_c = sample_palette(temp)
        beam = get_doppler(v[0], v[1])
        gz = get_redshift(r)
        col_d[i] = base_c * beam * gz

    # Jets
    for i in range(N_JET):
        p, v = pos_j[i], vel_j[i]
        p += v * dt
        # Helical twist
        angle = 10.0 * dt
        v.x, v.y = v.x*ti.cos(angle) - v.y*ti.sin(angle), v.x*ti.sin(angle) + v.y*ti.cos(angle)
        # Respawn
        if p.norm() > R_DISK_OUT * 0.8:
            side = 1.0 if v.z > 0 else -1.0
            p = ti.Vector([0.0, 0.0, 0.0])
            ang = ti.random() * TPI
            rad = ti.random() * 0.2
            v = ti.Vector([rad*ti.cos(ang), rad*ti.sin(ang), side * 1.5]).normalized() * C_LIGHT * 0.95
        pos_j[i], vel_j[i] = p, v
        col_j[i] = ti.Vector([0.5, 0.8, 1.0]) if v.z > 0 else ti.Vector([1.0, 0.4, 0.8])

    # Ship
    sp, sv = ship_pos[0], ship_vel[0]
    sr = sp.norm()
    s_acc = -gm / (sr*sr*sr + EPS) * sp
    # Grav. Time Dilation effect (approx)
    td = get_redshift(sr)
    sv += s_acc * dt * td
    sp += sv * dt * td
    ship_pos[0], ship_vel[0] = sp, sv

# -----------------------------------------------------------------------------
# 9. PASS A: RENDER UNDISTORTED
# -----------------------------------------------------------------------------
@ti.kernel
def render_pass_a():
    # Clear
    for i, j in buffer_a:
        buffer_a[i, j] = ti.Vector([0.0, 0.0, 0.02]) # Faint background blue

    cx, cy = CANVAS_W // 2, CANVAS_H // 2

    # Stars
    for i in range(N_STARS):
        px = ti.cast(star_pos[i].x * CANVAS_W, ti.i32)
        py = ti.cast(star_pos[i].y * CANVAS_H, ti.i32)
        if 0 <= px < CANVAS_W and 0 <= py < CANVAS_H:
            buffer_a[px, py] += star_col[i]

    # Disk Particles
    for i in range(N_DISK):
        p2 = project_3d_to_2d(pos_d[i])
        px = ti.cast(cx + p2.x * SCALE, ti.i32)
        py = ti.cast(cy + p2.y * SCALE, ti.i32)
        if 0 <= px < CANVAS_W and 0 <= py < CANVAS_H:
            # Additive blend
            buffer_a[px, py] += col_d[i] * 0.3
            # Simple splat
            if px + 1 < CANVAS_W: buffer_a[px+1, py] += col_d[i] * 0.05
            if py + 1 < CANVAS_H: buffer_a[px, py+1] += col_d[i] * 0.05

    # Jets
    for i in range(N_JET):
        p2 = project_3d_to_2d(pos_j[i])
        px = ti.cast(cx + p2.x * SCALE, ti.i32)
        py = ti.cast(cy + p2.y * SCALE, ti.i32)
        if 0 <= px < CANVAS_W and 0 <= py < CANVAS_H:
            buffer_a[px, py] += col_j[i] * 0.2

    # Spacecraft
    sp2 = project_3d_to_2d(ship_pos[0])
    spx = ti.cast(cx + sp2.x * SCALE, ti.i32)
    spy = ti.cast(cy + sp2.y * SCALE, ti.i32)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if 0 <= spx+dx < CANVAS_W and 0 <= spy+dy < CANVAS_H:
                buffer_a[spx+dx, spy+dy] = ti.Vector([1.0, 1.0, 1.0])

    # Event Horizon (Black hole itself)
    # We'll render it as a solid black circle over the background in Pass A
    # but the REAL silhouette is formed by the lensing pass.
    # However, to avoid "seeing through" the hole in pass A, we black it out.
    rs_px = ti.cast(R_S * SCALE, ti.i32)
    for x, y in buffer_a:
        if (x-cx)**2 + (y-cy)**2 < rs_px**2:
            buffer_a[x, y] = ti.Vector([0, 0, 0])

# -----------------------------------------------------------------------------
# 10. PASS B: LENSING DISTORTION
# -----------------------------------------------------------------------------
@ti.kernel
def render_pass_b():
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    
    for i, j in canvas_img:
        dx = (ti.cast(i, ti.f32) - cx) / SCALE
        dy = (ti.cast(j, ti.f32) - cy) / SCALE
        
        b = ti.sqrt(dx*dx + dy*dy) + EPS # Impact parameter
        
        # Shadow radius in lensing is roughly sqrt(27)*GM/c^2 ~ 2.6*R_S
        # We'll use a conservative threshold for the silhouette
        shadow_rad = R_S * 1.3
        
        col = ti.Vector([0.0, 0.0, 0.0])
        
        if b < shadow_rad:
            # Black Hole Silhouette
            col = ti.Vector([0, 0, 0])
        else:
            # LENSING WARP
            # Observed position P. 
            # We want to sample the undistorted buffer_a at a bent position.
            # The bending 'wraps' the background.
            
            # Simple Einstein Ring approximation for post-process:
            # s = b - alpha(b)
            # alpha(b) = LENS_STRENGTH / b
            
            # 1. Primary Mapping (Direct light)
            # The light is bent slightly. b_source is slightly smaller than b.
            b_src1 = b * (1.0 - (LENS_STRENGTH / (b * b + EPS)) * 0.4)
            
            # Sample Pass A at shifted radius
            sx1 = ti.cast(cx + (dx * b_src1 / b) * SCALE, ti.i32)
            sy1 = ti.cast(cy + (dy * b_src1 / b) * SCALE, ti.i32)
            
            if 0 <= sx1 < CANVAS_W and 0 <= sy1 < CANVAS_H:
                col += buffer_a[sx1, sy1]
                
            # 2. Secondary Mapping (Wraparound)
            # This is the back side of the disk appearing near the shadow.
            # Rays that loop around the hole.
            if b < shadow_rad * 3.5:
                # The secondary image maps b to a source position on the OPPOSITE side
                # OR, for the 'wraparound' look, we pull pixels from the disk plane.
                # Here we use a different warp to pull the 'top' of the disk 'down'
                # and the 'bottom' of the disk 'up' near the horizon.
                
                # Wraparound warp factor
                w = 0.5 * (shadow_rad / b)**2
                # We sample PASS A symmetrically to get top and bottom arcs
                # as requested by the prompt.
                
                # Offset angularly or vertically
                # Sampling the disk 'behind' the hole
                # If dy > 0 (looking above BH), sample PASS A further 'up' (where back disk is)
                # If dy < 0 (looking below BH), sample PASS A further 'down'
                
                # Let's try sampling from the opposite side of the disk plane
                # or just further out in Y.
                sy_arc_top = ti.cast(cy + (dy + w * 5.0) * SCALE, ti.i32)
                sx_arc     = ti.cast(cx + dx * SCALE, ti.i32)
                
                if 0 <= sx_arc < CANVAS_W and 0 <= sy_arc_top < CANVAS_H:
                    col += buffer_a[sx_arc, sy_arc_top] * w * 1.5
                
                sy_arc_bot = ti.cast(cy + (dy - w * 5.0) * SCALE, ti.i32)
                if 0 <= sx_arc < CANVAS_W and 0 <= sy_arc_bot < CANVAS_H:
                    col += buffer_a[sx_arc, sy_arc_bot] * w * 1.5

            # 3. Photon Sphere Ring
            # Bright ring at b = shadow_rad
            gap = b - shadow_rad
            if gap < 0.2:
                ring_bri = ti.exp(-gap * 15.0) * 0.8
                col += ti.Vector([1.0, 0.8, 0.4]) * ring_bri

        # Tonemapping / Bloom approximation
        canvas_img[i, j] = ti.tanh(col)

# -----------------------------------------------------------------------------
# 11. INTERACTION
# -----------------------------------------------------------------------------
@ti.kernel
def inject_gas():
    gm = GM * bh_mass_mul[None]
    TPI = 2.0 * math.pi
    ip = inject_coord[0]
    # Find closest disk radius
    r_target = ip.norm()
    for _ in range(500):
        idx = ti.cast(ti.random() * N_DISK, ti.i32) % N_DISK
        ang = ti.random() * TPI
        r = r_target + (ti.random()-0.5) * 1.0
        r = ti.max(r, R_DISK_IN)
        vc = get_v_circ(r, gm)
        pos_d[idx] = ti.Vector([r * ti.cos(ang), r * ti.sin(ang), (ti.random()-0.5)*0.1])
        vel_d[idx] = ti.Vector([-vc * ti.sin(ang), vc * ti.cos(ang), 0.0])
        life_d[idx] = 10.0

# -----------------------------------------------------------------------------
# 12. MAIN
# -----------------------------------------------------------------------------
def main():
    print("="*60)
    print(" BLACK HOLE SIMULATION V3 - GARGANTUA STYLE")
    print("="*60)
    print(" Controls:")
    print("  SPACE      - Toggle Pause")
    print("  + / -      - Time Scaling")
    print("  R          - Reset")
    print("  LMB Drag   - Inject Gas")
    print("="*60)

    upload_palette(build_palette_np())
    bh_mass_mul[None] = 1.0
    init_all()

    gui = ti.GUI("Gargantua Simulation V3", (CANVAS_W, CANVAS_H), fast_gui=True)
    
    paused = False
    ts = 1.0
    
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.SPACE: paused = not paused
            elif e.key in ['+', '=']: ts *= 1.5
            elif e.key == '-': ts /= 1.5
            elif e.key == 'r': init_all()

        if not paused:
            update_physics(0.01 * ts)
        
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            wx = (mx - 0.5) * CANVAS_W / SCALE
            wy = (my - 0.5) * CANVAS_H / SCALE
            inject_coord[0] = ti.Vector([wx, wy])
            inject_gas()

        render_pass_a()
        render_pass_b()
        
        gui.set_image(canvas_img)
        gui.show()

if __name__ == "__main__":
    main()
