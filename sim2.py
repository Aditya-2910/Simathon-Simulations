"""
sim2.py -- Schwarzschild Black Hole: Version 2
  ● 3-D geometry rendered at ~20° above the disk plane (Interstellar / Gargantua view)
  ● Gravitational lensing arc wrapping above & below the silhouette (the key V2 fix)
  ● Thin, coherent accretion disk with orbital motion-blur trails
  ● Clearly asymmetric Doppler beaming
  ● Gravitational redshift colour / dimming near r_s
  ● Faint background star lensing (Einstein-arc smearing near shadow edge)
  ● Twin relativistic helical jets (bright cyan / magenta)
  ● One spacecraft / pod on a stable orbit near the photon sphere

Run:
    uv run --python 3.12 --with taichi sim2.py

CONTROLS  (printed at startup):
  SPACE      -- Pause / Resume
  + / =      -- Speed up time  (×2, max 8×)
  -          -- Slow down time  (÷2, min 0.125×)
  R          -- Reset
  L          -- Toggle background stars
  LMB        -- Inject fresh gas clump at cursor radius
  RMB drag   -- Drag up/down to change BH mass (0.3× – 3×)

═══════════════════════════════════════════════════════════════════
UNIT SYSTEM
  G·M_bh  = GM = 1.0       (gravitational parameter)
  c       = 1.0            (speed of light)
  → r_s   = 2 GM / c²  = 2.0  (world units)
  → beta at ISCO = v_circ/c = sqrt(GM/r_isco)/c ≈ 0.408

WORLD → PIXEL MAPPING
  SCALE  = 28 px / r_s    (event horizon ≈ 56 px wide)
  Disk outer radius  → 38 r_s  ≈ 2128 wu-px
  Canvas 1280 × 800
═══════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
import taichi as ti

# ──────────────────────────────────────────────────────────────────────────────
# 1.  TAICHI INIT
# ──────────────────────────────────────────────────────────────────────────────
try:
    ti.init(arch=ti.gpu)
    print("[Taichi] GPU backend initialised.")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] CPU fallback. Lower N_DISK if FPS < 30.")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  PHYSICAL CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
GM       = 1.0           # G × BH mass  (1.0 world units)
C_LIGHT  = 1.0           # speed of light (1.0 world unit / time unit)

# GR characteristic radii
R_S    = 2.0 * GM / (C_LIGHT ** 2)   # Schwarzschild radius   r_s = 2.0
R_PH   = 1.5 * R_S                    # photon sphere          r_ph = 3.0
R_ISCO = 3.0 * R_S                    # ISCO                   r_isco = 6.0

# Pixel mapping
CANVAS_W, CANVAS_H = 1280, 800
SCALE = 28.0 / R_S          # pixels per world unit  (=14.0 px/wu)
# In pixels:
R_S_PX    = R_S    * SCALE   # 28 px  – event horizon radius
R_PH_PX   = R_PH   * SCALE   # 42 px  – photon sphere
R_ISCO_PX = R_ISCO * SCALE   # 84 px  – ISCO

# Disk extent
R_DISK_IN  = R_ISCO                  # inner edge at ISCO
R_DISK_OUT = 38.0 * R_S              # outer edge

# ── Camera / view parameters ─────────────────────────────────────────────────
# We view the disk from a point elevated θ_cam above the disk plane.
# Rotation is around the X-axis: y' = y·cos θ – z·sin θ  (screen-y)
#                                 x stays as screen-x
# θ_cam = 20° above edge-on  →  cos/sin precomputed here.
THETA_CAM  = math.radians(20.0)   # camera tilt above disk plane
CAM_COS    = math.cos(THETA_CAM)  # ≈ 0.940
CAM_SIN    = math.sin(THETA_CAM)  # ≈ 0.342

# In-kernel versions (ti.f32 constants used inside @ti.kernel)
_cam_cos = float(CAM_COS)
_cam_sin = float(CAM_SIN)

# Lensing arc parameters
# The lensing-arc radius in pixels is set to R_PH_PX so it hugs the silhouette.
ARC_INNER_PX = R_PH_PX * 1.05    # inner edge of the bright lensing arc band
ARC_OUTER_PX = R_PH_PX * 2.40    # outer extent of the glow

# ── Jet parameters ────────────────────────────────────────────────────────────
JET_V_Z     = 0.85 * C_LIGHT     # initial jet speed (fraction of c)
JET_R0      = 0.6  * R_S         # injection radius around jet axis
HELIX_OMEGA = 3.5                # magnetic twist rate (rad per unit vz-travel)
HELIX_FORCE = 0.04               # z-pinch collimation strength

# Screen-space projection of jet: z-axis → screen-Y via tilt.
# Vertical extent of jet on screen ≈ canvas half-height / sin(tilt + 70°)
# We keep it simple: jet z-travel projects as: screen_y += vz * CAM_COS * dt
# (the disk plane occupies the horizontal; jets rise perpendicular to it)
# The jet axis is the disk-normal → after tilt, it points mostly upward.
# screen dy contribution from vz: dy_screen = vz * cos(90° - theta_cam) = vz * sin(theta_cam)?
# Actually: jet axis is disk-normal = (0,0,1) in 3D.
# After our camera rotation (Rx by theta_cam):
#   screen_x = x (unchanged)
#   screen_y = y * cos(theta_cam) - z * sin(theta_cam)   ... but jet motion is in z
#   → for a jet particle at (x_disk, y_disk, z_jet):
#       screen_y = y_disk*cos - z_jet*sin
#   for y_disk ≈ 0 (thin disk), screen_y ≈ -z_jet * sin(theta_cam)
# Wait that means jets go DOWN when z>0.  Negate sin so upper jet goes up on screen.
#   screen_y = y_disk*cam_cos + z_jet*cam_sin     ← this is what we use
# This makes positive z → upward on screen which is intuitive.
JET_SCREEN_FACTOR = float(CAM_SIN)   # z → screen_y scale
DISK_SCREEN_FACTOR = float(CAM_COS)  # y → screen_y scale

EPS = 1e-6   # global epsilon to prevent division-by-zero

# ──────────────────────────────────────────────────────────────────────────────
# 3.  PARTICLE COUNTS
# ──────────────────────────────────────────────────────────────────────────────
N_DISK  = 200_000   # disk accretion particles
N_JET   =  16_000   # jet particles (both jets combined)
N_STARS =   4_000   # background stars
N_TRAIL =  12       # motion-blur trail points per disk particle

# ──────────────────────────────────────────────────────────────────────────────
# 4.  TAICHI FIELDS
# ──────────────────────────────────────────────────────────────────────────────
# Disk: 3-D positions  (x, y, z)  – disk is in the x-y plane, z≈0
pos_d  = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)   # (x, y, z) world
vel_d  = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)
col_d  = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)
life_d = ti.field(dtype=ti.f32, shape=N_DISK)

# Disk trail buffer: last N_TRAIL screen-space positions per particle
trail_x = ti.field(dtype=ti.f32, shape=(N_DISK, N_TRAIL))
trail_y = ti.field(dtype=ti.f32, shape=(N_DISK, N_TRAIL))

# Jet: 3-D
pos_j  = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
vel_j  = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
col_j  = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
life_j = ti.field(dtype=ti.f32, shape=N_JET)

# Background stars (2-D normalised coords in [0,1]²)
star_base   = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)
star_lensed = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)
star_col    = ti.Vector.field(3, dtype=ti.f32, shape=N_STARS)

# Pixel raster
canvas_img = ti.Vector.field(3, dtype=ti.f32, shape=(CANVAS_W, CANVAS_H))

# Scalars
bh_mass    = ti.field(dtype=ti.f32, shape=())
inject_buf = ti.Vector.field(2, dtype=ti.f32, shape=1)   # cursor in world x-y

# Spacecraft state  (x, y, z, vx, vy, vz)
ship_pos = ti.Vector.field(3, dtype=ti.f32, shape=1)
ship_vel = ti.Vector.field(3, dtype=ti.f32, shape=1)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  BLACKBODY PALETTE
# ──────────────────────────────────────────────────────────────────────────────
PALETTE_N = 512
palette   = ti.Vector.field(3, dtype=ti.f32, shape=PALETTE_N)

def build_palette_np():
    """Piecewise blackbody colour approximation.
    t=0 → cold deep-red (outer disk)
    t=1 → hot blue-white (ISCO inner edge)
    """
    pal = np.zeros((PALETTE_N, 3), dtype=np.float32)
    for i in range(PALETTE_N):
        t = i / (PALETTE_N - 1.0)
        if t < 0.15:
            u = t / 0.15
            r, g, b = 0.35 + 0.45*u, 0.02 + 0.05*u, 0.0
        elif t < 0.38:
            u = (t - 0.15) / 0.23
            r, g, b = 0.80 + 0.12*u, 0.07 + 0.28*u, 0.0 + 0.02*u
        elif t < 0.62:
            u = (t - 0.38) / 0.24
            r, g, b = 0.92 + 0.08*u, 0.35 + 0.50*u, 0.02 + 0.18*u
        elif t < 0.82:
            u = (t - 0.62) / 0.20
            r, g, b = 1.0, 0.85 + 0.12*u, 0.20 + 0.55*u
        else:
            u = (t - 0.82) / 0.18
            r, g, b = 1.0 - 0.15*u, 0.97 + 0.03*u, 0.75 + 0.25*u
        pal[i] = np.clip([r, g, b], 0.0, 1.0)
    return pal

@ti.kernel
def upload_palette(pal_np: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(PALETTE_N):
        palette[i] = ti.Vector([pal_np[i, 0], pal_np[i, 1], pal_np[i, 2]])

# ──────────────────────────────────────────────────────────────────────────────
# 6.  INLINE PHYSICS HELPERS   (@ti.func – inlined, no call overhead)
# ──────────────────────────────────────────────────────────────────────────────

@ti.func
def sample_palette(t: ti.f32):
    tc  = ti.max(0.0, ti.min(1.0, t))
    idx = ti.cast(tc * (PALETTE_N - 1), ti.i32)
    return palette[idx]

@ti.func
def orbital_speed(r: ti.f32, gm: ti.f32) -> ti.f32:
    """Keplerian: v_circ = sqrt(GM / r)."""
    return ti.sqrt(gm / (r + EPS))

@ti.func
def shakura_temp(r: ti.f32) -> ti.f32:
    """T(r) ∝ r^{-3/4}.  Normalised to 1 at ISCO."""
    return ti.min(1.0, (R_ISCO / (r + EPS)) ** 0.75)

@ti.func
def doppler_boost(vx: ti.f32, vy: ti.f32) -> ti.f32:
    """
    Relativistic Doppler beaming.
    Observer is at +x direction (after projection the right side of disk
    approaches, left recedes from camera's perspective).
    δ = 1 / (γ (1 - β·cos θ))     intensity ∝ δ⁴.
    """
    v2    = vx*vx + vy*vy + EPS
    v_mag = ti.sqrt(v2)
    beta  = ti.min(v_mag / C_LIGHT, 0.998)
    gamma = 1.0 / ti.sqrt(1.0 - beta*beta + EPS)
    # line-of-sight direction is +x (screen right = approaching side)
    cos_t = vx / (v_mag + EPS)
    delta = 1.0 / (gamma * (1.0 - beta * cos_t) + EPS)
    return ti.pow(delta, 4.0)   # δ⁴ makes asymmetry obvious at β≈0.4

@ti.func
def grav_redshift(r: ti.f32) -> ti.f32:
    """sqrt(1 - r_s/r),  → 0 at horizon, → 1 far away."""
    return ti.sqrt(ti.max(0.0, 1.0 - R_S / (r + EPS)))

@ti.func
def project_to_screen(x3: ti.f32, y3: ti.f32, z3: ti.f32):
    """
    Camera projection.
    World: x=right, y=away-from-viewer (horizontal disk direction), z=up.
    Camera elevated θ_cam above the disk plane → Rx rotation:
        screen_x =  x3
        screen_y =  y3 * cos(θ) + z3 * sin(θ)
    Returns (screen_x, screen_y) in world units (still needs × SCALE + centre).
    """
    sx = x3
    sy = y3 * _cam_cos + z3 * _cam_sin
    return sx, sy

# ──────────────────────────────────────────────────────────────────────────────
# 7.  INITIALISATION KERNELS
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def init_disk():
    gm  = GM * bh_mass[None]
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_DISK):
        # Bias sampling toward inner bright ISCO region
        u     = ti.random()
        r_frac = u * u
        r     = R_DISK_IN + r_frac * (R_DISK_OUT - R_DISK_IN)

        angle  = ti.random() * TPI
        v_circ = orbital_speed(r, gm)
        dv     = (ti.random() - 0.5) * 0.05 * v_circ

        # 3-D position: disk in x-y plane, z confined to thin slab
        disc_h = R_S * 0.04  # half-thickness = 4% of r_s (very thin)
        z_disk = (ti.random() - 0.5) * 2.0 * disc_h

        pos_d[i] = ti.Vector([r * ti.cos(angle),
                               r * ti.sin(angle),
                               z_disk])
        # Velocity: tangential in x-y plane
        vel_d[i] = ti.Vector([-(v_circ + dv) * ti.sin(angle),
                                (v_circ + dv) * ti.cos(angle),
                                0.0])
        life_d[i] = ti.random() * 10000.0
        col_d[i]  = ti.Vector([1.0, 0.5, 0.1])

        # Init trail to current position
        sx, sy = project_to_screen(pos_d[i][0], pos_d[i][1], pos_d[i][2])
        px_f = float(CANVAS_W) * 0.5 + sx * SCALE
        py_f = float(CANVAS_H) * 0.5 + sy * SCALE
        for t in range(N_TRAIL):
            trail_x[i, t] = px_f
            trail_y[i, t] = py_f


@ti.kernel
def init_jets():
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_JET):
        side = 1.0 if i < N_JET // 2 else -1.0   # +z or -z

        ca = ti.random() * TPI
        cr = ti.random() * JET_R0 * 0.4
        pos_j[i] = ti.Vector([cr * ti.cos(ca), cr * ti.sin(ca), 0.0])
        vel_j[i] = ti.Vector([(ti.random()-0.5)*JET_V_Z*0.03,
                               (ti.random()-0.5)*JET_V_Z*0.03,
                                side * JET_V_Z * (0.80 + ti.random()*0.20)])
        life_j[i] = ti.random() * 400.0

        if side > 0.0:
            col_j[i] = ti.Vector([0.10, 0.55, 1.0])   # electric cyan
        else:
            col_j[i] = ti.Vector([1.0, 0.20, 0.85])   # hot magenta


@ti.kernel
def init_stars():
    for i in range(N_STARS):
        bx = ti.random()
        by = ti.random()
        star_base[i]   = ti.Vector([bx, by])
        star_lensed[i] = ti.Vector([bx, by])
        rr = ti.random()
        if rr < 0.30:
            star_col[i] = ti.Vector([0.60, 0.72, 1.0])   # blue-white
        elif rr < 0.60:
            star_col[i] = ti.Vector([1.0,  1.0,  0.95])  # white
        elif rr < 0.80:
            star_col[i] = ti.Vector([1.0,  0.88, 0.40])  # yellow
        else:
            star_col[i] = ti.Vector([1.0,  0.55, 0.22])  # orange


@ti.kernel
def init_ship():
    """Place the spacecraft just outside the photon sphere."""
    gm    = GM * bh_mass[None]
    r_orb = R_PH * 1.6     # orbit between photon sphere and ISCO (~outer edge inner disk)
    angle = 0.75            # starting angle (world units, radians)
    v_circ = orbital_speed(r_orb, gm)
    ship_pos[0] = ti.Vector([r_orb * ti.cos(angle),
                               r_orb * ti.sin(angle),
                               0.0])
    ship_vel[0] = ti.Vector([-v_circ * ti.sin(angle),
                               v_circ * ti.cos(angle),
                               0.0])

# ──────────────────────────────────────────────────────────────────────────────
# 8.  PHYSICS UPDATE KERNELS
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def update_disk(dt: ti.f32, trail_idx: ti.i32):
    """
    Advance disk particles.  3-D Newtonian + 1PN GR + semi-implicit Euler.
    Per-particle colour: Shakura-Sunyaev × Doppler × gravitational-redshift.
    Trail buffer updated every call.
    """
    gm  = GM * bh_mass[None]
    TPI = 2.0 * ti.acos(-1.0)
    cx  = float(CANVAS_W) * 0.5
    cy  = float(CANVAS_H) * 0.5

    for i in range(N_DISK):
        p = pos_d[i]
        v = vel_d[i]

        # Radial distance in disk plane
        r = ti.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]) + EPS

        # Newtonian gravity
        a = -(gm / (r*r*r)) * p

        # 1PN GR angular-momentum correction (perihelion precession)
        Lz = p[0]*v[1] - p[1]*v[0]   # specific angular momentum z-component
        pn_fac = 3.0 * gm * Lz * Lz / (C_LIGHT*C_LIGHT * (r**4) + EPS)
        a = a - (pn_fac / (r + EPS)) * p

        # Restoring force to keep disk thin (harmonic oscillator in z)
        # ω_z ≈ Ω_orbital  (same order)
        omega_z = orbital_speed(r, gm) / (r + EPS)
        a[2] = a[2] - omega_z * omega_z * p[2]

        # Semi-implicit Euler
        v = v + a * dt
        p = p + v * dt

        # ── Re-spawn: swallowed by horizon ─────────────────────────────────
        if r < R_S * 1.05:
            ang   = ti.random() * TPI
            r_new = R_DISK_OUT * (0.50 + ti.random() * 0.50)
            vc    = orbital_speed(r_new, gm)
            dv    = (ti.random()-0.5)*0.05*vc
            p = ti.Vector([r_new*ti.cos(ang), r_new*ti.sin(ang), 0.0])
            v = ti.Vector([-(vc+dv)*ti.sin(ang), (vc+dv)*ti.cos(ang), 0.0])
            life_d[i] = 0.0

        # ── Re-spawn: escaped outer boundary ───────────────────────────────
        if ti.sqrt(p[0]*p[0]+p[1]*p[1]) > R_DISK_OUT * 1.35:
            ang   = ti.random() * TPI
            r_new = R_DISK_IN + ti.random()*(R_DISK_OUT - R_DISK_IN)*0.50
            vc    = orbital_speed(r_new, gm)
            dv    = (ti.random()-0.5)*0.05*vc
            p = ti.Vector([r_new*ti.cos(ang), r_new*ti.sin(ang), 0.0])
            v = ti.Vector([-(vc+dv)*ti.sin(ang), (vc+dv)*ti.cos(ang), 0.0])
            life_d[i] = 0.0

        pos_d[i]   = p
        vel_d[i]   = v
        life_d[i] += dt

        # ── Colour ─────────────────────────────────────────────────────────
        r_cur = ti.sqrt(p[0]*p[0]+p[1]*p[1]) + EPS
        temp  = shakura_temp(r_cur)
        base  = sample_palette(temp)

        # Doppler beaming uses screen-x velocity (vx unchanged by y-tilt)
        beam = ti.max(0.20, ti.min(doppler_boost(v[0], v[1]), 8.0))

        gz   = grav_redshift(r_cur)
        bri  = beam * gz * ti.min(1.5, 1.8*temp + 0.10)

        rc = ti.min(1.0, base[0] * bri)
        gc = ti.min(1.0, base[1] * bri * gz)
        bc = ti.min(1.0, base[2] * bri * gz * gz)
        col_d[i] = ti.Vector([rc, gc, bc])

        # ── Trail: shift buffer and record current screen position ──────────
        sx, sy = project_to_screen(p[0], p[1], p[2])
        px_f = cx + sx * SCALE
        py_f = cy + sy * SCALE
        # Shift old trail entries back one slot
        for t in ti.static(range(N_TRAIL - 1, 0, -1)):
            trail_x[i, t] = trail_x[i, t-1]
            trail_y[i, t] = trail_y[i, t-1]
        trail_x[i, 0] = px_f
        trail_y[i, 0] = py_f


@ti.kernel
def update_jets(dt: ti.f32, sim_time: ti.f32):
    """
    Advance jet particles with helical magnetic twist and z-pinch collimation.
    3-D positions, projected to screen by view kernel.
    """
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_JET):
        p  = pos_j[i]
        v  = vel_j[i]
        vz = v[2]
        side = 1.0 if vz >= 0.0 else -1.0

        # In-plane radius
        rxy = ti.sqrt(p[0]*p[0] + p[1]*p[1]) + EPS

        # Helical twist: rotate in-plane velocity by tiny angle each step
        phi   = HELIX_OMEGA / (ti.abs(vz) + EPS) * ti.abs(vz) * dt * 0.0025
        cos_p = ti.cos(phi)
        sin_p = ti.sin(phi)
        vx2 = v[0]*cos_p - v[1]*sin_p
        vy2 = v[0]*sin_p + v[1]*cos_p
        v[0] = vx2
        v[1] = vy2

        # z-pinch: collimates jet toward axis (x=0, y=0)
        pinch_x = -HELIX_FORCE * p[0] / (rxy + EPS) * ti.abs(vz)
        pinch_y = -HELIX_FORCE * p[1] / (rxy + EPS) * ti.abs(vz)
        v[0] += pinch_x * dt
        v[1] += pinch_y * dt

        # Semi-implicit Euler
        p = p + v * dt

        # Re-spawn when too far
        max_z = R_DISK_OUT * 3.5
        if ti.abs(p[2]) > max_z or rxy > R_DISK_IN * 2.5:
            ca  = ti.random() * TPI
            cr  = ti.random() * JET_R0 * 0.4
            p   = ti.Vector([cr*ti.cos(ca), cr*ti.sin(ca), 0.0])
            v   = ti.Vector([(ti.random()-0.5)*JET_V_Z*0.03,
                              (ti.random()-0.5)*JET_V_Z*0.03,
                               side * JET_V_Z * (0.80 + ti.random()*0.20)])
            life_j[i] = 0.0

        pos_j[i]   = p
        vel_j[i]   = v
        life_j[i] += dt

        # ── Jet colour: fade with distance, pulsing knots ──────────────────
        z_frac   = ti.abs(p[2]) / (R_DISK_OUT * 3.5 + EPS)
        fade     = ti.max(0.0, 1.0 - z_frac * 1.10) ** 1.6
        # Knot pulsing: bright blobs travel along jet
        knot_freq = 14.0
        pulse = 0.50 + 0.50 * ti.abs(ti.sin(p[2]*knot_freq + sim_time * 6.0))
        bri   = fade * (0.55 + 0.45*pulse)
        if side > 0.0:
            col_j[i] = ti.Vector([0.06 + 0.40*pulse, 0.50*pulse, 1.0]) * bri
        else:
            col_j[i] = ti.Vector([1.0, 0.18*pulse, 0.85*pulse]) * bri


@ti.kernel
def update_ship(dt: ti.f32):
    """Advance the spacecraft on a Keplerian orbit, with time-dilation hint."""
    gm   = GM * bh_mass[None]
    p    = ship_pos[0]
    v    = ship_vel[0]
    r    = ti.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]) + EPS
    a    = -(gm / (r*r*r)) * p

    # Gravitational time dilation:  dt_proper ≈ sqrt(1 - r_s/r) × dt_coord
    # Slow the ship down slightly near horizon (purely visual hint)
    td   = ti.sqrt(ti.max(0.5, 1.0 - R_S / (r + EPS)))
    v = v + a * dt * td
    p = p + v * dt * td
    ship_pos[0] = p
    ship_vel[0] = v


@ti.kernel
def update_lensing():
    """
    Compute gravitationally lensed positions for background stars.
    Stars are pushed radially OUTWARD (image displacement away from BH).
    α = 4GM/(c²·b)  (Einstein deflection formula).
    """
    cx, cy = 0.5, 0.5
    WX2 = float(CANVAS_W) / SCALE   # world-x span (full)
    WY2 = float(CANVAS_H) / SCALE   # world-y span
    for i in range(N_STARS):
        bx = star_base[i][0]
        by = star_base[i][1]
        # Convert normalised [0,1] → world coords
        dx = (bx - cx) * WX2
        dy = (by - cy) * WY2

        b2    = dx*dx + dy*dy + EPS
        b     = ti.sqrt(b2)
        alpha = 4.0 * GM / (C_LIGHT * C_LIGHT * b + EPS)
        alpha = ti.min(alpha, 0.80 * b)   # cap to prevent inversion

        # Outward displacement in world units, converted back to normalised
        sx = alpha * dx / (b + EPS)
        sy = alpha * dy / (b + EPS)

        lx = bx + sx / WX2
        ly = by + sy / WY2
        star_lensed[i] = ti.Vector([lx, ly])


# ──────────────────────────────────────────────────────────────────────────────
# 9.  BACKGROUND RASTER  (BH shadow, corona, photon sphere, lensing arcs)
#     This kernel renders the structural scene elements to canvas_img.
#     Key new element: the lensing WRAPAROUND ARC above & below the horizon.
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def draw_background():
    """
    Per-pixel raster of static scene elements.

    Centre of BH is at (cx, cy_bh):
      cy_bh is shifted DOWN by the disk foreshortening —
      the disk we see is "floor-level" and the BH silhouette sits above.
      We keep BH at true screen centre here for simplicity.

    NEW:  Lensing wraparound arc.
      Because the disk is nearly edge-on, light from the back half of the disk
      is gravitationally bent OVER and UNDER the silhouette, creating a thin
      bright arc above and below the black disk (the one visible in real GR
      ray-traced renders like Gargantua).

      We model it as a bright glow band centred on  r = R_PH_PX × 1.3,
      restricted to the top and bottom (|cos θ| < 0.55) to represent the
      lensed back-disk light that wraps around rather than the front-disk
      (which we render as particles).

    All distances in pixels, origin at (cx, cy).
    """
    cx = float(CANVAS_W) * 0.5
    cy = float(CANVAS_H) * 0.5

    for px, py in canvas_img:
        dx = float(px) - cx
        dy = float(py) - cy
        d  = ti.sqrt(dx*dx + dy*dy) + EPS

        # Angular position on screen (used for arc restriction)
        cos_phi = dx / (d + EPS)   # +1 = right, -1 = left
        sin_phi = dy / (d + EPS)   # +1 = top,   -1 = bottom

        # (i) Deep-space background: dark near-black with faint violet nebula
        edge   = d / float(CANVAS_W)
        nebula = ti.exp(-edge * edge * 6.0) * 0.012
        col = ti.Vector([0.004 + 0.014*edge*edge + nebula*0.5,
                         0.003 + 0.006*edge*edge + nebula*0.2,
                         0.011 + 0.028*edge*edge + nebula])

        # (ii) Event horizon shadow — solid black
        if d < R_S_PX * 0.98:
            col = ti.Vector([0.0, 0.0, 0.0])

        # (iii) Corona: hot glow emanating from just outside r_s
        corona_d = d - R_S_PX
        if 0.0 < corona_d < R_S_PX * 10.0:
            s = ti.exp(-corona_d*corona_d / (R_S_PX*R_S_PX * 7.0)) * 1.1
            col = col + ti.Vector([s*1.00, s*0.30, s*0.04])

        # (iv) Photon sphere ring — bright narrow ring at r_ph = 1.5 r_s
        ph_d = ti.abs(d - R_PH_PX)
        if ph_d < 4.0:
            s = ti.exp(-ph_d*ph_d / 3.5) * 1.8
            col = col + ti.Vector([s*1.2, s*0.78, s*0.18])

        # (v)  *** LENSING WRAPAROUND ARC ***
        #   Bright arc above and below the silhouette from lensed back-disk light.
        #   The arc spans radii R_PH_PX..ARC_OUTER_PX.
        #   It is brightest at top / bottom (arc wraps over the equator of the BH).
        #   Angular mask: |sin_phi| > 0.25  (exclude extreme equatorial sides
        #   where forward-facing disk particles already show) and
        #   cos_phi restricted to avoid the full equatorial region.
        #
        #   The arc brightness fades from the photon sphere outward and
        #   is modulated by |sin_phi|^1.5 — strongest directly above/below.
        arc_r = d - R_PH_PX
        if 0.0 < arc_r < ARC_OUTER_PX - R_PH_PX:
            r_frac = arc_r / (ARC_OUTER_PX - R_PH_PX)
            # Radial Gaussian centred just outside photon sphere
            s_rad = ti.exp(-r_frac * r_frac * 3.5) * ti.exp(-arc_r * 0.018)
            # Angular mask: strong at top/bottom, weak at sides
            ang_mask = ti.pow(ti.abs(sin_phi), 1.2)
            # Suppress the near-side (forward disk) region: only arc at back
            # In a tilted view, left side (cos_phi < 0) is mainly back-disk.
            # For a cinematic look we let it be visible all around but
            # strongest at top/bottom.
            arc_bri = s_rad * ang_mask * 1.60
            # Colour: hot orange-white blending into bluish at extreme angles
            col = col + ti.Vector([ti.min(1.0, arc_bri * 1.0),
                                   ti.min(1.0, arc_bri * 0.70),
                                   ti.min(1.0, arc_bri * 0.28)])

        # (vi) ISCO ring — soft cyan boundary marker at r = 3 r_s
        isco_d = ti.abs(d - R_ISCO_PX)
        if isco_d < 2.5:
            s = ti.exp(-isco_d*isco_d / 1.5) * 0.35
            col = col + ti.Vector([s*0.02, s*0.65, s*1.0])

        # Clamp and write
        canvas_img[px, py] = ti.Vector([ti.min(col[0], 1.0),
                                         ti.min(col[1], 1.0),
                                         ti.min(col[2], 1.0)])


# ──────────────────────────────────────────────────────────────────────────────
# 10.  GAS INJECTION AT CURSOR
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def inject_gas():
    """
    Overwrite N_INJECT random disk particles centred on cursor world-pos.
    Cursor position (in world x-y units) stored in inject_buf[0].
    """
    N_INJECT = 800
    TPI = 2.0 * ti.acos(-1.0)
    gm  = GM * bh_mass[None]
    for _ in range(N_INJECT):
        ip      = inject_buf[0]
        r_ref   = ti.sqrt(ip[0]*ip[0] + ip[1]*ip[1]) + EPS
        angle   = ti.atan2(ip[1], ip[0])
        r_new   = ti.max(R_DISK_IN * 1.1,
                         r_ref + (ti.random()-0.5) * R_DISK_IN * 0.70)
        ang_new = angle + (ti.random()-0.5) * 0.50
        vc      = orbital_speed(r_new, gm)
        dv      = (ti.random()-0.5) * 0.08 * vc
        idx     = ti.cast(ti.random() * N_DISK, ti.i32) % N_DISK
        pos_d[idx] = ti.Vector([r_new*ti.cos(ang_new),
                                  r_new*ti.sin(ang_new),
                                  0.0])
        vel_d[idx] = ti.Vector([-(vc+dv)*ti.sin(ang_new),
                                   (vc+dv)*ti.cos(ang_new),
                                   0.0])
        life_d[idx] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 11.  PARTICLE RENDERING  (additive splat into canvas_img)
# ──────────────────────────────────────────────────────────────────────────────

@ti.kernel
def render_particles(show_stars: ti.i32):
    """
    Project every particle to screen, then additively splat a glowing dot.
    Disk particles also render motion-blur trails (progressively dimmer).
    Jets render with a brighter splat radius.
    Stars are in normalised [0,1] coords.
    """
    cx = float(CANVAS_W) * 0.5
    cy = float(CANVAS_H) * 0.5

    # ── Disk particles + trails ─────────────────────────────────────────────
    for i in range(N_DISK):
        p   = pos_d[i]
        rgb = col_d[i]

        # Current position projected
        sx, sy = project_to_screen(p[0], p[1], p[2])
        px_cur = cx + sx * SCALE
        py_cur = cy + sy * SCALE

        # Splat head particle (brightest)
        for dx_px in range(-1, 2):
            for dy_px in range(-1, 2):
                ix = ti.cast(px_cur, ti.i32) + dx_px
                iy = ti.cast(py_cur, ti.i32) + dy_px
                if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                    w = ti.exp(-float(dx_px*dx_px + dy_px*dy_px) * 0.55) * 0.48
                    canvas_img[ix, iy] = ti.Vector([
                        ti.min(canvas_img[ix, iy][0] + rgb[0]*w, 1.0),
                        ti.min(canvas_img[ix, iy][1] + rgb[1]*w, 1.0),
                        ti.min(canvas_img[ix, iy][2] + rgb[2]*w, 1.0)])

        # Motion-blur trail (older positions dimmer)
        for t in range(1, N_TRAIL):
            tx = trail_x[i, t]
            ty = trail_y[i, t]
            if tx > 0.0 and ty > 0.0:
                trail_weight = ti.pow(1.0 - float(t) / float(N_TRAIL), 2.0) * 0.20
                ix = ti.cast(tx, ti.i32)
                iy = ti.cast(ty, ti.i32)
                if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                    canvas_img[ix, iy] = ti.Vector([
                        ti.min(canvas_img[ix, iy][0] + rgb[0]*trail_weight, 1.0),
                        ti.min(canvas_img[ix, iy][1] + rgb[1]*trail_weight, 1.0),
                        ti.min(canvas_img[ix, iy][2] + rgb[2]*trail_weight, 1.0)])

    # ── Jet particles ───────────────────────────────────────────────────────
    for i in range(N_JET):
        p   = pos_j[i]
        rgb = col_j[i]
        sx, sy = project_to_screen(p[0], p[1], p[2])
        px_f = cx + sx * SCALE
        py_f = cy + sy * SCALE
        for dx_px in range(-2, 3):
            for dy_px in range(-2, 3):
                ix = ti.cast(px_f, ti.i32) + dx_px
                iy = ti.cast(py_f, ti.i32) + dy_px
                if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                    w = ti.exp(-float(dx_px*dx_px + dy_px*dy_px) * 0.35) * 0.62
                    canvas_img[ix, iy] = ti.Vector([
                        ti.min(canvas_img[ix, iy][0] + rgb[0]*w, 1.0),
                        ti.min(canvas_img[ix, iy][1] + rgb[1]*w, 1.0),
                        ti.min(canvas_img[ix, iy][2] + rgb[2]*w, 1.0)])

    # ── Background stars ────────────────────────────────────────────────────
    if show_stars == 1:
        for i in range(N_STARS):
            sp  = star_lensed[i]
            rgb = star_col[i]
            px_f = sp[0] * float(CANVAS_W)
            py_f = sp[1] * float(CANVAS_H)
            ix = ti.cast(px_f, ti.i32)
            iy = ti.cast(py_f, ti.i32)
            if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                w = 0.28
                canvas_img[ix, iy] = ti.Vector([
                    ti.min(canvas_img[ix, iy][0] + rgb[0]*w, 1.0),
                    ti.min(canvas_img[ix, iy][1] + rgb[1]*w, 1.0),
                    ti.min(canvas_img[ix, iy][2] + rgb[2]*w, 1.0)])


# ──────────────────────────────────────────────────────────────────────────────
# 12.  SPACECRAFT RENDERING  (CPU-side, drawn via gui.lines / gui.circles)
# ──────────────────────────────────────────────────────────────────────────────

def draw_spacecraft(gui, cx_n, cy_n):
    """
    Render the spacecraft as a compact capsule + engine glow using gui primitives.
    cx_n, cy_n: screen-space normalised centre [0,1].

    Silhouette: a tiny elongated hull (2 line segments forming a capsule shape) +
    side nacelles + a bright engine glow dot at the rear.
    """
    s = 0.012   # half-length scale (normalised coords)
    # Forward direction: tangent to orbit = perpendicular to position vector
    p3 = ship_pos[0].to_numpy()
    v3 = ship_vel[0].to_numpy()
    # Project position direction to screen space
    sx, sy = p3[0], p3[1]*CAM_COS + p3[2]*CAM_SIN
    vx, vy = v3[0], v3[1]*CAM_COS + v3[2]*CAM_SIN
    v_norm = math.sqrt(vx**2 + vy**2) + 1e-9
    tx, ty = vx/v_norm, vy/v_norm   # tangent (forward)
    nx, ny = -ty, tx                  # normal (perpendicular)

    # Hull centre in normalised screen coords
    hx = cx_n + sx * SCALE / CANVAS_W
    hy = cy_n + sy * SCALE / CANVAS_H

    def nd(dx, dy):
        return (hx + dx * s / CANVAS_W * CANVAS_W,
                hy + dy * s / CANVAS_H * CANVAS_H)

    # Convert to normalised [0,1] with correct aspect
    asp = CANVAS_H / CANVAS_W
    s_x = s
    s_y = s * asp

    def pt(dtx, dty, dnx, dny):
        """Point: d along tangent × s_x, d along normal × s_y."""
        return (hx + dtx * s_x * tx - dty * s_y * ny,
                hy + dtx * s_x * ty - dty * s_y * nx)

    # Hull body (elongated rhombus)
    hull_color = 0xCCDDFF
    gui.line(pt(1.0, 0, 0, 0), pt(-0.8, 0, 0, 0), radius=2.0, color=hull_color)
    # Nacelles
    gui.line(pt(0.2, 0, 0, 0), pt(0.1, 0.0, 0.5, 0.35), radius=1.5, color=hull_color)
    gui.line(pt(0.2, 0, 0, 0), pt(0.1, 0.0, -0.5, -0.35), radius=1.5, color=hull_color)

    # Engine glow (rear)
    ep = pt(-0.85, 0, 0, 0)
    gui.circle(ep, color=0x00CCFF, radius=3)
    gui.circle(ep, color=0xFFFFFF, radius=1)

    # Running lights (faint white dots on hull)
    gui.circle(pt(0.8, 0, 0, 0),   color=0xFFFF88, radius=2)
    gui.circle(pt(0.0, 0, 0.5, 0.35), color=0xFF4444, radius=2)
    gui.circle(pt(0.0, 0, -0.5, -0.35), color=0xFF4444, radius=2)


# ──────────────────────────────────────────────────────────────────────────────
# 13.  MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SCHWARZSCHILD BLACK HOLE  -  Version 2  (Taichi real-time GR sim)")
    print("=" * 70)
    print(f"  Camera tilt    = {math.degrees(THETA_CAM):.0f} deg above disk plane  (Gargantua-style)")
    print(f"  r_s  (horizon) = {R_S_PX:.1f} px   [{R_S:.3f} wu]")
    print(f"  r_ph (photon)  = {R_PH_PX:.1f} px   [{R_PH:.3f} wu]")
    print(f"  r_isco         = {R_ISCO_PX:.1f} px   [{R_ISCO:.3f} wu]")
    print(f"  beta at ISCO   = {math.sqrt(GM/R_ISCO)/C_LIGHT:.3f}  (Doppler beaming exponent 4)")
    print(f"  Particles:      disk={N_DISK:,}  jets={N_JET:,}  stars={N_STARS:,}")
    print()
    print("  CONTROLS:")
    print("    SPACE        - Pause / Resume")
    print("    + / =        - Speed up (x2, max 8x)")
    print("    -            - Slow down (div2, min 0.125x)")
    print("    R            - Reset all")
    print("    L            - Toggle background stars")
    print("    LMB click    - Inject gas clump at cursor")
    print("    RMB drag     - Drag up/down to change BH mass (0.3x - 3x)")
    print()
    print("  Physics:")
    print("    [OK] 3-D Keplerian + 1PN GR perihelion advance")
    print("    [OK] Shakura-Sunyaev T(r) ~ r^{-3/4} palette")
    print("    [OK] Relativistic Doppler beaming delta^4  (obvious asymmetry)")
    print("    [OK] Gravitational redshift sqrt(1-r_s/r)  (dims/reddens at horizon)")
    print("    [OK] Lensing wraparound arc (bright arcs above/below shadow)")
    print("    [OK] Background star gravitational lensing alpha=4GM/c^2/b")
    print("    [OK] Helical magnetic-pinch relativistic jets")
    print("    [OK] Spacecraft near photon sphere with time-dilation orbit")
    print("=" * 70)

    # --- Initialise ----------------------------------------------------------
    pal_np = build_palette_np()
    upload_palette(pal_np)
    bh_mass[None] = 1.0
    init_disk()
    init_jets()
    init_stars()
    init_ship()
    update_lensing()

    # Timestep: 400 steps per ISCO orbital period
    T_isco  = 2.0 * math.pi * R_ISCO / math.sqrt(GM / R_ISCO)
    dt_base = T_isco / 400.0
    print(f"\n  ISCO period = {T_isco:.4e} wu-t,  dt_base = {dt_base:.4e} wu-t")

    # --- GUI window ----------------------------------------------------------
    gui = ti.GUI(
        "Schwarzschild Black Hole V2",
        res=(CANVAS_W, CANVAS_H),
        fast_gui=True,
        background_color=0x000000,
    )

    paused      = False
    time_scale  = 1.0
    sim_time    = 0.0
    show_stars  = True
    frame_cnt   = 0
    drag_prev_y = 0.5
    right_held  = False
    trail_idx   = 0   # which trail slot to write this frame (cycled)

    while gui.running:
        # -- Events -----------------------------------------------------------
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.SPACE:
                paused = not paused
                print(f"[sim] {'Paused' if paused else 'Running'}")
            elif e.key in ('=', '+'):
                time_scale = min(time_scale * 2.0, 8.0)
                print(f"[sim] Time scale: {time_scale:.3f}x")
            elif e.key == '-':
                time_scale = max(time_scale / 2.0, 0.125)
                print(f"[sim] Time scale: {time_scale:.3f}x")
            elif e.key in ('r', 'R'):
                bh_mass[None] = 1.0
                init_disk(); init_jets(); init_stars(); init_ship()
                update_lensing()
                sim_time = 0.0; time_scale = 1.0; paused = False
                print("[sim] Reset!")
            elif e.key in ('l', 'L'):
                show_stars = not show_stars
                print(f"[sim] Stars: {'ON' if show_stars else 'OFF'}")

        # -- Left click: inject gas -------------------------------------------
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            # Convert screen [0,1] → world x-y  (ignoring view tilt for click)
            wx = (mx - 0.5) * CANVAS_W / SCALE
            wy = (my - 0.5) * CANVAS_H / SCALE
            r_click = math.sqrt(wx**2 + wy**2)
            if r_click > R_DISK_IN * 1.05:
                inject_buf[0] = [wx, wy]
                inject_gas()

        # -- Right click: nudge BH mass ---------------------------------------
        rmb = gui.is_pressed(ti.GUI.RMB)
        if rmb:
            mx, my = gui.get_cursor_pos()
            if right_held:
                dy_drag = my - drag_prev_y
                bh_mass[None] = max(0.3, min(3.0, bh_mass[None] + dy_drag * 4.0))
            drag_prev_y = my
            right_held  = True
        else:
            right_held = False

        # -- Physics ----------------------------------------------------------
        if not paused:
            dt = dt_base * time_scale
            sim_time += dt
            update_disk(dt, trail_idx)
            update_jets(dt, sim_time)
            update_ship(dt)
            trail_idx = (trail_idx + 1) % N_TRAIL
            if frame_cnt % 60 == 0 and show_stars:
                update_lensing()

        frame_cnt += 1

        # -- Render -----------------------------------------------------------
        draw_background()
        render_particles(1 if show_stars else 0)

        # Draw spacecraft via GUI primitives (on top of raster)
        sp3 = ship_pos[0].to_numpy()
        sv3 = ship_vel[0].to_numpy()
        sx_s = sp3[0] * SCALE / CANVAS_W + 0.5
        sy_s = (sp3[1]*CAM_COS + sp3[2]*CAM_SIN) * SCALE / CANVAS_H + 0.5
        draw_spacecraft(gui, sx_s, sy_s)

        gui.set_image(canvas_img)

        # HUD
        gui.text(f"Time: {time_scale:.2f}x   BH mass: {bh_mass[None]:.2f}x M0",
                 pos=(0.01, 0.98), color=0xBBCCDD, font_size=18)
        gui.text(f"{'[PAUSED]  ' if paused else ''}Stars: {'ON' if show_stars else 'OFF'}   "
                 f"cam: {math.degrees(THETA_CAM):.0f} deg above disk",
                 pos=(0.01, 0.95), color=0xBBCCDD, font_size=17)
        gui.text("SPACE=Pause  +/-=Speed  R=Reset  L=Stars  LMB=inject gas  RMB=BH mass",
                 pos=(0.01, 0.015), color=0x778899, font_size=15)

        gui.show()

    print("[sim] Window closed.")


if __name__ == "__main__":
    main()
