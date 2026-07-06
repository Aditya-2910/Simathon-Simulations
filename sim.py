"""
sim.py -- Schwarzschild Black Hole: Accretion Disk, Gravitational Lensing & Relativistic Jets
A cinematic, real-time physics simulation using Taichi.

Run with:
    $env:Path = "C:\\Users\\asadi\\.local\\bin;$env:Path"
    uv run --python 3.12 --with taichi sim.py

UNIT SYSTEM:
  We choose "natural GR-ish units" where:
      G * M_bh  = GM = 1.0    (gravitational parameter)
      c              = 300.0  (speed of light — large so v/c ~ 0.3..0.6 near ISCO)

  Schwarzschild formula:  r_s = 2 G M / c^2
      r_s = 2 * 1.0 / 300^2 = 2.22e-5  (world units)

  WORLD_SCALE = 22 px / r_s  ->  event horizon is 22 px wide.
  Disk extends to 38 r_s ~ 836 px.

  At ISCO:  v_circ = sqrt(GM / r_isco) = 122 wu/t
            beta = v_circ / c = 0.41   -> meaningful Doppler beaming

CONTROLS (printed at startup):
  SPACE        -- Pause / Resume
  + / =        -- Speed up time (x2 per press, max 8x)
  -            -- Slow time (divide by 2, min 0.125x)
  R            -- Reset everything
  L            -- Toggle lensed background stars
  Left Click   -- Inject fresh gas clump at cursor
  Right Click  -- Hold + drag up/down to nudge BH mass (0.3x -- 3x)
"""

import math
import numpy as np
import taichi as ti

# =============================================================================
# 1.  TAICHI INIT
# =============================================================================
try:
    ti.init(arch=ti.gpu)
    print("[Taichi] GPU backend initialised (CUDA).")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Taichi] Running on CPU. Lower N_DISK if FPS < 30.")

# =============================================================================
# 2.  PHYSICAL CONSTANTS
# =============================================================================
GM        = 1.0          # G * M_bh  (gravitational parameter, world units)
C_LIGHT   = 300.0        # speed of light in world units

# GR characteristic radii (world units)
R_S    = 2.0 * GM / (C_LIGHT ** 2)   # Schwarzschild radius  r_s = 2GM/c^2
R_PH   = 1.5 * R_S                    # photon sphere          r_ph = 1.5 r_s
R_ISCO = 3.0 * R_S                    # ISCO                   r_isco = 3 r_s

# Canvas & world <-> pixel mapping
CANVAS_W, CANVAS_H = 1280, 800
WORLD_SCALE = 22.0 / R_S            # pixels per world unit  (~3.0e6)
WX = CANVAS_W / WORLD_SCALE         # half-scene width in world units
WY = CANVAS_H / WORLD_SCALE         # half-scene height in world units
SCENE_R = 0.5 * math.sqrt(WX**2 + WY**2)

# Radii in pixels (used in background draw kernel)
R_S_PX    = R_S    * WORLD_SCALE    # 22.0 px
R_PH_PX   = R_PH   * WORLD_SCALE    # 33.0 px
R_ISCO_PX = R_ISCO * WORLD_SCALE    # 66.0 px

print(f"[BHsim] r_s={R_S_PX:.1f}px   r_ph={R_PH_PX:.1f}px   r_ISCO={R_ISCO_PX:.1f}px")

# Disk radial extent (world units)
R_DISK_IN  = R_ISCO           # inner edge = ISCO
R_DISK_OUT = 38.0 * R_S       # outer disk edge

# Jet parameters
JET_V0      = 0.85 * C_LIGHT  # initial jet speed (85% of c)
JET_RADIUS  = 0.8 * R_S       # injection radius (just outside horizon)
HELIX_OMEGA = 1.8e8            # helical rotation rate for magnetic pinch (rad/t)
HELIX_FORCE = 0.008            # pinch force coefficient

EPS = 1e-9   # global epsilon -- prevents any division by zero

# =============================================================================
# 3.  PARTICLE COUNTS
# =============================================================================
N_DISK  = 180_000   # disk accretion particles
N_JET   =  14_000   # jet particles (both jets)
N_STARS =   3_500   # lensed background stars

# =============================================================================
# 4.  TAICHI FIELDS  (all per-particle state lives here -- never Python lists)
# =============================================================================
# Disk particles
pos_d  = ti.Vector.field(2, dtype=ti.f32, shape=N_DISK)
vel_d  = ti.Vector.field(2, dtype=ti.f32, shape=N_DISK)
col_d  = ti.Vector.field(3, dtype=ti.f32, shape=N_DISK)    # RGB for GUI
life_d = ti.field(dtype=ti.f32, shape=N_DISK)

# Jet particles
pos_j  = ti.Vector.field(2, dtype=ti.f32, shape=N_JET)
vel_j  = ti.Vector.field(2, dtype=ti.f32, shape=N_JET)
col_j  = ti.Vector.field(3, dtype=ti.f32, shape=N_JET)
life_j = ti.field(dtype=ti.f32, shape=N_JET)
velz_j = ti.field(dtype=ti.f32, shape=N_JET)   # out-of-plane (jet axis) velocity

# Background stars
star_base   = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)  # true position
star_lensed = ti.Vector.field(2, dtype=ti.f32, shape=N_STARS)  # lensed position
star_col    = ti.Vector.field(3, dtype=ti.f32, shape=N_STARS)

# Background image (rings + deep-space gradient) -- pixel raster
canvas_img = ti.Vector.field(3, dtype=ti.f32, shape=(CANVAS_W, CANVAS_H))

# Interactive
bh_mass    = ti.field(dtype=ti.f32, shape=())     # BH mass multiplier
inject_buf = ti.Vector.field(2, dtype=ti.f32, shape=1)  # cursor world-coord

# =============================================================================
# 5.  BLACKBODY COLOUR PALETTE
#     Built in NumPy and uploaded once to a Taichi field for O(1) lookup.
#     t=0 -> cold / deep-red (outer disk)
#     t=1 -> hot / blue-white (ISCO, inner disk)
# =============================================================================
PALETTE_N = 512
palette   = ti.Vector.field(3, dtype=ti.f32, shape=PALETTE_N)

def build_palette_np():
    """Piecewise blackbody colour approximation from t=0 (cold) to t=1 (hot)."""
    pal = np.zeros((PALETTE_N, 3), dtype=np.float32)
    for i in range(PALETTE_N):
        t = i / (PALETTE_N - 1.0)
        if t < 0.18:
            u = t / 0.18
            r, g, b = 0.30 + 0.50*u, 0.02 + 0.06*u, 0.0
        elif t < 0.40:
            u = (t - 0.18) / 0.22
            r, g, b = 0.80 + 0.15*u, 0.08 + 0.30*u, 0.0 + 0.02*u
        elif t < 0.62:
            u = (t - 0.40) / 0.22
            r, g, b = 0.95 + 0.05*u, 0.38 + 0.50*u, 0.02 + 0.18*u
        elif t < 0.82:
            u = (t - 0.62) / 0.20
            r, g, b = 1.00 + 0.0*u, 0.88 + 0.12*u, 0.20 + 0.50*u
        else:
            u = (t - 0.82) / 0.18
            r, g, b = 1.00 - 0.15*u, 0.95 + 0.05*u, 0.70 + 0.30*u
        pal[i] = np.clip([r, g, b], 0.0, 1.0)
    return pal


@ti.kernel
def upload_palette(pal_np: ti.types.ndarray(dtype=ti.f32, ndim=2)):
    for i in range(PALETTE_N):
        palette[i] = ti.Vector([pal_np[i, 0], pal_np[i, 1], pal_np[i, 2]])


# =============================================================================
# 6.  INLINE PHYSICS FUNCTIONS  (@ti.func -- gets inlined into calling kernels)
#     NOTE: Return-type annotations are dropped for Vector returns to avoid
#     Taichi 1.7.x TaichiSyntaxError (type-inference handles these fine).
# =============================================================================

@ti.func
def sample_palette(t: ti.f32):
    """Look up blackbody palette at temperature fraction t in [0, 1]."""
    tc  = ti.max(0.0, ti.min(1.0, t))
    idx = ti.cast(tc * (PALETTE_N - 1), ti.i32)
    return palette[idx]


@ti.func
def orbital_speed(r: ti.f32, gm: ti.f32) -> ti.f32:
    """Keplerian circular orbit speed: v_circ = sqrt(GM / r)."""
    return ti.sqrt(gm / (r + EPS))


@ti.func
def shakura_sunyaev_temp(r: ti.f32) -> ti.f32:
    """
    Shakura-Sunyaev accretion disk temperature profile:
        T(r)  proportional to  r^{-3/4}
    Normalised: T = 1 at r = R_ISCO, decreasing outward.
    Returns palette index fraction in [0, 1].
    """
    return ti.min(1.0, (R_ISCO / (r + EPS)) ** 0.75)


@ti.func
def doppler_beaming(velx: ti.f32, vely: ti.f32) -> ti.f32:
    """
    Relativistic Doppler beaming brightness boost.

    Formula (special relativity):
        delta = 1 / (gamma * (1 - beta * cos(theta)))

    where:
        beta      = |v| / c              (orbital speed fraction of c)
        gamma     = 1 / sqrt(1 - beta^2) (Lorentz factor)
        theta     = angle between velocity and line-of-sight (x-axis here)
        cos(theta)= v_x / |v|            (observer at +x infinity)

    Observed intensity boost ~ delta^3 (isotropic emitter in co-moving frame).
    """
    v_mag     = ti.sqrt(velx*velx + vely*vely) + EPS
    beta      = ti.min(v_mag / C_LIGHT, 0.9999)
    gamma     = 1.0 / ti.sqrt(1.0 - beta * beta + EPS)
    cos_theta = velx / (v_mag + EPS)          # v . x_hat / |v|
    delta     = 1.0 / (gamma * (1.0 - beta * cos_theta) + EPS)
    return ti.pow(delta, 3.0)                 # delta^3 intensity boost


@ti.func
def grav_redshift(r: ti.f32) -> ti.f32:
    """
    Gravitational redshift frequency ratio:
        f_obs / f_emit = sqrt(1 - r_s / r)

    -> 0 as r -> r_s  (infinite redshift at horizon; light fades and reddens)
    -> 1 as r -> inf  (no shift far from the hole)
    """
    return ti.sqrt(ti.max(0.0, 1.0 - R_S / (r + EPS)))


@ti.func
def grav_decel(vz: ti.f32) -> ti.f32:
    """Tiny gravitational deceleration on fast off-axis jet particles."""
    return GM * 0.3 / (vz * vz + EPS)


# =============================================================================
# 7.  INITIALISATION KERNELS
# =============================================================================

@ti.kernel
def init_disk():
    gm  = GM * bh_mass[None]
    TPI = 2.0 * ti.acos(-1.0)    # 2*pi computed inside kernel
    for i in range(N_DISK):
        # Inner-heavy radial sampling: squaring uniform RNG gives more particles
        # near the bright ISCO region (rho(r) biased toward small radii).
        u      = ti.random()
        r_frac = u * u
        r      = R_DISK_IN + r_frac * (R_DISK_OUT - R_DISK_IN)

        angle  = ti.random() * TPI
        v_circ = orbital_speed(r, gm)
        dv     = (ti.random() - 0.5) * 0.07 * v_circ   # +/-3.5% orbital perturbation

        pos_d[i]  = ti.Vector([r * ti.cos(angle), r * ti.sin(angle)])
        vel_d[i]  = ti.Vector([-(v_circ + dv) * ti.sin(angle),
                                 (v_circ + dv) * ti.cos(angle)])
        life_d[i] = ti.random() * 8000.0    # stagger orbital phases
        col_d[i]  = ti.Vector([1.0, 0.5, 0.1])  # placeholder; overwritten in update


@ti.kernel
def init_jets():
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_JET):
        side = 1.0 if i < N_JET // 2 else -1.0   # upper (+y) or lower (-y) jet

        ca = ti.random() * TPI
        cr = ti.random() * JET_RADIUS * 0.5
        pos_j[i]  = ti.Vector([cr * ti.cos(ca), 0.0])
        vel_j[i]  = ti.Vector([(ti.random() - 0.5) * JET_V0 * 0.04, 0.0])
        velz_j[i] = side * JET_V0 * (0.85 + ti.random() * 0.15)
        life_j[i] = ti.random() * 300.0

        if side > 0.0:
            col_j[i] = ti.Vector([0.15, 0.60, 1.0])    # electric cyan (up-jet)
        else:
            col_j[i] = ti.Vector([1.0, 0.20, 0.85])    # hot magenta (down-jet)


@ti.kernel
def init_stars():
    for i in range(N_STARS):
        bx = ti.random()
        by = ti.random()
        star_base[i]   = ti.Vector([bx, by])
        star_lensed[i] = ti.Vector([bx, by])
        rr = ti.random()
        if rr < 0.30:
            star_col[i] = ti.Vector([0.65, 0.75, 1.0])    # blue-white
        elif rr < 0.60:
            star_col[i] = ti.Vector([1.0, 1.0, 0.95])     # white
        elif rr < 0.80:
            star_col[i] = ti.Vector([1.0, 0.88, 0.45])    # yellow
        else:
            star_col[i] = ti.Vector([1.0, 0.55, 0.25])    # orange


# =============================================================================
# 8.  PHYSICS UPDATE KERNELS
# =============================================================================

@ti.kernel
def update_disk(dt: ti.f32):
    """
    Advance all disk particles by one timestep.

    Gravity model: Newtonian + post-Newtonian GR correction
        a = -(GM/r^3) * r_vec   +   (3 GM L^2)/(c^2 r^4) * r_hat
    where L = r x v (specific angular momentum, 2-D scalar = r cross v).

    Integration: semi-implicit Euler
        (1) v <- v + a * dt    (velocity updated from forces FIRST)
        (2) p <- p + v * dt    (position updated SECOND, using new velocity)
    This keeps orbits stable without energy drift.

    Per-particle colour from three physics effects:
        (i)  Shakura-Sunyaev T(r) ~ r^{-3/4} (inner disk is hotter/bluer)
        (ii) Doppler beaming delta^3           (approaching side is brighter)
        (iii)Gravitational redshift sqrt(1-r_s/r) (dims/reddens near horizon)
    """
    gm  = GM * bh_mass[None]
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_DISK):
        p = pos_d[i]
        v = vel_d[i]
        r = p.norm() + EPS

        # -- Gravity (Newtonian + post-Newtonian GR perihelion-advance term) ---
        L      = p[0] * v[1] - p[1] * v[0]           # 2-D angular momentum L = r x v
        a_newt = -(gm / (r * r * r)) * p              # Newtonian: -GM r_hat / r^2

        # Post-Newtonian GR correction (1PN, first-order Schwarzschild approx):
        #   delta_a = -(3 GM L^2) / (c^2 r^4) * r_hat
        # This makes orbits precess (equivalent to Mercury's precession formula).
        pn_fac = 3.0 * gm * L * L / (C_LIGHT * C_LIGHT * (r ** 4) + EPS)
        a_pn   = -(pn_fac / (r + EPS)) * p            # attractive, same direction
        a      = a_newt + a_pn

        # -- Semi-implicit Euler integration ----------------------------------
        v = v + a * dt    # (1) velocity from forces
        p = p + v * dt    # (2) position from new velocity

        # -- Re-spawn: crossed event horizon ----------------------------------
        # Gas inside r_s has been accreted. Re-inject into outer disk so the
        # scene stays populated forever (the "show never stops" requirement).
        if r < R_S * 1.05:
            ang   = ti.random() * TPI
            r_new = R_DISK_OUT * (0.45 + ti.random() * 0.55)
            vc    = orbital_speed(r_new, gm)
            dv    = (ti.random() - 0.5) * 0.05 * vc
            p = ti.Vector([r_new * ti.cos(ang),  r_new * ti.sin(ang)])
            v = ti.Vector([-(vc + dv) * ti.sin(ang), (vc + dv) * ti.cos(ang)])
            life_d[i] = 0.0

        # -- Re-spawn: escaped the disk boundary ------------------------------
        if p.norm() > R_DISK_OUT * 1.35:
            ang   = ti.random() * TPI
            r_new = R_DISK_IN + ti.random() * (R_DISK_OUT - R_DISK_IN) * 0.45
            vc    = orbital_speed(r_new, gm)
            dv    = (ti.random() - 0.5) * 0.05 * vc
            p = ti.Vector([r_new * ti.cos(ang),  r_new * ti.sin(ang)])
            v = ti.Vector([-(vc + dv) * ti.sin(ang), (vc + dv) * ti.cos(ang)])
            life_d[i] = 0.0

        pos_d[i]  = p
        vel_d[i]  = v
        life_d[i] += dt

        # -- Per-particle colour (all three GR effects combined) --------------
        r_cur = p.norm() + EPS

        # (i) Shakura-Sunyaev temperature -> blackbody colour
        temp     = shakura_sunyaev_temp(r_cur)
        base_rgb = sample_palette(temp)

        # (ii) Relativistic Doppler beaming: approaching side brighter
        beam = ti.max(0.25, ti.min(doppler_beaming(v[0], v[1]), 5.0))

        # (iii) Gravitational redshift: dims and reddens near horizon
        gz   = grav_redshift(r_cur)

        brightness = beam * gz * ti.min(1.2, 1.6 * temp + 0.15)

        # Redshift also shifts colour: blue channel fades faster (higher freq -> red)
        rc = ti.min(1.0, base_rgb[0] * brightness)
        gc = ti.min(1.0, base_rgb[1] * brightness * gz)
        bc = ti.min(1.0, base_rgb[2] * brightness * gz * gz)

        col_d[i] = ti.Vector([rc, gc, bc])


@ti.kernel
def update_jets(dt: ti.f32, sim_time: ti.f32):
    """
    Advance jet particles.

    Physics:
      * Initial v_z = +/-0.85 c (relativistic, perpendicular to disk plane)
      * Helical magnetic pinch (Lorentz force approximation):
          - Twist: rotate in-plane velocity by small angle each step
                   (mimics the toroidal B-field winding around the jet axis)
          - Pinch: centripetal force F ~ -x/|x| (collimates toward x=0 axis)
      * v_z projected onto screen y-axis
        (observer views disk face-on; jets go up/down on screen).
    """
    TPI = 2.0 * ti.acos(-1.0)
    for i in range(N_JET):
        p  = pos_j[i]
        v  = vel_j[i]
        vz = velz_j[i]
        side = 1.0 if vz > 0.0 else -1.0

        # -- Helical twist (magnetic field winding) ----------------------------
        # Toroidal B-field wraps around the jet axis; Lorentz force rotates
        # in-plane velocity components by a small angle each step.
        twist = HELIX_OMEGA * dt * 1e-9    # tiny rotation angle per step
        vx2 = v[0] * ti.cos(twist) - v[1] * ti.sin(twist)
        vy2 = v[0] * ti.sin(twist) + v[1] * ti.cos(twist)
        v   = ti.Vector([vx2, vy2])

        # -- Magnetic pinch force (toward jet axis x=0) -----------------------
        # Represents z-pinch of the toroidal current; keeps jet collimated.
        pinch = -HELIX_FORCE * p[0] / (ti.abs(p[0]) + EPS) * v.norm()
        v[0] += pinch * dt

        # -- Move (semi-implicit Euler) ----------------------------------------
        py = p[1] + vz * dt * side    # project z -> screen-y
        p  = ti.Vector([p[0] + v[0] * dt, py])

        # -- Re-spawn at injection zone ----------------------------------------
        max_y = WY * 0.96
        if ti.abs(p[1]) > max_y or p.norm() > SCENE_R * 1.15:
            ca  = ti.random() * TPI
            cr  = ti.random() * JET_RADIUS * 0.5
            p   = ti.Vector([cr * ti.cos(ca), 0.0])
            v   = ti.Vector([(ti.random() - 0.5) * JET_V0 * 0.04, 0.0])
            vz  = side * JET_V0 * (0.85 + ti.random() * 0.15)
            life_j[i] = 0.0

        pos_j[i]  = p
        vel_j[i]  = v
        velz_j[i] = vz
        life_j[i] += dt

        # -- Jet colour: fades with distance, pulses along helix ---------------
        dist_frac = ti.abs(p[1]) / (WY + EPS)
        fade  = ti.max(0.0, 1.0 - dist_frac * 1.05) ** 1.8
        pulse = 0.55 + 0.45 * ti.abs(ti.sin(p[1] * 900.0 + sim_time * 5.0))

        if side > 0.0:
            col_j[i] = ti.Vector([0.05 + 0.35*pulse, 0.45*pulse, 1.0*pulse]) * fade
        else:
            col_j[i] = ti.Vector([1.0*pulse, 0.15*pulse, 0.82*pulse]) * fade


@ti.kernel
def update_lensing():
    """
    Compute lensed apparent positions for background stars.

    Weak-field gravitational lensing (Einstein formula):
        alpha = 4GM / (c^2 * b)    (deflection angle)
    where b = impact parameter (distance from BH in lens plane).

    Stars appear pushed radially AWAY from the BH centre (the image of a
    source seen through a converging mass lens is pushed outward from the lens).
    Stars near the shadow edge arc into Einstein-ring segments.
    """
    bh_cx, bh_cy = 0.5, 0.5
    for i in range(N_STARS):
        bx = star_base[i][0]
        by = star_base[i][1]
        # Convert to world units for the formula
        dx = (bx - bh_cx) * WX * 2.0
        dy = (by - bh_cy) * WY * 2.0

        b2    = dx*dx + dy*dy + EPS
        b     = ti.sqrt(b2)
        alpha = 4.0 * GM / (C_LIGHT * C_LIGHT * b + EPS)  # deflection [wu/wu]
        alpha = ti.min(alpha, 0.85 * b)    # cap: prevent stars inverting through BH

        # Apparent source pushed outward from BH by alpha in the b direction
        sx = alpha * dx / (b + EPS)
        sy = alpha * dy / (b + EPS)

        lx = bx + sx / (WX * 2.0)
        ly = by + sy / (WY * 2.0)
        star_lensed[i] = ti.Vector([lx, ly])


# =============================================================================
# 9.  BACKGROUND IMAGE -- deep-space gradient + BH structural rings
# =============================================================================

@ti.kernel
def draw_background():
    """
    Per-pixel raster render of the static scene elements:
      (i)   Deep-space gradient background (near-black with faint violet nebula)
      (ii)  Event horizon shadow (solid black disc, r < r_s)
      (iii) Corona glow just outside r_s (hot accretion corona, orange-red)
      (iv)  Photon sphere ring (bright orange-white ring at r = 1.5 r_s)
      (v)   Lensing photon halo (faint diffuse arc between shadow and ISCO)
      (vi)  ISCO ring marker (soft cyan ring at r = 3 r_s = ISCO boundary)
    """
    cx = CANVAS_W // 2
    cy = CANVAS_H // 2

    for x, y in canvas_img:
        dx = ti.cast(x - cx, ti.f32)
        dy = ti.cast(y - cy, ti.f32)
        d  = ti.sqrt(dx*dx + dy*dy) + EPS

        # (i) Deep-space background: slight purple-violet nebula at edges
        edge = d / float(CANVAS_W)
        col = ti.Vector([0.006 + 0.018 * edge * edge,
                          0.004 + 0.008 * edge * edge,
                          0.013 + 0.032 * edge * edge])

        # (ii) Event horizon shadow
        if d < R_S_PX * 0.98:
            col = ti.Vector([0.0, 0.0, 0.0])

        # (iii) Corona: soft orange-red glow radiating from r_s outward
        corona_d = d - R_S_PX
        if 0.0 < corona_d < R_S_PX * 8.5:
            s = ti.exp(-corona_d * corona_d / (R_S_PX * R_S_PX * 5.5)) * 0.95
            col = col + ti.Vector([s * 0.95, s * 0.32, s * 0.05])

        # (iv) Photon sphere: bright narrow ring at r = 1.5 r_s
        ph_d = ti.abs(d - R_PH_PX)
        if ph_d < 3.2:
            s = ti.exp(-ph_d * ph_d / 2.2) * 1.4
            col = col + ti.Vector([s * 1.2, s * 0.82, s * 0.22])

        # (v) Lensing photon halo: diffuse glow ring
        halo_d = d - R_PH_PX * 1.75
        if 0.0 < halo_d < 20.0:
            s = ti.exp(-halo_d * halo_d / 14.0) * 0.30
            col = col + ti.Vector([s * 1.0, s * 0.62, s * 0.16])

        # (vi) ISCO ring: soft cyan boundary marker at r = 3 r_s
        isco_d = ti.abs(d - R_ISCO_PX)
        if isco_d < 2.8:
            s = ti.exp(-isco_d * isco_d / 1.8) * 0.42
            col = col + ti.Vector([s * 0.04, s * 0.68, s * 1.0])

        canvas_img[x, y] = ti.Vector([ti.min(col[0], 1.0),
                                       ti.min(col[1], 1.0),
                                       ti.min(col[2], 1.0)])


# =============================================================================
# 10.  GAS INJECTION AT CURSOR
# =============================================================================

@ti.kernel
def inject_gas():
    """
    Overwrite N_INJECT random disk particles with new gas clustered
    around the cursor world-position stored in inject_buf[0].
    Particles are given Keplerian orbital velocity at that radius.
    """
    N_INJECT = 600
    TPI = 2.0 * ti.acos(-1.0)
    gm  = GM * bh_mass[None]
    for _ in range(N_INJECT):
        ip      = inject_buf[0]
        r_ref   = ip.norm() + EPS
        angle   = ti.atan2(ip[1], ip[0])
        r_new   = ti.max(R_DISK_IN * 1.1,
                         r_ref + (ti.random() - 0.5) * R_DISK_IN * 0.6)
        ang_new = angle + (ti.random() - 0.5) * 0.4
        vc      = orbital_speed(r_new, gm)
        dv      = (ti.random() - 0.5) * 0.09 * vc
        idx     = ti.cast(ti.random() * N_DISK, ti.i32) % N_DISK
        pos_d[idx] = ti.Vector([r_new * ti.cos(ang_new),
                                  r_new * ti.sin(ang_new)])
        vel_d[idx] = ti.Vector([-(vc + dv) * ti.sin(ang_new),
                                   (vc + dv) * ti.cos(ang_new)])
        life_d[idx] = 0.0


# =============================================================================
# 11.  PIXEL-SPACE RENDERING  (accumulate particles onto canvas_img)
#      We add coloured "splats" for each particle directly into canvas_img
#      so we can use ti.GUI which works without Vulkan.
# =============================================================================

@ti.kernel
def render_particles_to_image(show_stars: ti.i32):
    """
    Splat each particle as a soft glowing dot onto canvas_img.
    The background has already been drawn by draw_background().
    We add particle light additively (bloom-like effect).
    """
    cx = ti.cast(CANVAS_W // 2, ti.f32)
    cy = ti.cast(CANVAS_H // 2, ti.f32)

    # -- Disk particles -------------------------------------------------------
    for i in range(N_DISK):
        p   = pos_d[i]
        rgb = col_d[i]
        # Convert world coords to pixel coords
        px_f = cx + p[0] * WORLD_SCALE
        py_f = cy + p[1] * WORLD_SCALE
        # Splat into a 2x2 pixel neighbourhood for soft dots
        for dx_px in range(-1, 2):
            for dy_px in range(-1, 2):
                ix = ti.cast(px_f, ti.i32) + dx_px
                iy = ti.cast(py_f, ti.i32) + dy_px
                if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                    # Gaussian weight for the 3x3 neighbourhood
                    w = ti.exp(-float(dx_px*dx_px + dy_px*dy_px) * 0.5) * 0.45
                    canvas_img[ix, iy] = ti.Vector([
                        ti.min(canvas_img[ix, iy][0] + rgb[0] * w, 1.0),
                        ti.min(canvas_img[ix, iy][1] + rgb[1] * w, 1.0),
                        ti.min(canvas_img[ix, iy][2] + rgb[2] * w, 1.0)])

    # -- Jet particles --------------------------------------------------------
    for i in range(N_JET):
        p   = pos_j[i]
        rgb = col_j[i]
        px_f = cx + p[0] * WORLD_SCALE
        py_f = cy + p[1] * WORLD_SCALE
        for dx_px in range(-1, 2):
            for dy_px in range(-1, 2):
                ix = ti.cast(px_f, ti.i32) + dx_px
                iy = ti.cast(py_f, ti.i32) + dy_px
                if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                    w = ti.exp(-float(dx_px*dx_px + dy_px*dy_px) * 0.5) * 0.55
                    canvas_img[ix, iy] = ti.Vector([
                        ti.min(canvas_img[ix, iy][0] + rgb[0] * w, 1.0),
                        ti.min(canvas_img[ix, iy][1] + rgb[1] * w, 1.0),
                        ti.min(canvas_img[ix, iy][2] + rgb[2] * w, 1.0)])

    # -- Background stars -----------------------------------------------------
    if show_stars == 1:
        for i in range(N_STARS):
            sp  = star_lensed[i]
            rgb = star_col[i]
            # Stars are in normalised [0,1] coords
            px_f = sp[0] * float(CANVAS_W)
            py_f = sp[1] * float(CANVAS_H)
            ix = ti.cast(px_f, ti.i32)
            iy = ti.cast(py_f, ti.i32)
            if 0 <= ix < CANVAS_W and 0 <= iy < CANVAS_H:
                w = 0.30
                canvas_img[ix, iy] = ti.Vector([
                    ti.min(canvas_img[ix, iy][0] + rgb[0] * w, 1.0),
                    ti.min(canvas_img[ix, iy][1] + rgb[1] * w, 1.0),
                    ti.min(canvas_img[ix, iy][2] + rgb[2] * w, 1.0)])


# =============================================================================
# 12.  MAIN LOOP
# =============================================================================
def main():
    print("=" * 68)
    print("  SCHWARZSCHILD BLACK HOLE - Taichi Real-Time Physics Simulation")
    print("=" * 68)
    print(f"  r_s  (event horizon)  = {R_S_PX:.1f} px   [{R_S:.4e} wu]")
    print(f"  r_ph (photon sphere)  = {R_PH_PX:.1f} px   [{R_PH:.4e} wu]")
    print(f"  r_isco (stable orbit) = {R_ISCO_PX:.1f} px   [{R_ISCO:.4e} wu]")
    print(f"  beta_isco = v_circ/c  = {math.sqrt(GM/R_ISCO)/C_LIGHT:.3f}  (strong Doppler beaming)")
    print(f"  Disk particles: {N_DISK:,}   Jet: {N_JET:,}   Stars: {N_STARS:,}")
    print()
    print("  CONTROLS:")
    print("    SPACE     -- Pause / Resume")
    print("    + / =     -- Speed up time (x2 per press, max 8x)")
    print("    -         -- Slow time (divide 2, min 0.125x)")
    print("    R         -- Reset all particles")
    print("    L         -- Toggle lensed background stars")
    print("    LMB click -- Inject fresh gas clump at cursor")
    print("    RMB drag  -- Drag up/down to change BH mass (0.3x - 3x)")
    print()
    print("  Physics active:")
    print("    [OK] Keplerian orbits + post-Newtonian GR perihelion advance")
    print("    [OK] Shakura-Sunyaev T(r) ~ r^{-3/4} temperature profile")
    print("    [OK] Relativistic Doppler beaming delta^3 (disk asymmetry)")
    print("    [OK] Gravitational redshift sqrt(1 - r_s/r)")
    print("    [OK] Weak-field lensing alpha = 4GM/c^2*b (Einstein-ring arcs)")
    print("    [OK] Helical magnetic-pinch relativistic jets")
    print("    [OK] ISCO inspiral re-spawn (disk replenishment)")
    print("=" * 68)

    # -- Initialise -----------------------------------------------------------
    pal_np = build_palette_np()
    upload_palette(pal_np)
    bh_mass[None] = 1.0
    init_disk()
    init_jets()
    init_stars()
    update_lensing()

    # Stable timestep: 400 steps per ISCO orbit
    T_isco  = 2.0 * math.pi * R_ISCO / math.sqrt(GM / R_ISCO)
    dt_base = T_isco / 400.0
    print(f"\n  ISCO orbital period = {T_isco:.4e} wu,  dt_base = {dt_base:.4e}")

    # -- GUI window (ti.GUI works without Vulkan) ------------------------------
    gui = ti.GUI(
        "Schwarzschild Black Hole",
        res=(CANVAS_W, CANVAS_H),
        fast_gui=True,   # directly presents frame buffer (fastest path)
        background_color=0x000000,
    )

    paused     = False
    time_scale = 1.0
    sim_time   = 0.0
    show_stars = True
    frame_cnt  = 0
    drag_prev_y = 0.5
    right_held  = False

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
                init_disk();  init_jets();  init_stars()
                update_lensing()
                sim_time = 0.0;  time_scale = 1.0;  paused = False
                print("[sim] Reset!")
            elif e.key in ('l', 'L'):
                show_stars = not show_stars
                print(f"[sim] Stars: {'ON' if show_stars else 'OFF'}")

        # -- Left click: inject gas -------------------------------------------
        if gui.is_pressed(ti.GUI.LMB):
            mx, my = gui.get_cursor_pos()
            wx_pos = (mx - 0.5) * WX * 2.0
            wy_pos = (my - 0.5) * WY * 2.0
            r_click = math.sqrt(wx_pos**2 + wy_pos**2)
            if r_click > R_DISK_IN * 1.05:
                inject_buf[0] = [wx_pos, wy_pos]
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

        # -- Physics step -----------------------------------------------------
        if not paused:
            dt = dt_base * time_scale
            sim_time += dt
            update_disk(dt)
            update_jets(dt, sim_time)
            if frame_cnt % 90 == 0 and show_stars:
                update_lensing()

        frame_cnt += 1

        # -- Render -----------------------------------------------------------
        # Redraw background (rings, corona, shadow) each frame
        draw_background()
        # Splat particles on top
        render_particles_to_image(1 if show_stars else 0)

        # Show the canvas_img via GUI
        gui.set_image(canvas_img)

        # HUD text
        gui.text(f"Time: {time_scale:.2f}x  BH Mass: {bh_mass[None]:.2f}x M0",
                 pos=(0.01, 0.98), color=0xCCCCCC, font_size=18)
        gui.text(f"{'[PAUSED]' if paused else ''}  Stars: {'ON' if show_stars else 'OFF'}",
                 pos=(0.01, 0.95), color=0xCCCCCC, font_size=18)
        gui.text("SPACE=Pause  +/-=Speed  R=Reset  L=Stars  LMB=inject  RMB=mass",
                 pos=(0.01, 0.92), color=0x888899, font_size=16)

        gui.show()

    print("[sim] Window closed.")


if __name__ == "__main__":
    main()
