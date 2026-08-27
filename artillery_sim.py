#!/usr/bin/env python3
"""
SIH26098 – Low-Cost Precision Guidance Kit for 155 mm Artillery Shell
=====================================================================
Monte-Carlo trajectory simulation demonstrating CEP improvement
from a Course Correction Fuze (canards) guidance system.

Educational / prototype demonstration only.
No explosive, fuze arming, or detonation details included.

Author : SIH Team
Version: 1.0
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple
import time
import math

# ============================================================
# SECTION 1 – TUNABLE PARAMETERS (edit these freely)
# ============================================================

@dataclass
class ShellParams:
    """Physical parameters of a 155 mm M795-class HE shell."""
    mass: float = 43.5          # kg – total projectile mass
    diameter: float = 0.155     # m  – calibre
    Cd0: float = 0.295          # –  – zero-yaw drag coefficient (subsonic approx)
    muzzle_vel: float = 827.0   # m/s – charge-dependent muzzle velocity
    spin_rate: float = 300.0    # Hz – rifling-induced spin (simplified, not used in 3-DOF)

    @property
    def ref_area(self) -> float:
        """Reference cross-section area [m²]."""
        return np.pi * (self.diameter / 2.0) ** 2


@dataclass
class FireMission:
    """Firing geometry and target definition."""
    elevation_rad: float = np.radians(45.0)   # Quadrant elevation [rad]
    azimuth_rad: float = 0.0                   # Firing azimuth from North [rad]
    target_range: float = 25000.0              # Desired range to target [m]
    target_cross: float = 0.0                  # Cross-range offset of target [m]


@dataclass
class SimConfig:
    """Numerical simulation settings."""
    dt: float = 0.01            # Integration time step [s]
    t_max: float = 120.0        # Maximum flight time [s]
    g: float = 9.80665          # Gravitational acceleration [m/s²]


# ============================================================
# SECTION 2 – STANDARD ATMOSPHERE (density vs altitude)
# ============================================================

def air_density(altitude_m: float) -> float:
    """
    International Standard Atmosphere (ISA) air density.
    Valid for troposphere (0-11 km) and lower stratosphere (11-20 km).
    Optimized for high-speed scalar computations.
    """
    if altitude_m < 0:
        altitude_m = 0.0

    if altitude_m <= 11000.0:
        T = 288.15 - 0.0065 * altitude_m
        p = 101325.0 * (T / 288.15) ** 5.2559
    elif altitude_m <= 20000.0:
        T = 216.65
        p = 22632.0 * math.exp(-9.80665 * (altitude_m - 11000.0) / 62192.1157)
    else:
        T = 216.65
        p = 5475.05  # Precomputed stratosphere limit: 22632 * exp(-9.80665 * 9000 / 62192.1157)

    return p / (287.058 * T)


# ============================================================
# SECTION 3 – EQUATIONS OF MOTION (3-DOF Point-Mass)
# ============================================================

def compute_derivatives(state: np.ndarray, shell: ShellParams,
                        cfg: SimConfig, wind: np.ndarray,
                        guidance_accel: np.ndarray) -> np.ndarray:
    """
    Compute derivatives for 3-DOF point-mass model.
    Optimized to minimize array allocations.
    """
    x, y, z, vx, vy, vz = state
    wx, wy, wz = wind
    gax, gay, gaz = guidance_accel

    vrx = vx - wx
    vry = vy - wy
    vrz = vz - wz
    speed = math.sqrt(vrx*vrx + vry*vry + vrz*vrz)

    if speed > 1e-6:
        rho = air_density(y)
        drag_factor = -0.5 * rho * shell.Cd0 * shell.ref_area * speed / shell.mass
        ax = drag_factor * vrx + gax
        ay = drag_factor * vry - cfg.g + gay
        az = drag_factor * vrz + gaz
    else:
        ax = gax
        ay = -cfg.g + gay
        az = gaz

    return np.array([vx, vy, vz, ax, ay, az])


# ============================================================
# SECTION 4 – RK4 INTEGRATOR
# ============================================================

def rk4_step(state: np.ndarray, dt: float, shell: ShellParams,
             cfg: SimConfig, wind: np.ndarray,
             guidance_accel: np.ndarray) -> np.ndarray:
    """
    Fourth-order Runge-Kutta integration step.
    Fully inlined using fast scalar math to eliminate numpy overhead in loop.
    """
    x, y, z, vx, vy, vz = state[0], state[1], state[2], state[3], state[4], state[5]
    wx, wy, wz = wind[0], wind[1], wind[2]
    gax, gay, gaz = guidance_accel[0], guidance_accel[1], guidance_accel[2]

    Cd_factor = -0.5 * shell.Cd0 * shell.ref_area / shell.mass
    g = cfg.g

    def get_derivs(y_val, vx_val, vy_val, vz_val):
        vrx = vx_val - wx
        vry = vy_val - wy
        vrz = vz_val - wz
        speed = math.sqrt(vrx*vrx + vry*vry + vrz*vrz)
        if speed > 1e-6:
            alt = y_val if y_val > 0 else 0.0
            if alt <= 11000.0:
                T = 288.15 - 0.0065 * alt
                p = 101325.0 * (T / 288.15) ** 5.2559
            elif alt <= 20000.0:
                T = 216.65
                p = 22632.0 * math.exp(-9.80665 * (alt - 11000.0) / 62192.1157)
            else:
                T = 216.65
                p = 5475.05
            
            rho = p / (287.058 * T)
            drag_factor = Cd_factor * rho * speed
            ax = drag_factor * vrx + gax
            ay = drag_factor * vry - g + gay
            az = drag_factor * vrz + gaz
        else:
            ax = gax
            ay = -g + gay
            az = gaz
        return vx_val, vy_val, vz_val, ax, ay, az

    # k1
    dx1, dy1, dz1, dvx1, dvy1, dvz1 = get_derivs(y, vx, vy, vz)

    # k2
    dt_half = 0.5 * dt
    dx2, dy2, dz2, dvx2, dvy2, dvz2 = get_derivs(
        y + dt_half * dy1,
        vx + dt_half * dvx1,
        vy + dt_half * dvy1,
        vz + dt_half * dvz1
    )

    # k3
    dx3, dy3, dz3, dvx3, dvy3, dvz3 = get_derivs(
        y + dt_half * dy2,
        vx + dt_half * dvx2,
        vy + dt_half * dvy2,
        vz + dt_half * dvz2
    )

    # k4
    dx4, dy4, dz4, dvx4, dvy4, dvz4 = get_derivs(
        y + dt * dy3,
        vx + dt * dvx3,
        vy + dt * dvy3,
        vz + dt * dvz3
    )

    # Combine
    dt_sixth = dt / 6.0
    x_next = x + dt_sixth * (dx1 + 2.0*dx2 + 2.0*dx3 + dx4)
    y_next = y + dt_sixth * (dy1 + 2.0*dy2 + 2.0*dy3 + dy4)
    z_next = z + dt_sixth * (dz1 + 2.0*dz2 + 2.0*dz3 + dz4)
    vx_next = vx + dt_sixth * (dvx1 + 2.0*dvx2 + 2.0*dvx3 + dvx4)
    vy_next = vy + dt_sixth * (dvy1 + 2.0*dvy2 + 2.0*dvy3 + dvy4)
    vz_next = vz + dt_sixth * (dvz1 + 2.0*dvz2 + 2.0*dvz3 + dvz4)

    return np.array([x_next, y_next, z_next, vx_next, vy_next, vz_next])


# ============================================================
# SECTION 5 – TRAJECTORY PROPAGATION
# ============================================================

def propagate_trajectory(shell: ShellParams, fire: FireMission,
                         cfg: SimConfig,
                         wind: np.ndarray = np.zeros(3),
                         guidance_func=None) -> dict:
    """
    Propagate a single trajectory from launch to ground impact.

    Returns dict with time histories and impact data.
    """
    # Initial velocity components
    v0 = shell.muzzle_vel
    vx0 = v0 * np.cos(fire.elevation_rad) * np.cos(fire.azimuth_rad)
    vy0 = v0 * np.sin(fire.elevation_rad)
    vz0 = v0 * np.cos(fire.elevation_rad) * np.sin(fire.azimuth_rad)

    # State: [x, y, z, vx, vy, vz]
    state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])

    # Storage
    t = 0.0
    history_t = [t]
    history_state = [state.copy()]
    history_guid = [np.zeros(3)]

    while t < cfg.t_max:
        # Guidance acceleration (zero if no guidance)
        if guidance_func is not None:
            g_accel = guidance_func(t, state, fire)
        else:
            g_accel = np.zeros(3)

        state = rk4_step(state, cfg.dt, shell, cfg, wind, g_accel)
        t += cfg.dt

        history_t.append(t)
        history_state.append(state.copy())
        history_guid.append(g_accel.copy())

        # Stop if projectile hits ground (y <= 0 after ascending)
        if state[1] <= 0.0 and t > 1.0:
            break

    history_state = np.array(history_state)
    history_guid = np.array(history_guid)
    history_t = np.array(history_t)

    # Impact point (interpolate to y=0)
    impact_x = history_state[-1, 0]
    impact_z = history_state[-1, 2]

    return {
        't': history_t,
        'x': history_state[:, 0],
        'y': history_state[:, 1],
        'z': history_state[:, 2],
        'vx': history_state[:, 3],
        'vy': history_state[:, 4],
        'vz': history_state[:, 5],
        'guid': history_guid,
        'impact_x': impact_x,
        'impact_z': impact_z,
        'flight_time': history_t[-1],
    }


# ============================================================
# SECTION 6 – WIND MODEL (altitude-layered)
# ============================================================

@dataclass
class WindModel:
    """
    Layered wind model: wind speed/direction varies with altitude.
    Each layer: (altitude_m, wind_x_mps, wind_z_mps)
      wind_x = along-range component (headwind positive)
      wind_z = cross-range component (right positive)
    Linearly interpolated between layers.
    """
    layers: list = None  # List of (alt, wx, wz) tuples

    def __post_init__(self):
        if self.layers is None:
            # Default: moderate crosswind increasing with altitude
            self.layers = [
                (0,     2.0,  3.0),    # Surface: light
                (1000,  4.0,  6.0),    # 1 km
                (3000,  6.0,  10.0),   # 3 km
                (5000,  5.0,  8.0),    # 5 km
                (8000,  3.0,  5.0),    # 8 km
            ]

    def get_wind(self, altitude_m: float) -> np.ndarray:
        """Return [wx, wy, wz] wind vector at given altitude. wy=0 (no vertical wind)."""
        layers = self.layers
        if altitude_m <= layers[0][0]:
            return np.array([layers[0][1], 0.0, layers[0][2]])
        if altitude_m >= layers[-1][0]:
            return np.array([layers[-1][1], 0.0, layers[-1][2]])

        # Linear interpolation
        for i in range(len(layers) - 1):
            a0, wx0, wz0 = layers[i]
            a1, wx1, wz1 = layers[i + 1]
            if a0 <= altitude_m <= a1:
                frac = (altitude_m - a0) / (a1 - a0)
                wx = wx0 + frac * (wx1 - wx0)
                wz = wz0 + frac * (wz1 - wz0)
                return np.array([wx, 0.0, wz])

        return np.zeros(3)


def propagate_trajectory_wind(shell: ShellParams, fire: FireMission,
                              cfg: SimConfig,
                              wind_model: WindModel = None,
                              guidance_func=None) -> dict:
    """
    Propagate trajectory with altitude-dependent wind model.
    Like propagate_trajectory but uses WindModel instead of constant wind.
    """
    if wind_model is None:
        wind_model = WindModel(layers=[(0, 0, 0)])

    v0 = shell.muzzle_vel
    vx0 = v0 * np.cos(fire.elevation_rad) * np.cos(fire.azimuth_rad)
    vy0 = v0 * np.sin(fire.elevation_rad)
    vz0 = v0 * np.cos(fire.elevation_rad) * np.sin(fire.azimuth_rad)

    state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])

    t = 0.0
    history_t = [t]
    history_state = [state.copy()]
    history_guid = [np.zeros(3)]

    while t < cfg.t_max:
        # Get wind at current altitude
        wind = wind_model.get_wind(max(state[1], 0.0))

        if guidance_func is not None:
            g_accel = guidance_func(t, state, fire)
        else:
            g_accel = np.zeros(3)

        state = rk4_step(state, cfg.dt, shell, cfg, wind, g_accel)
        t += cfg.dt

        history_t.append(t)
        history_state.append(state.copy())
        history_guid.append(g_accel.copy())

        if state[1] <= 0.0 and t > 1.0:
            break

    history_state = np.array(history_state)
    history_guid = np.array(history_guid)
    history_t = np.array(history_t)

    return {
        't': history_t,
        'x': history_state[:, 0],
        'y': history_state[:, 1],
        'z': history_state[:, 2],
        'vx': history_state[:, 3],
        'vy': history_state[:, 4],
        'vz': history_state[:, 5],
        'guid': history_guid,
        'impact_x': history_state[-1, 0],
        'impact_z': history_state[-1, 2],
        'flight_time': history_t[-1],
    }


# ============================================================
# SECTION 7 – MONTE CARLO ERROR SOURCES
# ============================================================

@dataclass
class ErrorSources:
    """
    Statistical error sources for unguided artillery fire.
    Each parameter is a 1-sigma standard deviation.
    """
    mv_sigma: float = 3.0          # Muzzle velocity variation [m/s]
    elev_sigma_mrad: float = 0.5   # Elevation angle error [mrad]
    azim_sigma_mrad: float = 0.3   # Azimuth angle error [mrad]
    cd_sigma_pct: float = 3.0      # Drag coefficient variation [%]
    mass_sigma: float = 0.5        # Mass variation [kg]
    wind_sigma: float = 2.0        # Wind uncertainty per component [m/s]


def compute_cep(range_errors: np.ndarray, cross_errors: np.ndarray) -> float:
    """
    Compute Circular Error Probable (CEP) – radius within which
    50% of rounds impact. Uses the median radial miss distance.
    """
    radii = np.sqrt(range_errors**2 + cross_errors**2)
    return np.median(radii)


def run_monte_carlo_unguided(n_runs: int, shell_base: ShellParams,
                              fire_base: FireMission, cfg: SimConfig,
                              wind_base: WindModel, errors: ErrorSources,
                              rng: np.random.Generator = None) -> dict:
    """
    Run n_runs Monte-Carlo simulations with randomized error sources.
    Returns impact points and trajectories.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Nominal impact (no errors, with nominal wind)
    r_nom = propagate_trajectory_wind(shell_base, fire_base, cfg, wind_base)
    nom_x = r_nom['impact_x']
    nom_z = r_nom['impact_z']

    impacts_x = []
    impacts_z = []
    trajectories = []

    for i in range(n_runs):
        # Perturb parameters
        shell = ShellParams(
            mass=shell_base.mass + rng.normal(0, errors.mass_sigma),
            diameter=shell_base.diameter,
            Cd0=shell_base.Cd0 * (1.0 + rng.normal(0, errors.cd_sigma_pct / 100.0)),
            muzzle_vel=shell_base.muzzle_vel + rng.normal(0, errors.mv_sigma),
        )

        fire = FireMission(
            elevation_rad=fire_base.elevation_rad + rng.normal(0, errors.elev_sigma_mrad * 1e-3),
            azimuth_rad=fire_base.azimuth_rad + rng.normal(0, errors.azim_sigma_mrad * 1e-3),
            target_range=fire_base.target_range,
            target_cross=fire_base.target_cross,
        )

        # Perturb wind
        perturbed_layers = []
        for alt, wx, wz in wind_base.layers:
            pwx = wx + rng.normal(0, errors.wind_sigma)
            pwz = wz + rng.normal(0, errors.wind_sigma)
            perturbed_layers.append((alt, pwx, pwz))
        wind = WindModel(layers=perturbed_layers)

        result = propagate_trajectory_wind(shell, fire, cfg, wind)
        impacts_x.append(result['impact_x'])
        impacts_z.append(result['impact_z'])

        # Store a subset of trajectories for plotting
        if i < 20:
            trajectories.append(result)

    impacts_x = np.array(impacts_x)
    impacts_z = np.array(impacts_z)

    # Errors relative to nominal
    range_err = impacts_x - nom_x
    cross_err = impacts_z - nom_z

    return {
        'impacts_x': impacts_x,
        'impacts_z': impacts_z,
        'range_err': range_err,
        'cross_err': cross_err,
        'nom_x': nom_x,
        'nom_z': nom_z,
        'cep': compute_cep(range_err, cross_err),
        'trajectories': trajectories,
    }


# ============================================================
# SECTION 8 – GUIDANCE LAW (Course Correction Fuze)
# ============================================================

@dataclass
class GuidanceParams:
    """
    Parameters for the Course Correction Fuze guidance system.
    """
    gnss_lock_time: float = 5.0     # Time before GNSS position available [s]
    max_accel_g: float = 5.0         # Maximum canard lateral accel [g's]
    nav_gain_range: float = 3.0      # Proportional nav gain (range axis)
    nav_gain_cross: float = 4.0      # Proportional nav gain (cross-range axis)
    cmd_tau: float = 0.2             # Command filter time constant [s]
    update_rate: float = 10.0        # GNSS update rate [Hz]
    gnss_noise_m: float = 3.0        # GNSS position noise 1-sigma [m]


def make_guidance_func(guid_params: GuidanceParams, target_x: float,
                       target_z: float, cfg: SimConfig):
    """
    Create a guidance closure for Course Correction Fuze.

    The CCF canards generate aerodynamic forces PERPENDICULAR to the
    velocity vector. This means:
    - Cross-range correction (z-axis): direct lateral force
    - Range correction: achieved by modifying the trajectory arc via
      vertical (y-axis) force — pulling up extends range, pushing down shortens.
    - No along-track (x-axis) thrust is possible from canards.

    Uses Zero Effort Miss (ZEM) based guidance law.
    """
    g = cfg.g
    max_accel = guid_params.max_accel_g * g
    
    # Initialize generator once for this simulation run
    rng = np.random.default_rng(42)
    
    state_holder = {
        'last_cmd': np.zeros(3),
        'last_update_t': -1.0,
    }

    def guidance_func(t: float, state: np.ndarray, fire: FireMission) -> np.ndarray:
        # No guidance before GNSS lock
        if t < guid_params.gnss_lock_time:
            return np.zeros(3)

        x, y, z, vx, vy, vz = state

        # GNSS update at discrete rate
        dt_update = 1.0 / guid_params.update_rate
        if t - state_holder['last_update_t'] < dt_update:
            return state_holder['last_cmd'].copy()

        state_holder['last_update_t'] = t

        # Add GNSS noise using the shared generator
        x_meas = x + rng.normal(0, guid_params.gnss_noise_m)
        z_meas = z + rng.normal(0, guid_params.gnss_noise_m)
        y_meas = y  # Altitude from GNSS (less noisy in practice)

        # --- Estimate time-to-impact (t_go) ---
        # Use vertical kinematics: y + vy*t_go - 0.5*g*t_go² = 0
        # Quadratic: -0.5*g*t_go² + vy*t_go + y = 0
        if y_meas > 10.0:  # Only guide when above ground
            disc = vy**2 + 2.0 * g * y_meas
            if disc > 0:
                t_go = (vy + np.sqrt(disc)) / g
            else:
                t_go = 2.0
        else:
            t_go = 0.5
        t_go = max(t_go, 0.5)

        # --- Zero Effort Miss (ZEM) ---
        # Predicted impact without guidance:
        #   x_impact_pred = x + vx * t_go
        #   z_impact_pred = z + vz * t_go
        zem_x = (x_meas + vx * t_go) - target_x   # Overshoot if positive
        zem_z = (z_meas + vz * t_go) - target_z   # Right miss if positive

        # --- Acceleration commands ---
        # Cross-range: direct lateral correction
        # a_z = -N * zem_z / (0.5 * t_go²)
        az_cmd = -guid_params.nav_gain_cross * zem_z / (0.5 * t_go**2)

        # Range: correct via vertical force (change trajectory arc)
        # If overshooting (zem_x > 0) → push down (negative ay) to shorten
        # If undershooting (zem_x < 0) → pull up (positive ay) to extend
        ay_cmd = -guid_params.nav_gain_range * zem_x / (0.5 * t_go**2)

        # Canards produce forces perpendicular to velocity — no x-axis thrust
        cmd = np.array([0.0, ay_cmd, az_cmd])

        # Limit to max canard authority
        cmd_mag = np.linalg.norm(cmd)
        if cmd_mag > max_accel:
            cmd = cmd * (max_accel / cmd_mag)

        # First-order command filter (smooth actuator response)
        alpha = min(cfg.dt / guid_params.cmd_tau, 1.0)
        filtered_cmd = state_holder['last_cmd'] * (1 - alpha) + cmd * alpha
        state_holder['last_cmd'] = filtered_cmd

        return filtered_cmd

    return guidance_func


def run_monte_carlo_guided(n_runs: int, shell_base: ShellParams,
                            fire_base: FireMission, cfg: SimConfig,
                            wind_base: WindModel, errors: ErrorSources,
                            guid_params: GuidanceParams,
                            target_x: float, target_z: float,
                            rng: np.random.Generator = None) -> dict:
    """
    Run guided Monte-Carlo simulations.
    Target position is specified in absolute coordinates.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    impacts_x = []
    impacts_z = []
    trajectories = []

    for i in range(n_runs):
        # Perturb parameters (same as unguided)
        shell = ShellParams(
            mass=shell_base.mass + rng.normal(0, errors.mass_sigma),
            diameter=shell_base.diameter,
            Cd0=shell_base.Cd0 * (1.0 + rng.normal(0, errors.cd_sigma_pct / 100.0)),
            muzzle_vel=shell_base.muzzle_vel + rng.normal(0, errors.mv_sigma),
        )

        fire = FireMission(
            elevation_rad=fire_base.elevation_rad + rng.normal(0, errors.elev_sigma_mrad * 1e-3),
            azimuth_rad=fire_base.azimuth_rad + rng.normal(0, errors.azim_sigma_mrad * 1e-3),
            target_range=fire_base.target_range,
            target_cross=fire_base.target_cross,
        )

        perturbed_layers = []
        for alt, wx, wz in wind_base.layers:
            pwx = wx + rng.normal(0, errors.wind_sigma)
            pwz = wz + rng.normal(0, errors.wind_sigma)
            perturbed_layers.append((alt, pwx, pwz))
        wind = WindModel(layers=perturbed_layers)

        # Create guidance function targeting the desired point
        guid_func = make_guidance_func(guid_params, target_x, target_z, cfg)

        result = propagate_trajectory_wind(shell, fire, cfg, wind, guid_func)
        impacts_x.append(result['impact_x'])
        impacts_z.append(result['impact_z'])

        if i < 20:
            trajectories.append(result)

    impacts_x = np.array(impacts_x)
    impacts_z = np.array(impacts_z)

    range_err = impacts_x - target_x
    cross_err = impacts_z - target_z

    return {
        'impacts_x': impacts_x,
        'impacts_z': impacts_z,
        'range_err': range_err,
        'cross_err': cross_err,
        'target_x': target_x,
        'target_z': target_z,
        'cep': compute_cep(range_err, cross_err),
        'trajectories': trajectories,
    }


# ============================================================
# FULL SIMULATION – Unguided + Guided Monte Carlo Comparison
# ============================================================

SAVE_DIR = '/home/rishikanth/.gemini/antigravity/scratch/sih26098_guidance_sim'

if __name__ == "__main__":
    print("=" * 70)
    print("  SIH26098 – Low-Cost Precision Guidance Kit for 155 mm Shell")
    print("  Monte-Carlo CEP Comparison: Unguided vs Guided (CCF)")
    print("=" * 70)

    # --- Configuration ---
    N_RUNS = 100
    shell = ShellParams()
    fire = FireMission(elevation_rad=np.radians(43.0))
    cfg = SimConfig()
    wind = WindModel()
    errors = ErrorSources()
    guid_params = GuidanceParams()

    print(f"\n{'─'*40}")
    print(f"SHELL PARAMETERS")
    print(f"  Mass          : {shell.mass} kg")
    print(f"  Muzzle velocity: {shell.muzzle_vel} m/s")
    print(f"  Calibre       : {shell.diameter*1000:.0f} mm")
    print(f"  Cd            : {shell.Cd0}")
    print(f"  Elevation     : {np.degrees(fire.elevation_rad):.1f}°")

    print(f"\n{'─'*40}")
    print(f"ERROR SOURCES (1-σ)")
    print(f"  MV variation  : ±{errors.mv_sigma} m/s")
    print(f"  Elevation     : ±{errors.elev_sigma_mrad} mrad")
    print(f"  Azimuth       : ±{errors.azim_sigma_mrad} mrad")
    print(f"  Cd variation  : ±{errors.cd_sigma_pct}%")
    print(f"  Mass variation: ±{errors.mass_sigma} kg")
    print(f"  Wind noise    : ±{errors.wind_sigma} m/s")

    print(f"\n{'─'*40}")
    print(f"GUIDANCE PARAMETERS")
    print(f"  GNSS lock time: {guid_params.gnss_lock_time} s")
    print(f"  Max canard accel: {guid_params.max_accel_g} g")
    print(f"  GNSS noise    : {guid_params.gnss_noise_m} m (1-σ)")
    print(f"  Update rate   : {guid_params.update_rate} Hz")

    # --- Get nominal target ---
    r_nom = propagate_trajectory_wind(shell, fire, cfg, wind)
    target_x = r_nom['impact_x']
    target_z = r_nom['impact_z']
    print(f"\n{'─'*40}")
    print(f"NOMINAL TRAJECTORY")
    print(f"  Range      : {target_x/1000:.2f} km")
    print(f"  Cross-range: {target_z:.1f} m")
    print(f"  Flight time: {r_nom['flight_time']:.1f} s")
    print(f"  Max altitude: {np.max(r_nom['y'])/1000:.2f} km")

    # --- Unguided Monte Carlo ---
    print(f"\n{'─'*40}")
    print(f"Running UNGUIDED Monte Carlo ({N_RUNS} runs)...")
    t0 = time.time()
    mc_ung = run_monte_carlo_unguided(N_RUNS, shell, fire, cfg, wind, errors)
    t_ung = time.time() - t0
    print(f"  Done in {t_ung:.1f}s")

    # --- Guided Monte Carlo ---
    print(f"Running GUIDED Monte Carlo ({N_RUNS} runs)...")
    t0 = time.time()
    mc_gui = run_monte_carlo_guided(N_RUNS, shell, fire, cfg, wind, errors,
                                     guid_params, target_x, target_z)
    t_gui = time.time() - t0
    print(f"  Done in {t_gui:.1f}s")

    # ====================================================
    # SUMMARY STATISTICS
    # ====================================================
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'':30s} {'UNGUIDED':>12s} {'GUIDED':>12s}")
    print(f"{'─'*54}")
    print(f"{'CEP (m)':30s} {mc_ung['cep']:12.1f} {mc_gui['cep']:12.1f}")
    print(f"{'Mean range error (m)':30s} {np.mean(mc_ung['range_err']):12.1f} {np.mean(mc_gui['range_err']):12.1f}")
    print(f"{'Mean cross error (m)':30s} {np.mean(mc_ung['cross_err']):12.1f} {np.mean(mc_gui['cross_err']):12.1f}")
    print(f"{'Std range error (m)':30s} {np.std(mc_ung['range_err']):12.1f} {np.std(mc_gui['range_err']):12.1f}")
    print(f"{'Std cross error (m)':30s} {np.std(mc_ung['cross_err']):12.1f} {np.std(mc_gui['cross_err']):12.1f}")
    radii_ung = np.sqrt(mc_ung['range_err']**2 + mc_ung['cross_err']**2)
    radii_gui = np.sqrt(mc_gui['range_err']**2 + mc_gui['cross_err']**2)
    print(f"{'Max miss (m)':30s} {np.max(radii_ung):12.1f} {np.max(radii_gui):12.1f}")
    print(f"{'Min miss (m)':30s} {np.min(radii_ung):12.1f} {np.min(radii_gui):12.1f}")
    print(f"{'─'*54}")
    print(f"{'CEP improvement factor':30s} {'':>12s} {mc_ung['cep']/max(mc_gui['cep'],0.1):12.1f}x")
    print(f"\n  ✅ Guided CEP = {mc_gui['cep']:.1f} m  (target: ≤ 30 m)")
    if mc_gui['cep'] <= 30:
        print(f"  ✅ REQUIREMENT MET: CEP ≤ 30 m")
    else:
        print(f"  ⚠️  CEP > 30 m — tune guidance gains")

    # ====================================================
    # PLOT 1: Side-by-side trajectory comparison
    # ====================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for traj in mc_ung['trajectories'][:15]:
        ax.plot(traj['x']/1000, traj['y']/1000, 'r-', alpha=0.25, lw=0.7)
    ax.plot([], [], 'r-', alpha=0.6, label='Unguided')
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Unguided Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for traj in mc_gui['trajectories'][:15]:
        ax.plot(traj['x']/1000, traj['y']/1000, 'g-', alpha=0.25, lw=0.7)
    ax.plot([], [], 'g-', alpha=0.6, label='Guided (CCF)')
    ax.set_xlabel('Downrange (km)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title('Guided Trajectories (Course Correction Fuze)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('SIH26098 – 155 mm Shell Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot1_trajectories.png', dpi=150)
    print(f"\nPlot saved: plot1_trajectories.png")

    # ====================================================
    # PLOT 2: Impact scatter with CEP circles
    # ====================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    theta = np.linspace(0, 2*np.pi, 200)

    # Unguided
    ax = axes[0]
    ax.scatter(mc_ung['range_err'], mc_ung['cross_err'], c='red', s=20, alpha=0.5, label='Impacts')
    ax.scatter(0, 0, c='black', marker='+', s=200, lw=2)
    ax.plot(mc_ung['cep']*np.cos(theta), mc_ung['cep']*np.sin(theta), 'k--', lw=2,
            label=f'CEP = {mc_ung["cep"]:.0f} m')
    ax.set_xlabel('Range Error (m)')
    ax.set_ylabel('Cross-range Error (m)')
    ax.set_title(f'Unguided Impact Scatter (CEP = {mc_ung["cep"]:.0f} m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Guided
    ax = axes[1]
    ax.scatter(mc_gui['range_err'], mc_gui['cross_err'], c='green', s=20, alpha=0.5, label='Impacts')
    ax.scatter(0, 0, c='black', marker='+', s=200, lw=2)
    ax.plot(mc_gui['cep']*np.cos(theta), mc_gui['cep']*np.sin(theta), 'k--', lw=2,
            label=f'CEP = {mc_gui["cep"]:.1f} m')
    # Also show 30m circle for reference
    ax.plot(30*np.cos(theta), 30*np.sin(theta), 'b:', lw=1.5,
            label='30 m requirement')
    ax.set_xlabel('Range Error (m)')
    ax.set_ylabel('Cross-range Error (m)')
    ax.set_title(f'Guided Impact Scatter (CEP = {mc_gui["cep"]:.1f} m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.suptitle('SIH26098 – Impact Point Scatter Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot2_scatter.png', dpi=150)
    print(f"Plot saved: plot2_scatter.png")

    # ====================================================
    # PLOT 3: Combined scatter overlay
    # ====================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(mc_ung['range_err'], mc_ung['cross_err'], c='red', s=20, alpha=0.4, label=f'Unguided (CEP={mc_ung["cep"]:.0f}m)')
    ax.scatter(mc_gui['range_err'], mc_gui['cross_err'], c='green', s=25, alpha=0.6, label=f'Guided (CEP={mc_gui["cep"]:.1f}m)')
    ax.scatter(0, 0, c='black', marker='+', s=300, lw=3, zorder=10, label='Target')
    ax.plot(mc_ung['cep']*np.cos(theta), mc_ung['cep']*np.sin(theta), 'r--', lw=1.5)
    ax.plot(mc_gui['cep']*np.cos(theta), mc_gui['cep']*np.sin(theta), 'g--', lw=2)
    ax.plot(30*np.cos(theta), 30*np.sin(theta), 'b:', lw=1.5, label='30 m requirement')
    ax.set_xlabel('Range Error (m)', fontsize=12)
    ax.set_ylabel('Cross-range Error (m)', fontsize=12)
    ax.set_title(f'SIH26098 – CEP Comparison: {mc_ung["cep"]:.0f}m → {mc_gui["cep"]:.1f}m', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot3_cep_comparison.png', dpi=150)
    print(f"Plot saved: plot3_cep_comparison.png")

    # ====================================================
    # PLOT 4: Time histories from a single guided run
    # ====================================================
    traj = mc_gui['trajectories'][0]  # First guided trajectory
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Position
    ax = axes[0, 0]
    ax.plot(traj['t'], traj['x']/1000, 'b-', lw=1.2, label='Downrange')
    ax.plot(traj['t'], traj['y']/1000, 'r-', lw=1.2, label='Altitude')
    ax.plot(traj['t'], traj['z'], 'g-', lw=1.2, label='Cross-range (m)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position')
    ax.set_title('Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    speed = np.sqrt(traj['vx']**2 + traj['vy']**2 + traj['vz']**2)
    ax.plot(traj['t'], speed, 'b-', lw=1.2, label='Total speed')
    ax.plot(traj['t'], traj['vx'], 'r-', lw=0.8, alpha=0.6, label='Vx (downrange)')
    ax.plot(traj['t'], traj['vy'], 'g-', lw=0.8, alpha=0.6, label='Vy (vertical)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Guidance acceleration
    ax = axes[1, 0]
    g_total = np.sqrt(traj['guid'][:, 1]**2 + traj['guid'][:, 2]**2) / 9.80665
    ax.plot(traj['t'], traj['guid'][:, 1] / 9.80665, 'b-', lw=1, label='Vertical (range corr.)')
    ax.plot(traj['t'], traj['guid'][:, 2] / 9.80665, 'r-', lw=1, label='Lateral (cross corr.)')
    ax.plot(traj['t'], g_total, 'k-', lw=1.5, alpha=0.5, label='|Total|')
    ax.axhline(y=guid_params.max_accel_g, color='gray', ls='--', alpha=0.4)
    ax.axhline(y=-guid_params.max_accel_g, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Canard Accel (g)')
    ax.set_title('Canard Correction Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Miss distance vs time
    ax = axes[1, 1]
    miss_vs_t = np.sqrt((traj['x'] - target_x)**2 + (traj['z'] - target_z)**2)
    ax.plot(traj['t'], miss_vs_t/1000, 'g-', lw=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ground Distance to Target (km)')
    ax.set_title('Closing on Target')
    ax.grid(True, alpha=0.3)

    plt.suptitle('SIH26098 – Guided Trajectory Time Histories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/plot4_time_histories.png', dpi=150)
    print(f"Plot saved: plot4_time_histories.png")

    print(f"\n{'='*70}")
    print(f"  ALL STEPS COMPLETE ✅")
    print(f"  Total simulation time: {t_ung + t_gui:.1f}s")
    print(f"{'='*70}")
