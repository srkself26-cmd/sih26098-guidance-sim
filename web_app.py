import os
import sys
import time
import uuid
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import simulation classes and functions from artillery_sim
from artillery_sim import (
    ShellParams, FireMission, SimConfig, WindModel, ErrorSources,
    GuidanceParams, propagate_trajectory_wind, run_monte_carlo_unguided,
    run_monte_carlo_guided, make_guidance_func, compute_cep
)

app = Flask(__name__)

# Create directories if they do not exist
os.makedirs('static/images', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/simulate', methods=['POST'])
def simulate():
    try:
        # Clean up old plots to save disk space
        import glob
        for f in glob.glob('static/images/*.png'):
            try:
                if time.time() - os.path.getmtime(f) > 30:  # Keep files for 30s
                    os.remove(f)
            except Exception:
                pass

        data = request.json or {}
        
        # Parse inputs with defaults
        mass = float(data.get('mass', 43.5))
        muzzle_vel = float(data.get('muzzle_vel', 827.0))
        elevation_deg = float(data.get('elevation', 43.0))
        n_runs = int(data.get('n_runs', 50))
        dt = float(data.get('dt', 0.05))  # Default to 0.05 for web app speed/accuracy balance
        
        # Error sources
        mv_sigma = float(data.get('mv_sigma', 3.0))
        elev_sigma_mrad = float(data.get('elev_sigma_mrad', 0.5))
        azim_sigma_mrad = float(data.get('azim_sigma_mrad', 0.3))
        cd_sigma_pct = float(data.get('cd_sigma_pct', 3.0))
        mass_sigma = float(data.get('mass_sigma', 0.5))
        wind_sigma = float(data.get('wind_sigma', 2.0))
        
        # Wind conditions
        wind_surface_x = float(data.get('wind_surface_x', 2.0))
        wind_surface_z = float(data.get('wind_surface_z', 3.0))
        
        # Guidance parameters
        gnss_lock_time = float(data.get('gnss_lock_time', 5.0))
        max_accel_g = float(data.get('max_accel_g', 5.0))
        nav_gain_range = float(data.get('nav_gain_range', 3.0))
        nav_gain_cross = float(data.get('nav_gain_cross', 4.0))
        gnss_noise_m = float(data.get('gnss_noise_m', 3.0))
        
        # Setup dataclasses
        shell = ShellParams(mass=mass, muzzle_vel=muzzle_vel, Cd0=0.295)
        fire = FireMission(elevation_rad=np.radians(elevation_deg))
        cfg = SimConfig(dt=dt)
        
        # Custom wind layers based on surface inputs
        wind_base = WindModel(layers=[
            (0,     wind_surface_x,       wind_surface_z),
            (1000,  wind_surface_x * 1.5, wind_surface_z * 1.5),
            (3000,  wind_surface_x * 2.0, wind_surface_z * 2.0),
            (5000,  wind_surface_x * 2.5, wind_surface_z * 2.2),
            (8000,  wind_surface_x * 3.0, wind_surface_z * 2.5),
        ])
        
        errors = ErrorSources(
            mv_sigma=mv_sigma,
            elev_sigma_mrad=elev_sigma_mrad,
            azim_sigma_mrad=azim_sigma_mrad,
            cd_sigma_pct=cd_sigma_pct,
            mass_sigma=mass_sigma,
            wind_sigma=wind_sigma
        )
        
        guid_params = GuidanceParams(
            gnss_lock_time=gnss_lock_time,
            max_accel_g=max_accel_g,
            nav_gain_range=nav_gain_range,
            nav_gain_cross=nav_gain_cross,
            gnss_noise_m=gnss_noise_m
        )
        
        # 1. Nominal trajectory to find target (use high precision dt=0.02 for accurate target definition)
        cfg_nominal = SimConfig(dt=0.02)
        r_nom = propagate_trajectory_wind(shell, fire, cfg_nominal, wind_base)
        target_x = r_nom['impact_x']
        target_z = r_nom['impact_z']
        
        # 2. Run unguided Monte Carlo (ballistic flight is extremely smooth; dt=0.1s is very fast and accurate)
        t0 = time.time()
        cfg_ung = SimConfig(dt=0.1)
        mc_ung = run_monte_carlo_unguided(n_runs, shell, fire, cfg_ung, wind_base, errors)
        t_ung = time.time() - t0
        
        # 3. Run guided Monte Carlo
        t0 = time.time()
        mc_gui = run_monte_carlo_guided(n_runs, shell, fire, cfg, wind_base, errors,
                                         guid_params, target_x, target_z)
        t_gui = time.time() - t0
        
        # Define base64 helper for plots
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor='#0f172a', edgecolor='none', dpi=120)
            plt.close(fig)
            buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')

        # PLOT 1: Side-by-side trajectories
        fig_traj, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.style.use('dark_background')
        fig_traj.patch.set_facecolor('#0f172a')  # Slate 900
        for ax in axes:
            ax.set_facecolor('#1e293b')  # Slate 800
            ax.grid(True, color='#475569', alpha=0.3)
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            ax.title.set_color('#f1f5f9')
            
        ax = axes[0]
        for traj in mc_ung['trajectories'][:15]:
            ax.plot(traj['x']/1000, traj['y']/1000, color='#ef4444', alpha=0.3, lw=0.8)
        ax.plot([], [], color='#ef4444', label='Unguided')
        ax.set_xlabel('Downrange (km)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Unguided Trajectories')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', loc='upper right')
        
        ax = axes[1]
        for traj in mc_gui['trajectories'][:15]:
            ax.plot(traj['x']/1000, traj['y']/1000, color='#22c55e', alpha=0.3, lw=0.8)
        ax.plot([], [], color='#22c55e', label='Guided (CCF)')
        ax.set_xlabel('Downrange (km)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Guided Trajectories')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', loc='upper right')
        
        fig_traj.tight_layout()
        plot1_b64 = fig_to_base64(fig_traj)
        
        # PLOT 2: CEP / Scatter comparison
        fig_cep, ax = plt.subplots(figsize=(8, 6.5))
        fig_cep.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        ax.grid(True, color='#475569', alpha=0.3)
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#f1f5f9')
        
        theta = np.linspace(0, 2*np.pi, 200)
        ax.scatter(mc_ung['range_err'], mc_ung['cross_err'], c='#ef4444', s=20, alpha=0.4, label=f'Unguided (CEP={mc_ung["cep"]:.1f}m)')
        ax.scatter(mc_gui['range_err'], mc_gui['cross_err'], c='#22c55e', s=25, alpha=0.7, label=f'Guided (CEP={mc_gui["cep"]:.1f}m)')
        ax.scatter(0, 0, c='#ffffff', marker='+', s=300, lw=2.5, zorder=10, label='Target')
        
        # CEP circles
        ax.plot(mc_ung['cep']*np.cos(theta), mc_ung['cep']*np.sin(theta), color='#ef4444', ls='--', lw=1.5)
        ax.plot(mc_gui['cep']*np.cos(theta), mc_gui['cep']*np.sin(theta), color='#22c55e', ls='--', lw=2)
        ax.plot(30*np.cos(theta), 30*np.sin(theta), color='#3b82f6', ls=':', lw=1.5, label='30 m requirement')
        
        ax.set_xlabel('Range Error (m)')
        ax.set_ylabel('Cross-range Error (m)')
        ax.set_title(f'CEP Reduction: {mc_ung["cep"]:.1f}m → {mc_gui["cep"]:.1f}m')
        ax.legend(facecolor='#1e293b', edgecolor='#475569', loc='upper right')
        ax.set_aspect('equal')
        fig_cep.tight_layout()
        plot2_b64 = fig_to_base64(fig_cep)

        # PLOT 3: Time Histories (guided run)
        traj = mc_gui['trajectories'][0]
        fig_hist, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig_hist.patch.set_facecolor('#0f172a')
        for row in axes:
            for sub_ax in row:
                sub_ax.set_facecolor('#1e293b')
                sub_ax.grid(True, color='#475569', alpha=0.3)
                sub_ax.tick_params(colors='#94a3b8')
                sub_ax.xaxis.label.set_color('#94a3b8')
                sub_ax.yaxis.label.set_color('#94a3b8')
                sub_ax.title.set_color('#f1f5f9')
                
        # 1. Position
        axes[0, 0].plot(traj['t'], traj['x']/1000, color='#3b82f6', lw=1.2, label='Downrange (km)')
        axes[0, 0].plot(traj['t'], traj['y']/1000, color='#f59e0b', lw=1.2, label='Altitude (km)')
        axes[0, 0].plot(traj['t'], traj['z'], color='#10b981', lw=1.2, label='Cross-range (m)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_title('Guided Position Profile')
        axes[0, 0].legend(facecolor='#1e293b', edgecolor='#475569', loc='upper left')
        
        # 2. Velocity
        speed = np.sqrt(traj['vx']**2 + traj['vy']**2 + traj['vz']**2)
        axes[0, 1].plot(traj['t'], speed, color='#6366f1', lw=1.2, label='Total speed')
        axes[0, 1].plot(traj['t'], traj['vx'], color='#ec4899', lw=0.8, alpha=0.6, label='Vx (downrange)')
        axes[0, 1].plot(traj['t'], traj['vy'], color='#14b8a6', lw=0.8, alpha=0.6, label='Vy (vertical)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity Profiles')
        axes[0, 1].legend(facecolor='#1e293b', edgecolor='#475569', loc='upper right')
        
        # 3. Guidance Commands
        g_total = np.sqrt(traj['guid'][:, 1]**2 + traj['guid'][:, 2]**2) / 9.80665
        axes[1, 0].plot(traj['t'], traj['guid'][:, 1] / 9.80665, color='#3b82f6', lw=1.0, label='Vertical (range)')
        axes[1, 0].plot(traj['t'], traj['guid'][:, 2] / 9.80665, color='#ef4444', lw=1.0, label='Lateral (cross)')
        axes[1, 0].plot(traj['t'], g_total, color='#f1f5f9', lw=1.5, alpha=0.6, label='|Total|')
        axes[1, 0].axhline(y=max_accel_g, color='#f43f5e', ls='--', alpha=0.4)
        axes[1, 0].axhline(y=-max_accel_g, color='#f43f5e', ls='--', alpha=0.4)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Canard Command (g)')
        axes[1, 0].set_title('Guidance Acceleration Commands')
        axes[1, 0].legend(facecolor='#1e293b', edgecolor='#475569', loc='upper right')
        
        # 4. Target distance
        miss_vs_t = np.sqrt((traj['x'] - target_x)**2 + (traj['z'] - target_z)**2)
        axes[1, 1].plot(traj['t'], miss_vs_t/1000, color='#10b981', lw=1.5)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Ground Distance to Target (km)')
        axes[1, 1].set_title('Closing Distance')
        
        fig_hist.tight_layout()
        plot3_b64 = fig_to_base64(fig_hist)

        # Build stats response
        stats = {
            'nominal_range_km': round(target_x / 1000, 2),
            'nominal_cross_m': round(target_z, 1),
            'nominal_time_s': round(r_nom['flight_time'], 1),
            'nominal_apogee_km': round(np.max(r_nom['y']) / 1000, 2),
            
            'ung_cep_m': round(mc_ung['cep'], 2),
            'ung_mean_range_err_m': round(float(np.mean(mc_ung['range_err'])), 2),
            'ung_mean_cross_err_m': round(float(np.mean(mc_ung['cross_err'])), 2),
            'ung_std_range_err_m': round(float(np.std(mc_ung['range_err'])), 2),
            'ung_std_cross_err_m': round(float(np.std(mc_ung['cross_err'])), 2),
            'ung_max_miss_m': round(float(np.max(np.sqrt(mc_ung['range_err']**2 + mc_ung['cross_err']**2))), 2),
            
            'gui_cep_m': round(mc_gui['cep'], 2),
            'gui_mean_range_err_m': round(float(np.mean(mc_gui['range_err'])), 2),
            'gui_mean_cross_err_m': round(float(np.mean(mc_gui['cross_err'])), 2),
            'gui_std_range_err_m': round(float(np.std(mc_gui['range_err'])), 2),
            'gui_std_cross_err_m': round(float(np.std(mc_gui['cross_err'])), 2),
            'gui_max_miss_m': round(float(np.max(np.sqrt(mc_gui['range_err']**2 + mc_gui['cross_err']**2))), 2),
            
            'improvement_x': round(mc_ung['cep'] / max(mc_gui['cep'], 0.1), 1),
            
            'plot_trajectories': plot1_b64,
            'plot_cep': plot2_b64,
            'plot_histories': plot3_b64
        }
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Listen on localhost (port 5000)
    app.run(host='127.0.0.1', port=5000, debug=True)
