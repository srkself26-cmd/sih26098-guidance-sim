# SIH26098 – Low-Cost Precision Guidance Kit (155 mm Artillery Shell) Simulation

[![Deploy to Render](https://render.com/images/deploy-to-render.svg)](https://render.com/deploy?repo=https://github.com/srkself26-cmd/sih26098-guidance-sim)

Welcome to the **Precision Guidance Kit (PGK)** simulation project. This software models the flight of a standard 155 mm artillery shell and demonstrates how an add-on smart fuze kit—equipped with steerable steering fins (**canards**)—takes a shell that would normally drift hundreds of meters off-course due to wind and firing errors, and steers it to land within just a few meters of the target.

---

## 📖 Table of Contents
1. [Project Focus & Scope (What It Models vs. What It Doesn't)](#-project-focus--scope)
2. [Layperson Glossary of Technical Terms](#-glossary-of-technical-terms)
3. [Input Parameters Explained](#-input-parameters-explained)
4. [How Changing Parameters Affects the Outcome](#-parameter-effects-sensitivity)
5. [How to Run the Simulation](#-how-to-run-the-simulation)

---

## 🎯 Project Focus & Scope

Artillery shells are unguided rockets or projectiles. When fired over long distances (20–30 km), tiny variations in wind, air density, shell weight, or gun barrel elevation compound into a large landing error at the target. This project simulates a **Course Correction Fuze (CCF)**, which replaces the standard nose fuze of a 155 mm shell with a GPS-guided fin package to steer the shell mid-flight.

### What the Simulation Focuses On (In-Scope):
*   **3-DOF Flight Physics**: Simulates the 3 Degrees of Freedom (downrange, altitude, and cross-range movement) of the shell using real-world forces: gravity, atmospheric drag (air resistance), and altitude-varying wind.
*   **Stochastic (Random) Errors**: Simulates a "Monte Carlo" study (firing multiple rounds under slightly different random conditions) to find the statistical landing scatter.
*   **ZEM (Zero Effort Miss) Guidance**: Simulates the onboard computer calculating where the shell *would* land if it stopped steering right now, and using that "miss distance" to issue corrections.
*   **Canard Actuator Limits**: Simulates real-world physical limits, ensuring the steering fins can only push the shell by a realistic amount (maximum of 5-10 $g$'s of force).
*   **Interactive Visualizations**: Includes a web-based dashboard to dynamically change parameters and instantly see how the trajectories and landing points change.

### What the Simulation Does NOT Focus On (Out-of-Scope):
*   **6-DOF Rigid-Body Dynamics**: Does not model the high-frequency spin rate, pitching, or yawing angles of the shell. (A simplified 3-DOF model is used for speed and clarity).
*   **Fuze Arming, Detonation, or Explosives**: Does not simulate any military payload details, explosives, or lethal radius calculations.
*   **Hardware-in-the-Loop (HIL)**: Does not interface with physical hardware, microcontrollers, or real canard servos.
*   **GPS Jamming/Electronic Warfare**: Models GPS positioning errors as simple random noise, but does not model complex radio-frequency interference or jamming.

---

## 🗂 Glossary of Technical Terms

If you do not have an engineering background, here are the core terms explained in plain English:

| Term | Simple Definition |
| :--- | :--- |
| **3-DOF (3 Degrees of Freedom)** | A physics model that tracks the shell's position in three axes: forward-backward (downrange), up-down (altitude), and left-right (cross-range). It ignores rotational angles (roll, pitch, yaw). |
| **CEP (Circular Error Probable)** | A standard military measurement of accuracy. It is the radius of a circle centered on the target that contains exactly 50% of the landing points. A smaller CEP means a more accurate gun. |
| **Canards** | Small, steerable aerodynamic wings (fins) mounted near the nose of the shell. By rotating them, they generate lift forces to push the shell left, right, up, or down. |
| **Monte Carlo Simulation** | A method of understanding uncertainty by running the same simulation dozens or hundreds of times, each time randomly perturbing inputs (like wind speed or gun angle) to see the spread of results. |
| **Muzzle Velocity** | The speed at which the shell exits the tip of the gun barrel. For a 155 mm shell, this is extremely fast (typically around 800 to 900 meters per second, or roughly Mach 2.5). |
| **Proportional Navigation** | A guidance rule where the steering commands are proportional to the rate at which the line of sight to the target is rotating. It ensures a smooth intercept path. |
| **Zero Effort Miss (ZEM)** | The distance by which the shell will miss the target if no further steering commands are applied from this moment onwards. |
| **RK4 (4th-order Runge-Kutta)** | A highly accurate mathematical formula used to solve the equations of motion step-by-step in time. It predicts where the shell moves next. |
| **Apogee** | The highest point in the shell's trajectory arc (peak altitude). |

---

## 🎛 Input Parameters Explained

The simulation contains several adjustable parameters divided into four categories:

### 1. Gun & Projectile Parameters
*   **Shell Mass ($kg$)**: The weight of the steel shell. Standard 155 mm shells (like the M107 or M795) weigh around 43 to 45 kg.
*   **Muzzle Velocity ($m/s$)**: The initial speed of the shell leaving the gun. Higher velocities push the shell further.
*   **Launch Elevation (Degrees)**: The angle of the gun barrel relative to the ground. 45 degrees theoretically gives the maximum range.

### 2. Environment & Wind Parameters
*   **Wind X (Head/Tail Wind) ($m/s$)**: Wind blowing along the gun's firing direction. Positive values push the shell from behind (increasing range); negative values blow against it (reducing range).
*   **Wind Z (Crosswind) ($m/s$)**: Wind blowing sideways. Positive values push the shell to the right; negative values push it to the left.
*   *Note: Wind increases in speed at higher altitudes, which the simulation automatically calculates.*

### 3. Canard Guidance (CCF) Parameters
*   **Max Canard Authority ($g$'s)**: The maximum lateral force the small steering fins can generate, measured in "$g$-forces" (multiples of gravity). A value of $5.0$ means the fins can pull a sideways acceleration of $5.0 \times 9.81$ m/s².
*   **GNSS Lock Acquisition ($s$)**: The delay (in seconds) after firing before the shell acquires GPS signals. The guidance system cannot steer until it has a GPS lock.

### 4. Stochastic Error Sources (1-Sigma $\sigma$)
*Artillery fire is subject to natural variations. The "1-sigma" value represents the standard deviation of these variations (68% of shots fall within this range).*
*   **Muzzle Velocity Error ($m/s$)**: The random variation in speed from shot to shot due to powder temperature and barrel wear.
*   **Elevation Aim Error ($mrad$)**: The tiny random variation in the gun barrel's vertical aim angle, measured in milliradians (1 mrad is 0.057 degrees).
*   **Wind Uncertainty ($m/s$)**: The random noise and gusts added to the average wind profile for each shot.

---

## 📈 Parameter Effects (Sensitivity)

Here is how changing the parameters shifts the final landing points:

### If you INCREASE:
| Parameter | Effect on Unguided Shell | Effect on Guided Shell |
| :--- | :--- | :--- |
| **Shell Mass** | **Shorter range.** A heavier shell slows down faster due to gravity, landing short of the target. | **No change in landing point**, provided the error is within the canard authority limits. |
| **Muzzle Velocity** | **Longer range.** The shell carries more kinetic energy, flies higher, and lands further. | **No change.** The guidance kit steers the shell to land on the target. |
| **Launch Elevation** | **Alters range.** For angles below 45°, increasing it extends range. For angles above 45°, it increases height but shortens ground range. | **No change.** The guidance kit corrects the trajectory arc. |
| **Crosswind (Wind Z)** | **Heavy lateral drift.** The shell is blown far to the side (up to 300+ meters off-course). | **No change.** The canards steer against the wind to hit the target. |
| **Max Canard Authority** | *No effect.* | **Higher wind-resistance.** The shell can correct for much stronger crosswinds and larger firing errors. |
| **GNSS Lock Delay** | *No effect.* | **Reduced correction capacity.** If the GPS lock takes too long (e.g., 15+ seconds), the shell spends too much time drifting, leaving it with too little altitude/time to steer back. |
| **Stochastic Errors** | **Wider scatter (Huge CEP).** The unguided landing points spread out into a massive ellipse (up to 400+ meters wide). | **Tighter group (CEP stays small).** The guidance corrects each individual shot, maintaining a CEP under 4 meters unless the error exceeds fin steering limits. |

---

## 🚀 How to Run the Simulation

You can run the simulation in two ways:

### Method 1: The Interactive Web Dashboard (Recommended)
This launches a beautiful, modern browser dashboard where you can adjust sliders and see the plots update instantly.

1.  **Activate the virtual environment**:
    ```bash
    cd /home/rishikanth/.gemini/antigravity/scratch/sih26098_guidance_sim
    source venv/bin/activate
    ```
2.  **Start the web server**:
    ```bash
    python3 web_app.py
    ```
3.  **Open in your browser**:
    Go to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

### Method 2: The Command Line Script
This runs a batch simulation and saves static image plots directly in the project directory.

1.  **Activate the environment**:
    ```bash
    cd /home/rishikanth/.gemini/antigravity/scratch/sih26098_guidance_sim
    source venv/bin/activate
    ```
2.  **Run the script**:
    ```bash
    python3 artillery_sim.py
    ```
3.  **Inspect output images**:
    *   `plot1_trajectories.png` (Trajectory profiles)
    *   `plot2_scatter.png` (Landing points scatter & CEP circles)
    *   `plot3_cep_comparison.png` (CEP comparison)
    *   `plot4_time_histories.png` (Onboard sensor data)
