# AI-Powered-Vision-Segmentation-Force-Torque-Optimized-Traj-Control-for-7-DOF-Robotic-Articulations
# AI-Powered Vision Segmentation & Force-Torque Optimized Traj Control for 7-DOF Robotic Articulations

An end-to-end pipeline integrating AI-driven vision segmentation with force–torque–optimized model-predictive control to perform precise object articulation on a 7-DOF robotic manipulator.

## Repository Structure

```
.
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── .gitignore           # Files/folders to ignore
├── mujoco_5.py          # Main control script fileciteturn0file0
├── segmentations/       # Auto-generated segmentation masks
│   ├── seg_0000.png     # Example mask at frame 0
│   ├── seg_0001.png     # Example mask at frame 1
│   └── ...
└── scripts/             # Utility launch scripts
    └── run.sh           # Convenience runner
```

## Environment Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ai-visionseg-7dof-forcetraj.git
   cd ai-visionseg-7dof-forcetraj
   ```

2. **Create and activate a Python virtual environment** (Python ≥3.8):

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux/macOS
   venv\\Scripts\\activate.bat  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **MuJoCo Setup**

   * Download and install MuJoCo (e.g., 2.3.4) and place in `$HOME/.mujoco/mujoco210`.
   * Set environment variables:

     ```bash
     export MUJOCO_HOME=$HOME/.mujoco/mujoco210
     export LD_LIBRARY_PATH=$MUJOCO_HOME/bin:$LD_LIBRARY_PATH
     ```

5. **Rendering Drivers**

   * On Ubuntu, install OpenGL/GLFW for on-screen rendering:

     ```bash
     sudo apt-get install libglfw3 libglfw3-dev libglew-dev
     ```

## Running Locally

Use the helper script:

```bash
bash scripts/run.sh --env-name Door --robot Panda --seg-level instance --max-steps 2000
```

Or run directly:

```bash
python mujoco_5.py \
  --env-name Door \
  --robot Panda \
  --camera robot0_eye_in_hand \
  --seg-level element \
  --freq 20 \
  --max-steps 2000
```

* Segmentation masks are saved to `./segmentations/seg_{frame:04d}.png` until handle detection and through the pull sequence.
* Console logs report hinge angle and force/torque norms every 150 steps for analysis.

## Key Components

1. **Data Classes**

   * `ForceTorqueLimits`: Defines soft/hard thresholds for force and torque.
   * `CostConfig`: Weights for orientation, hinge, force, torque, and penalties.
   * `ControlConfig`: Gains, MPC horizon/samples, and backoff parameters.

2. **Vision Segmentation**

   * Uses Robosuite camera obs with segmentation at `element` or `instance` level.
   * Masks are written via:

     ```python
     seg = obs[f"{args.camera}_segmentation_{args.seg_level}"].squeeze(-1)
     imageio.imwrite(f"segmentations/seg_{frame:04d}.png", seg.astype(np.uint8))
     ```

3. **Force–Torque Sensing & Logging**

   * Reads wrench data at each step: force & torque vectors.
   * Logs every 150 steps:

     ```python
     f, t = read_wrench(env)
     print(f"Step {i} | Hinge {obs['hinge_qpos']:+.3f} | |F|{np.linalg.norm(f):.1f}N | |T|{np.linalg.norm(t):.1f}Nm")
     ```

4. **MPC-Based Control Loop**

   * Proportional servoing to align gripper with handle.
   * Samples action sequences over `mpc_horizon`, evaluates via snapshot simulation, selects minimal force norm.
   * Recovery logic backs off on force/torque spikes before retrying.

## Additional Files

* **requirements.txt**

  ```text
  robosuite
  mujoco-py
  numpy
  scipy
  imageio
  ```

* **.gitignore**

  ```gitignore
  __pycache__/
  *.pyc
  venv/
  segmentations/*.png
  ```

* **scripts/run.sh**

  ```bash
  #!/usr/bin/env bash
  set -e
  python mujoco_5.py "$@"
  ```

## Next Steps

* Tune `ControlConfig` gains and MPC parameters for varied door geometries.
* Extend to multi-handle or bi-manipulator scenarios.
* Post-process logs to plot force/torque trajectories using Pandas + Matplotlib.

---

Contributions and issues are welcome—feel free to open a PR or issue on GitHub!
