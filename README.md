# AI-Powered-Vision-Segmentation-Force-Torque-Optimized-Traj-Control-for-7-DOF-Robotic-Articulations
End-to-end pipeline integrating AI-powered vision segmentation with force-torque optimized model-predictive control for precise object articulation on a 7-DOF manipulator. Features include instance/element-level segmentation, real-time wrench sensing, MPC trajectory sampling, Robosuite/MuJoCo deployment, and comprehensive logging.
**Project Overview**

This repository contains a full-stack implementation of a robotic door manipulation pipeline using Robosuite, leveraging camera-based segmentation, force–torque sensing, and a custom model-predictive controller (MPC). The core script, `mujoco_5.py`, automates handle detection, grasping, and door-pulling actions, recording segmentation masks and printing force/torque trajectories.

**Repository Structure**

```
└── door_manipulation/
    ├── README.md            # This file
    ├── requirements.txt     # Python dependencies
    ├── .gitignore           # Files/folders to ignore
    ├── mujoco_5.py          # Main control scriptfileciteturn0file0
    ├── segmentations/       # Auto‑generated segmentation masks
    │   ├── seg_0000.png      
    │   ├── seg_0001.png      
    │   └── ...               
    └── scripts/             # Utility launch scripts
        └── run.sh            # Convenience runner
```

**Environment Setup & Installation**

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/door_manipulation.git
   cd door_manipulation
   ```

2. **Create a virtual environment** (Python ≥ 3.8 recommended)

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

4. **MuJoCo Installation**

   * Obtain a MuJoCo license and install MuJoCo (e.g., v2.3.4).
   * Set environment variables:

     ```bash
     export MUJOCO_HOME=$HOME/.mujoco/mujoco210
     export LD_LIBRARY_PATH=$MUJOCO_HOME/bin:$LD_LIBRARY_PATH
     ```

5. **Render drivers**

   * Ensure OpenGL/GLFW is installed for on-screen rendering. On Ubuntu:

     ```bash
     sudo apt-get install libglfw3 libglfw3-dev libglew-dev
     ```

**Running Locally**

A convenience script is provided:

```bash
bash scripts/run.sh --env-name Door --robot Panda --seg-level instance --max-steps 2000
```

Or invoke directly:

```bash
python mujoco_5.py \
  --env-name Door \
  --robot Panda \
  --camera robot0_eye_in_hand \
  --seg-level element \
  --freq 20 \
  --max-steps 2000
```

* **Segmentation masks** will be saved under `segmentations/` until the handle is detected and throughout the pull sequence.
* Console logs include hinge angle and force/torque norms every 150 steps.

**Key Components & Design**

1. **Data Classes**

   * `ForceTorqueLimits`: Soft/hard thresholds for force spikes and torque limits.
   * `CostConfig`: Weights for position, orientation, hinge, force, torque, and hard penalties.
   * `ControlConfig`: Gains and parameters for servoing, grasp levels, MPC horizon/samples, noise scale, and recovery back‑off.

2. **Segmentation-Based Handle Detection**

   * The script uses Robosuite’s camera obs with `camera_segmentations` set to `element` or `instance`.
   * Frame-by-frame random motions generate segmentation masks until the handle PX index is detected:

     ```python
     seg = obs[f"{args.camera}_segmentation_{args.seg_level}"].squeeze(-1)
     imageio.imwrite(f"segmentations/seg_{frame:04d}.png", seg.astype(np.uint8))
     ```

3. **Force–Torque Sensing**

   * Wrench data read each step via `read_wrench(env)`, returning force & torque vectors.
   * Magnitudes drive contact detection and spike recovery loops.

4. **Model-Predictive Control (MPC) Loop**

   * On each call to `controller.act(obs)`, the code:

     * Computes distance & orientation mismatch to handle.
     * If out of grasp range → servo with proportional gains.
     * Once near, samples `mpc_samples` action sequences over `mpc_horizon`.
     * Simulates in a snapshot to evaluate force cost, picking the sequence minimizing force norm.
   * Recovery: If force exceeds `f_hard` or spike > `f_soft`, the controller backs off before retrying.

5. **Force & Torque Trajectory Logging**

   * Every 150 steps:

     ```python
     f_val, t_val = read_wrench(env)
     print(f"Step {i:5d} | Hinge {obs['hinge_qpos']:+.3f} | |F|{np.linalg.norm(f_val):.1f}N | |T|{np.linalg.norm(t_val):.1f}Nm")
     ```
   * Use this output to visualize contact dynamics or tune thresholds.

**Sample Segmentations**

Browse `segmentations/` to see grayscale (element) or colored (instance) masks. The handle appears as pixel index `HANDLE_PX` (printed at detection frame).

**Creating Additional Files**

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
  *.log
  ```

* **scripts/run.sh**

  ```bash
  #!/usr/bin/env bash
  set -e
  python mujoco_5.py "$@"
  ```

**Next Steps & Customization**

* Tune `ControlConfig` gains and MPC settings for different door materials or geometries.
* Extend to multi-handle or bi‑manipulator scenarios.
* Record and plot force/torque trajectories using external tools (e.g., Pandas + Matplotlib).

---

Feel free to raise issues or contribute via pull requests!
