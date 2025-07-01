# AI-Powered Vision Segmentation & Force-Torque Optimized Trajectory Control for 7-DOF Robotic Articulations

An end-to-end framework that enables a 7-DOF robotic arm to accurately grasp and articulate objects by combining:

* **Deep AI Vision Segmentation** to detect and isolate the target handle in real time.
* **Force–Torque Sensing** to monitor contact forces and ensure safe interactions.
* **Model-Predictive Control (MPC)** to compute trajectories that minimize force/torque spikes and maintain smooth motion.

---

## Concept Overview

Modern robotic manipulation demands both perceptual intelligence and precise control. This pipeline integrates three core principles:

1. **Vision-Based Segmentation**
   A camera mounted on the manipulator captures RGB and segmentation observations. We leverage Robosuite’s built-in segmentation (either element‑ or instance‑level) so that each distinct part of the scene, most importantly the door handle, can be extracted as a binary mask. By processing that mask in each frame, the system pinpoints the handle’s pixel coordinates and converts them into a 3D position for grasp planning.

2. **Force–Torque Feedback**
   A six‑axis force–torque sensor at the wrist continuously measures the contact wrench (force vector **F** and torque vector **T**). Reading these values each control step allows the robot to detect collisions or excessive strain on the gripper, triggering recovery behaviors if thresholds are exceeded. This sensory feedback ensures both the safety of the mechanism and the integrity of the object.

3. **Model-Predictive Control (MPC)**
   Instead of following a fixed open‑loop path, MPC generates multiple candidate action sequences over a short horizon (e.g., 15 steps). Each sequence is simulated in a copy of the environment to forecast the resulting forces and hinge motions. By evaluating a cost function that heavily penalizes large force/torque norms, the controller selects the safest and most stable trajectory at runtime. This leads to smoother pulls with fewer jerks or spikes.

Together, these components enable the robot to autonomously detect, grasp, and pull a door handle with high precision while minimizing the risk of damage or jamming.

---

## Environment Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/dineshsairallapalli/ai-visionseg-7dof-forcetraj.git
   cd ai-visionseg-7dof-forcetraj
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\\Scripts\\activate.bat  # Windows
   ```

3. **Install Python dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **MuJoCo Installation**

   * Download MuJoCo (e.g., v2.3.4) and extract to `$HOME/.mujoco/mujoco210`.
   * Add to your environment:

   ```bash
   export MUJOCO_HOME=$HOME/.mujoco/mujoco210
   export LD_LIBRARY_PATH=$MUJOCO_HOME/bin:$LD_LIBRARY_PATH
   ```

5. **Rendering Drivers**

   * On Ubuntu:

   ```bash
   sudo apt-get install libglfw3 libglfw3-dev libglew-dev
   ```

---

* **Segmentation outputs** → `./segmentations/seg_{frame:04d}.png`
* **Logging** → hinge angle and “|F|” / “|T|” norms printed every 150 steps.

---

## Key Components & Code Highlights

### Data Classes

* **ForceTorqueLimits**: Sets soft/hard thresholds for magnitude of **F** and **T**.
* **CostConfig**: Balances positional, orientational, hinge, force, torque, and hard penalties.
* **ControlConfig**: Defines servo gains, MPC horizon/samples, noise levels, and backoff strategy.

### Vision Segmentation

```python
seg = obs[f"{args.camera}_segmentation_{args.seg_level}"].squeeze(-1)
imageio.imwrite(f"segmentations/seg_{frame:04d}.png", seg.astype(np.uint8))
```

Extracts a 2D mask per frame, enabling 2D→3D coordinate conversion for handle localization.

### Force–Torque Sensing & Logging

```python
f, t = read_wrench(env)
print(
    f"Step {i:4d} | Hinge {obs['hinge_qpos']:+.3f} |"
    f" |F|{np.linalg.norm(f):.1f}N |T|{np.linalg.norm(t):.1f}Nm"
)
```

Monitors wrench data to detect contact or spikes, ensuring safe operations.

### MPC-Based Control Loop

* **Servo Phase**: Proportional commands bring gripper within grasping range.
* **Sampling Phase**: Generate `mpc_samples` sequences across `mpc_horizon`.
* **Evaluation**: Simulate each in a snapshot and compute cost (force/torque penalty).
* **Execution**: Apply the sequence with minimal predicted force norm.
* **Recovery**: If observed force > hard threshold, back off along a safe retreat path.

---

## Next Steps & Extensions

* **Parameter Tuning**: Adjust `ControlConfig` gains and MPC settings for different door/material properties.
* **Multi-Object Handling**: Extend segmentation to detect multiple handles and plan multi-grip sequences.
* **Data Analysis**: Use Pandas + Matplotlib to post-process logs and visualize force/torque trajectories.

---

Contributions, issues, and feature requests are welcome! Feel free to open a PR or issue on GitHub.
