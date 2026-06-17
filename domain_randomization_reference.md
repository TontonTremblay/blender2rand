# Domain Randomization for Vision-Based Robot Control — Reference & Working Guide

A literature-grounded reference for an agent generating **domain-randomized (DR) synthetic data** for **vision-based control** policies. It collects the landmark sim-to-real / DR papers, distills the **MolmoBot** recipe (the closest analog to a "generate DR data for vision-based manipulation" pipeline), and turns it into concrete **what-to-randomize checklists** and **strategy choices** you can act on.

> **Core thesis you are operating under (from MolmoBot, Ai2 2026):** for vision-based manipulation, policies benefit *more from diversity across objects, configurations, and viewpoints than from photorealistic rendering*. The bottleneck is no longer collecting real data — it's **designing diverse virtual worlds**. Generate aggressively varied data; let the policy treat the real world as "just another variation."

---

## 1. Concepts & terminology

| Term | Meaning |
|---|---|
| **Reality gap** | Mismatch between simulator and real world (rendering, dynamics, sensor noise) that degrades transfer. |
| **Domain randomization (DR)** | Deliberately randomizing *non-essential* simulator parameters (textures, lighting, pose, dynamics) — often in *non-realistic* ways — so the model learns task-invariant features and sees the real world as one more sample of the training distribution. |
| **Visual / appearance DR** | Randomizing what the camera sees: textures, colors, lighting, materials, backgrounds, camera pose. Primary lever for *perception*. |
| **Dynamics DR** | Randomizing physics: mass, friction, damping, latency, actuator gains. Primary lever for *control / contact*. |
| **Zero-shot transfer** | Deploy a sim-trained policy on a real robot with **no** real-world fine-tuning. This is MolmoBot's and ADR's goal. |
| **Automatic / curriculum DR (ADR)** | Start narrow, expand randomization ranges automatically as the policy improves. |
| **Structured DR (SDR)** | Randomize *within a plausible scene structure / context* instead of fully at random (e.g., cars on roads, not floating). |
| **Sim-to-canonical** | Learn to map randomized images back to a single "canonical" rendering (RCAN), decoupling perception from appearance. |
| **Sim-real co-training** | Mix synthetic + a little real data during training to anchor the distribution. |

**Visual vs. dynamics, practical rule of thumb:** if your failures are *perception* (mis-localizing, missing the object under new lighting/background) → push **visual + camera** DR. If failures are *contact/control* (slipping, overshoot, instability) → push **dynamics** DR. Vision-based control needs both, but for an **RGB policy** like MolmoBot, visual + camera diversity is usually the dominant factor.

---

## 2. Landmark papers (chronological)

Each entry: what it did → **takeaway for your data generation**.

### Tobin et al., 2017 — *Domain Randomization for Transferring DNNs from Simulation to the Real World*
The paper that named DR. Trained an object-localization network purely on low-fidelity simulated images with randomized textures, lighting, camera position, and distractors; transferred to a real robot. Showed that with **enough variability, photorealism is unnecessary**.
→ Establishes the foundational move: randomize textures/lighting/pose/distractors; non-realistic is fine.
arXiv: <https://arxiv.org/abs/1703.06907>

### Tremblay et al. (NVIDIA), 2018 — *Training Deep Networks with Synthetic Data: Bridging the Reality Gap by DR*
DR for **object detection** (cars, KITTI). Introduced **"flying distractors"** — random geometric objects floating in the scene — to force the network to learn the object of interest. Found DR-only training is competitive, and **DR + a little real fine-tuning beats real data alone**.
→ Add random distractor geometry; plan for optional real-data fine-tuning on top of synthetic.
arXiv: <https://arxiv.org/abs/1804.06516>

### Peng et al., 2018 — *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*
Randomized the **dynamics** (mass, friction, damping, etc.) of a simulated arm during RL training; an object-pushing policy transferred zero-shot to a real robot and was robust to large calibration error. Memory-based policies *adapt online* to the real dynamics.
→ The canonical reference for **dynamics DR**; pair with a recurrent/memory policy so it can infer real parameters at test time.
arXiv: <https://arxiv.org/abs/1710.06537>

### OpenAI, 2018 — *Learning Dexterous In-Hand Manipulation*
Shadow Hand reorientation trained entirely in sim with randomized physics **and** vision (for pose estimation), transferred to hardware. Early proof that **combined visual + dynamics DR** supports a real vision-based control loop on a hard contact-rich task.
→ Precedent for randomizing perception and physics together for vision-based control.
arXiv: <https://arxiv.org/abs/1808.00177>

### Prakash et al. (NVIDIA), 2019 — *Structured Domain Randomization (SDR)*
Instead of placing objects fully at random, SDR samples scenes that respect **context/structure** (e.g., cars on the road surface, in lanes). Improved detection over unstructured DR by keeping the randomness *plausible* where structure matters.
→ Don't randomize blindly: preserve scene structure that is genuinely informative for the task (support surfaces, reachable workspace, gravity-consistent placement).
arXiv: <https://arxiv.org/abs/1810.10093>

### James et al., 2019 — *RCAN: Sim-to-Real via Sim-to-Sim (Randomized-to-Canonical Adaptation Networks)*
Learns an image-conditioned generator that maps **heavily randomized** renders → a single **canonical** rendering; the grasping policy then operates on canonical images. Decouples "see through the randomization" from "act," improving data efficiency.
→ Alternative to brute-force DR: generate randomized + paired canonical images so a model can learn the canonicalization mapping.
arXiv: <https://arxiv.org/abs/1812.07252>

### Chebotar et al. (NVIDIA), 2019 — *Closing the Sim-to-Real Loop (SimOpt)*
Rather than hand-tuning randomization ranges, **adapt the randomization distribution** using a few real-world rollouts interleaved with training, matching sim and real behavior. Transferred on swing-peg-in-hole and drawer-opening.
→ If you have *any* real rollouts, use them to tune which parameters/ranges to randomize instead of guessing.
arXiv: <https://arxiv.org/abs/1810.05687>

### OpenAI, 2019 — *Solving Rubik's Cube with a Robot Hand (Automatic Domain Randomization, ADR)*
**ADR** automatically generates an ever-harder distribution of randomized environments: start from a single non-randomized env, widen each parameter's range whenever the policy clears a performance threshold. Control policies *and* vision state estimators trained this way showed strong sim2real and emergent meta-learning. ADR was reported ~3× faster than well-tuned manual DR, and far better than poorly-chosen fixed ranges.
→ Prefer a **curriculum**: start narrow, auto-expand ranges by performance. Avoid the two failure modes — too little randomization (never transfers) and too much from the start (never converges).
arXiv: <https://arxiv.org/abs/1910.07113>

### RoboTwin 2.0, 2025 — *Scalable Data Generator + Benchmark with Strong DR for Bimanual Manipulation*
A modern DR **data generator**. Randomizes along five concrete axes: (1) **cluttered distractor objects**, (2) **background textures**, (3) **lighting**, (4) **tabletop heights**, (5) **diverse language instructions**. Uses a large generated texture library (LLM-prompted descriptions → Stable Diffusion → human-filtered ~11k textures) and collision-aware distractor placement that **excludes** distractors too similar to task objects.
→ A directly reusable axis list for tabletop manipulation, plus two strong practices: **big procedural texture library** and **distractor disambiguation**.
arXiv: <https://arxiv.org/abs/2506.18088>

### MolmoBot (Ai2), 2026 — *Training Robot Manipulation Entirely in Simulation* ⭐ anchor
Open suite trained **entirely on simulation data** (no real data, no fine-tuning), achieving **zero-shot** sim-to-real on two platforms (Franka FR3 tabletop, Rainbow Robotics RB-Y1 mobile manipulator), competitive with π0 / π0.5 which use large-scale real data. Built on **MuJoCo + aggressive DR + procedural environment generation**; **RGB-only** observations. Ships **MolmoBot-Engine** (the open data-gen pipeline: environment sampling, DR, expert trajectory generation), **MolmoBot-Data** (~1.7M expert trajectories, 11k+ objects, 94k+ procedurally generated environments, 8 task types), and assets sourced from iTHOR + Objaverse.
→ This is the template for your agent. See the detailed recipe in §3. Key lesson restated: **diversity (objects, placements, viewpoints, lighting, textures, dynamics) beats photorealism**, and **fully randomized cameras** are central to the result.
Blog: <https://allenai.org/blog/molmobot-robot-manipulation> · Paper: <https://arxiv.org/abs/2603.16861> · Engine code: <https://github.com/allenai/molmospaces> · Models code: <https://github.com/allenai/MolmoBot>

### Surveys (for breadth & taxonomy)
- **Zhao, Queralta & Westerlund, 2020** — *Sim-to-Real Transfer in Deep RL for Robotics: a Survey.* Frames techniques over the MDP (state/action/transition/reward); covers DR, domain adaptation, imitation, meta-learning, knowledge distillation. <https://arxiv.org/abs/2009.13303>
- **"The Reality Gap in Robotics: Challenges, Solutions, and Best Practices," 2025** — recent comprehensive taxonomy distinguishing methods that *reduce* vs *overcome* the reality gap (DR, real-to-sim, state/action abstraction, sim-real co-training); discusses open problems in rendering/sensor modeling and contact/deformation simulation. <https://arxiv.org/abs/2510.20808>

---

## 3. The MolmoBot recipe in detail (closest analog to your task)

MolmoBot-Engine performs DR across **three axes**, plus image augmentation at training time:

1. **Environment randomization** — after placing objects, randomize *all* visual and physical parameters of the scene: object textures/colors, materials, lighting, backgrounds, and physical dynamics. Procedurally generated environments vary objects, placements, lighting, textures, and dynamics across runs.
2. **Action randomization** — variation in the expert trajectories / action generation (the engine plans, randomizes, and **iteratively replans until a successful trajectory is found**).
3. **Camera perturbation** — **fully randomized cameras** (viewpoint/pose). This is repeatedly called out as central to zero-shot transfer; do not fix the camera.
4. **Image augmentation (training-time)** — standard augmentations applied on top during policy training, complementing scene-level DR.

Supporting practices worth copying:
- **Procedural environment generation** with broad object coverage (MolmoBot sources rigid assets from **Objaverse** and **iTHOR**; 11k+ objects).
- **Semantic filtering for task roles** — e.g., for a pick-and-place *receptacle*, filter candidate objects by metadata so the target is semantically valid; ensure **watertight collider meshes** for physically plausible contact.
- **RGB-only** policy observations even though the pipeline *can* emit depth + privileged simulator metadata — keeping observations RGB makes transfer results stronger and the policy cheaper to deploy. (You can still log depth/segmentation/privileged state for auxiliary losses or debugging.)
- **Expert trajectories via plan-and-replan** rather than teleoperation — scalable supervision without humans.

---

## 4. What to randomize — concrete checklists

Treat these as the parameter space your generator samples from. Start from a **narrow nominal range and widen** (ADR-style). Mark each parameter as *visual*, *geometry*, or *dynamics*.

### 4.1 Visual / appearance (perception)
- [ ] **Object textures & colors** — incl. non-realistic textures; large procedural texture library (cf. RoboTwin's ~11k).
- [ ] **Materials** — albedo, specular/roughness, metallic, reflectivity.
- [ ] **Backgrounds / table & wall surfaces** — swap from a big texture/image bank; random gradients and patterns.
- [ ] **Lighting** — number of lights, position/direction, intensity, color temperature, ambient level, shadows on/off.
- [ ] **Distractors** — random geometric "flying distractors" (Tremblay) and/or realistic clutter objects (RoboTwin). **Exclude distractors visually/semantically too similar to the task object** to avoid label ambiguity.
- [ ] **Post-process / sensor noise** — Gaussian/shot noise, blur, exposure/gamma jitter, color jitter, JPEG-like artifacts, occasional occluders.

### 4.2 Camera (often the highest-leverage axis for vision-based control)
- [ ] **Extrinsics**: camera position, orientation/look-at, height. *Fully randomize*, per MolmoBot. For mobile manipulation, vary base-relative pose.
- [ ] **Intrinsics**: focal length / FOV, principal point, aspect, mild lens distortion.
- [ ] **Resolution / crop / aspect** jitter; image-plane augmentation at train time.
- [ ] Multi-frame: keep temporal consistency within a trajectory but randomize across trajectories.

### 4.3 Object & scene geometry
- [ ] **Object instances & categories** — broad asset library (Objaverse / iTHOR / your own); novel held-out instances for eval.
- [ ] **Object pose & placement** — position, orientation, scale; respect **support surfaces and reachable workspace** (Structured DR — don't float objects unless using flying distractors deliberately).
- [ ] **Scene layout** — tabletop height, surface size, receptacle locations, number of objects, clutter density.
- [ ] **Articulation params** (drawers/doors/cabinets) — joint limits, hinge/handle placement, opening direction.
- [ ] **Collider quality** — watertight meshes; collision-aware placement so scenes are physically valid.

### 4.4 Physics / dynamics (control & contact)
- [ ] **Mass & inertia** of objects (and payload).
- [ ] **Friction** — static/dynamic, tangential & torsional; surface-pair friction.
- [ ] **Restitution / damping / stiffness**; soft-contact parameters.
- [ ] **Actuator model** — joint gains (P/D), torque/force limits, gear/transmission, motor noise.
- [ ] **Latency & control timing** — observation/action delay, control frequency jitter (critical for closed-loop transfer).
- [ ] **Initial-state distribution** — randomized start configs (cf. Peng: random initial object configurations).
- [ ] **External disturbances** — small pushes/perturbations during execution for robustness.

### 4.5 Task / language (for instruction-conditioned policies)
- [ ] **Diverse language instructions / paraphrases** per task (RoboTwin) and/or **point-based commands** (MolmoBot supports "pick"/"place"/"close").

### 4.6 Free auxiliary signals to log (even if the deployed policy is RGB-only)
Depth, surface normals, instance/semantic segmentation, object poses, contact forces, and privileged simulator state — useful for auxiliary losses, filtering bad episodes, or analysis. MolmoBot keeps the *policy input* RGB but the *pipeline* can emit these.

---

## 5. Strategy choices (how to randomize, not just what)

| Strategy | When to use | Source |
|---|---|---|
| **Uniform / fixed-range DR** | Baseline; simple and strong if ranges are well chosen. | Tobin '17, Tremblay '18 |
| **Dynamics randomization + memory policy** | Contact-rich control; want online adaptation to real physics. | Peng '18 |
| **Automatic / curriculum DR (ADR)** | You can measure policy performance and want to avoid hand-tuning ranges; maximize hardness without breaking convergence. | OpenAI '19 |
| **Structured / context-aware DR** | Scene structure is informative (support surfaces, lanes, reachable space). Avoids implausible scenes. | Prakash '19 |
| **Adaptive DR with real rollouts (SimOpt)** | You have a few real-world rollouts to calibrate the randomization distribution. | Chebotar '19 |
| **Sim-to-canonical (RCAN)** | Want to decouple perception from appearance; data-efficiency on grasping. | James '19 |
| **Aggressive DR + procedural gen, RGB-only, fully random cameras** | Goal is *zero-shot* sim-only training at scale for vision-based manipulation. | MolmoBot '26 |
| **Sim-real co-training / DR + real fine-tune** | A modest amount of real data is available; anchor the synthetic distribution. | Tremblay '18, surveys |

**Default recommendation for a sim-only RGB manipulation pipeline:** procedural environment generation + **aggressive visual & camera DR** + dynamics DR for contact, supervised by **plan-and-replan expert trajectories**, with an **ADR-style curriculum** widening ranges over training, and **structured placement** (respect support surfaces / workspace). This is essentially the MolmoBot configuration.

---

## 6. Practical tips & pitfalls for the data-gen agent

1. **Diversity > photorealism.** Spend compute on more objects/placements/viewpoints/textures, not on a prettier renderer. (MolmoBot's central finding.)
2. **Randomize the camera — don't fix it.** Fully randomized camera pose is repeatedly tied to zero-shot transfer. A fixed camera is a common silent cause of poor transfer.
3. **Calibrate the *amount* of randomization.** Too little → never transfers regardless of training length; too much from the start → may never converge. Curriculum/ADR resolves this; if doing fixed ranges, tune them (ideally with a few real rollouts, SimOpt-style).
4. **Keep task-relevant invariants intact.** Randomize *non-essential* factors. Don't randomize away the cues the task actually depends on (e.g., a color that defines the target), and respect physical structure (gravity, support, reachability).
5. **Disambiguate distractors.** Exclude distractors that are visually/semantically too similar to task objects — otherwise you inject label noise. (RoboTwin practice.)
6. **Guarantee physical validity.** Watertight colliders + collision-aware placement; reject/replan episodes that fail (MolmoBot iteratively replans until a successful expert trajectory exists). Filter out degenerate or failed episodes before they enter the dataset.
7. **Label/annotation fidelity.** Auto-generated labels (poses, masks, success flags, grasp points) must stay correct under every randomization. Add automated sanity checks.
8. **Match perturbation channels to failure modes.** Perception failures → visual/camera DR; contact/control failures → dynamics/latency DR.
9. **Evaluate under held-out perturbations.** Test robustness with eval-time shifts *not seen in training* — alternate renderer, new lighting, new camera, novel objects/instances (MolmoBot's evaluation protocol). Hold out objects and environments from training.
10. **Decide RGB vs depth deliberately.** RGB-only can transfer better and is cheaper to deploy; still log depth/segmentation/privileged state for auxiliary losses and debugging even if the policy never sees them.
11. **Make the pipeline reproducible.** Seed every randomization; record the sampled parameters per episode so datasets are auditable and regenerable (MolmoSpaces is built around reproducible trajectory generation).
12. **Plan for an optional real-data top-up.** Pure DR is strong; DR + a little real fine-tuning often beats either alone if/when real data becomes available.

---

## 7. Quick-reference table

| Year | Paper | Core contribution | Link |
|---|---|---|---|
| 2017 | Tobin et al. — Domain Randomization | Named DR; visual DR for localization; photorealism unnecessary | <https://arxiv.org/abs/1703.06907> |
| 2018 | Peng et al. — Dynamics Randomization | Randomize physics; memory policy adapts online | <https://arxiv.org/abs/1710.06537> |
| 2018 | Tremblay et al. — Synthetic Data DR | DR for detection; flying distractors; DR+real > real | <https://arxiv.org/abs/1804.06516> |
| 2018 | OpenAI — Dexterous In-Hand Manipulation | Combined visual+dynamics DR, contact-rich, real transfer | <https://arxiv.org/abs/1808.00177> |
| 2019 | Prakash et al. — Structured DR | Context-aware randomization within plausible structure | <https://arxiv.org/abs/1810.10093> |
| 2019 | James et al. — RCAN | Randomized→canonical image mapping; data-efficient grasping | <https://arxiv.org/abs/1812.07252> |
| 2019 | Chebotar et al. — SimOpt | Adapt DR distribution from real rollouts | <https://arxiv.org/abs/1810.05687> |
| 2019 | OpenAI — Rubik's Cube / ADR | Automatic curriculum DR; vision + control; meta-learning | <https://arxiv.org/abs/1910.07113> |
| 2020 | Zhao et al. — Survey | MDP-framed taxonomy of sim2real methods | <https://arxiv.org/abs/2009.13303> |
| 2025 | RoboTwin 2.0 | Scalable DR data generator; 5 axes; big texture library | <https://arxiv.org/abs/2506.18088> |
| 2025 | Reality Gap survey | Reduce-vs-overcome taxonomy; best practices | <https://arxiv.org/abs/2510.20808> |
| 2026 | **MolmoBot (Ai2)** ⭐ | **Sim-only RGB manipulation, zero-shot transfer; open DR engine** | <https://arxiv.org/abs/2603.16861> |

---

### Reusable open infrastructure
- **MolmoBot-Engine / MolmoSpaces** — open procedural data-gen + DR + expert-trajectory pipeline (MuJoCo). Closest off-the-shelf starting point: <https://github.com/allenai/molmospaces>
- **RoboTwin 2.0** — open bimanual DR data generator + benchmark: <https://arxiv.org/abs/2506.18088>
- **Asset sources** — Objaverse, iTHOR (used by MolmoBot for broad object coverage).
- **Simulators commonly used for DR** — MuJoCo (MolmoBot, RoboTwin), NVIDIA Isaac Sim, Unreal Engine (Tremblay/SDR), Blender/BlenderProc for synthetic detection data.

*Notes: arXiv identifiers and figures above are drawn from the cited papers and the MolmoBot release materials. Always confirm exact parameter ranges and asset licenses against each project's own repo/paper before building on them.*
