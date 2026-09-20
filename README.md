# smpl2g1

Retarget SMPL or SMPL-X motion to Unitree G1 29-DoF motion.

## Installation

```bash
conda env create -f environment.yml
conda activate smpl2g1
```

Download the SMPL-X body models and place them in `models/smplx/`:

```text
models/smplx/SMPLX_NEUTRAL.npz
models/smplx/SMPLX_MALE.npz
models/smplx/SMPLX_FEMALE.npz
```

## Quick start

Edit the paths in `retarget.sh`, then run:

```bash
source retarget.sh
```

The input is an SMPL-style `.npz` file containing `poses`, `trans`, `betas`, and a frame-rate field. The output contains `qpos_36`, `fps`, `valid`, and `joint_names`.

<!-- <modify>+Document the optional UMR/smpl2g1 hybrid refinement. -->
## Hybrid refinement

`refine.py` accepts an existing 30 FPS G1 trajectory, including UMR output. The validated default applies a local convex three-frame filter to the root pose, then projects joint positions and adjacent-frame velocities into the canonical G1 safety bounds:

```bash
python refine.py \
  --input umr_g1.npz \
  --output outputs/hybrid_g1.npz
```

Edit the paths in `refine.sh` and run it with `source refine.sh` for the repository's configured environment.

The complete 105-sequence validation is recorded in `experiments/hybrid_rootpose_full105.json`. Relative to the feasibility-only hybrid, root smoothing reduces mean jerk from 103.24 to 73.33 m/s³ while retaining 100% Joint Feasibility; all 105 SONIC rollouts were rerun.

<!-- <modify>+Document the optional controller-trackable upper-body prior. -->
When a time-aligned smpl2g1 trajectory is available, its waist and arm joints can be used as a soft tracking prior while the UMR root and legs remain unchanged:

```bash
python refine.py \
  --input umr_g1.npz \
  --smpl-reference smpl2g1_g1.npz \
  --upper-body-strength 0.75 \
  --output outputs/hybrid_upper_g1.npz
```

The fixed strength was selected on the motion-stratified 12-sequence pilot rather than per sequence. The feasibility projection is applied after fusion.

<!-- <modify>+Record the complete-cohort outcome of the fixed upper-body prior. -->
The complete 105-sequence result is archived in `experiments/hybrid_upper_full105.json`. Relative to the root-pose hybrid, Tracking Score improves from 0.2731 to 0.3255, MPJPE from 32.35 to 29.05 mm, jerk from 73.33 to 69.43 m/s³, and MSNR from 18.72 to 21.78 dB. Joint Feasibility remains 100%, and all 105 SONIC rollouts complete without a fall.
