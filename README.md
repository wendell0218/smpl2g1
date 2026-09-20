# smpl2g1

Retarget SMPL or SMPL-X motion to Unitree G1 29-DoF motion and refine existing G1 trajectories for contact quality and controller tracking.

## Installation

```bash
conda env create -f environment.yml
conda activate smpl2g1
```

Download the SMPL-X body models into `models/smplx/`:

```text
models/smplx/SMPLX_NEUTRAL.npz
models/smplx/SMPLX_MALE.npz
models/smplx/SMPLX_FEMALE.npz
```

## Retargeting

Edit the paths in `retarget.sh`, then run:

```bash
source retarget.sh
```

The input `.npz` contains `poses`, `trans`, `betas`, and a frame-rate field. The output contains `qpos_36`, `fps`, `valid`, and `joint_names`.

## Hybrid refinement

The final refinement combines root smoothing, a time-aligned smpl2g1 upper-body prior, G1 position and velocity projection, and contact-aware root-and-leg optimization:

```bash
python refine.py \
  --input umr_g1.npz \
  --smpl-reference smpl2g1_g1.npz \
  --upper-body-strength 0.75 \
  --contact \
  --output outputs/hybrid_contact_g1.npz
```

The contact stage uses the differentiable G1 kinematics and sole geometry included in this repository. It does not require OMG.

The complete 105-sequence experiment can be launched with:

```bash
source experiments/run_full105.sh
```

Its aggregate reference and SONIC measurements are stored in `experiments/hybrid_contact_full105.json`.

## Structure

```text
smpl2g1/
  assets/                 G1 MuJoCo model and meshes
  configs/                retargeting, limits, and kinematics data
  cli.py                  SMPL/SMPL-X retargeting command
  contact_refinement.py   contact-aware trajectory optimization
  constraints.py          joint position and velocity projection
  kinematics.py           differentiable G1 forward kinematics
  motion.py               human-motion loading and preprocessing
  pipeline.py             retargeting pipeline
  refine_cli.py           hybrid refinement command
  refinement.py           smoothing and upper-body fusion
  solver.py               inverse-kinematics solver
experiments/
  run_full105.py          complete-cohort experiment
  run_full105.sh          four-GPU launcher
  hybrid_contact_full105.json
tests/
  test_kinematics.py
  test_pipeline.py
retarget.py / retarget.sh
refine.py / refine.sh
```
