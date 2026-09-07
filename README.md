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
