# smpl2g1

Generate Unitree G1 motion from neutral SMPL or SMPL-X motion.

## Setup

```bash
conda env create -f environment.yml
```

Place the SMPL-X model files in `models/smplx/`:

```text
models/smplx/SMPLX_NEUTRAL.npz
models/smplx/SMPLX_MALE.npz
models/smplx/SMPLX_FEMALE.npz
```

Install [UMR](https://github.com/hanyang9/UMR) at commit `e24fc070030dc0bb0b2c024ecb9f795a3995d725` in its own Conda environment and prepare its G1 correspondence slots.

## Run

Set the input, output, model, UMR environment, and UMR slots paths in `generate.sh`, then run:

```bash
source generate.sh
```

Any setting can also be overridden when sourcing the script. Omitted values keep the defaults in `generate.sh`:

```bash
source generate.sh \
  --input motion.npz \
  --output outputs/final_g1.npz \
  --body-model-dir models \
  --gpu 0
```

The input is an SMPL or SMPL-X `.npz` motion file. The output is the final G1 `.npz` motion file.
