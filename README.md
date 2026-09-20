# smpl2g1

Convert SMPL or SMPL-X motion to Unitree G1 motion.

## Install

```bash
conda env create -f environment.yml
conda activate smpl2g1
```

Place the SMPL-X model files in `models/smplx/`:

```text
models/smplx/SMPLX_NEUTRAL.npz
models/smplx/SMPLX_MALE.npz
models/smplx/SMPLX_FEMALE.npz
```

## Retarget motion

Set the input, output, and model paths in `retarget.sh`, then run:

```bash
source retarget.sh
```

The input is an SMPL or SMPL-X `.npz` motion file. The output is a G1 `.npz` motion file.

## Refine G1 motion

Set the input, reference, and output paths in `refine.sh`, then run:

```bash
source refine.sh
```

You can also run it directly:

```bash
python refine.py \
  --input input_g1.npz \
  --smpl-reference reference_g1.npz \
  --upper-body-strength 0.75 \
  --contact \
  --output output_g1.npz
```
