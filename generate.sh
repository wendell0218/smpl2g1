cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate smpl2g1

INPUT_PATH="motion.npz"
OUTPUT_PATH="outputs/final_g1.npz"
BODY_MODEL_DIR="models"
UMR_ROOT="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/UMR"
UMR_ENV="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/umr"
UMR_SLOTS="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9/correspondence_slots_final.npz"

CUDA_VISIBLE_DEVICES=0 python generate.py --input "$INPUT_PATH" --output "$OUTPUT_PATH" --body-model-dir "$BODY_MODEL_DIR" --umr-root "$UMR_ROOT" --umr-env "$UMR_ENV" --umr-slots "$UMR_SLOTS" --up-axis y
