cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
conda activate smpl2g1

INPUT_PATH="motion.npz"
OUTPUT_PATH="outputs/motion_g1.npz"
BODY_MODEL_DIR="models"

python retarget.py --input "$INPUT_PATH" --output "$OUTPUT_PATH" --body-model-dir "$BODY_MODEL_DIR"
