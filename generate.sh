cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/gmr

INPUT_PATH="motion.npz"
OUTPUT_PATH="outputs/final_g1.npz"
BODY_MODEL_DIR="models"
UMR_ROOT="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/UMR"
UMR_ENV="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/umr"
UMR_SLOTS="/cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/UMR/output/correspondence_unitree_g1_smplx_neutral_betas_c4eb7419e9/correspondence_slots_final.npz"
UP_AXIS="y"
GPU="0"

while [ "$#" -gt 0 ]; do
    if [ "$#" -lt 2 ]; then
        echo "Missing value for: $1"
        return 2
    fi
    case "$1" in
        --input) INPUT_PATH="$2"; shift 2 ;;
        --output) OUTPUT_PATH="$2"; shift 2 ;;
        --body-model-dir) BODY_MODEL_DIR="$2"; shift 2 ;;
        --umr-root) UMR_ROOT="$2"; shift 2 ;;
        --umr-env) UMR_ENV="$2"; shift 2 ;;
        --umr-slots) UMR_SLOTS="$2"; shift 2 ;;
        --up-axis) UP_AXIS="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; return 2 ;;
    esac
done

CUDA_VISIBLE_DEVICES="$GPU" python generate.py --input "$INPUT_PATH" --output "$OUTPUT_PATH" --body-model-dir "$BODY_MODEL_DIR" --umr-root "$UMR_ROOT" --umr-env "$UMR_ENV" --umr-slots "$UMR_SLOTS" --up-axis "$UP_AXIS"
