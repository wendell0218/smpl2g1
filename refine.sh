cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/gmr

INPUT_PATH="/mnt/workspace/umr_jerk_eval/full_metrics/g1/UMR/omnihm_0000.npz"
OUTPUT_PATH="outputs/hybrid/omnihm_0000.npz"

python refine.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"
