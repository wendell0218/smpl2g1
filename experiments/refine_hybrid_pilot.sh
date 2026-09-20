cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/gmr
# <modify>+Module execution keeps the repository root on Python's import path.
python -m experiments.refine_hybrid_pilot
