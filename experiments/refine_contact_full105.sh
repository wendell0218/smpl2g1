cd /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/code/smpl2g1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /cpfs01/thscc/sharestorage/AIGC-06-01/Hithink_VideoGen_Group/user_workspace/buwendong/user/envs/OMG
CUDA_VISIBLE_DEVICES=0 python -m experiments.refine_contact_full105 --shard-index 0 --num-shards 4 > /mnt/workspace/refine_contact_shard_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m experiments.refine_contact_full105 --shard-index 1 --num-shards 4 > /mnt/workspace/refine_contact_shard_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m experiments.refine_contact_full105 --shard-index 2 --num-shards 4 > /mnt/workspace/refine_contact_shard_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m experiments.refine_contact_full105 --shard-index 3 --num-shards 4 > /mnt/workspace/refine_contact_shard_3.log 2>&1 &
wait
