cd /data/remote/github_code/face_detection/Pytorch_Retinaface;
python -W ignore -m torch.distributed.launch --nproc_per_node 8 train_ddp_lanch.py \
--training_file="/data/remote/dataset/wider_face/widerface/train/label.txt" \
--network="mobile0.25" \
--save_ckpt="/data/remote/dataset/face_detection/output_ckpt/mbet_0.25_widerface_150epoch_baseline_cosine" \
--pretrain=0 \
--log_dir="/data/remote/dataset/face_detection/output_log/mbet_0.25_widerface_150epoch_baseline_cosine" \
--pretrain_model="/data/remote/dataset/face_detection/output_ckpt/mbet_1.8w_retinaface_ddp/mobile0.25_epoch_220.pth" \
--epochs=150 --batch_size=64 --num_workers=8 --learning-rate=1e-3 \
--use_apex=0 \
--syncbn=0 --cosine_lr=1

# inner data
# /data/remote/dataset/face_detection/train_face_detection_v2.log