cd /data/remote/github_code/face_detection/Pytorch_Retinaface;
python -W ignore train_ddp.py \
--dist-url 'tcp://127.0.0.1:9999' \
--dist-backend 'nccl' \
--multiprocessing-distributed=1 \
--world-size=1 \
--rank=0 \
--training_file="/data/remote/dataset/face_detection/train_face_detection_v2.log" \
--network="mobile0.25" \
--save_ckpt="/data/remote/dataset/face_detection/output_ckpt/mbet_1.8w_retinaface_ddp_continue" \
--pretrain=1 \
--log_dir="/data/remote/dataset/face_detection/output_log/mbet_1.8w_retinaface_ddp_continue" \
--pretrain_model="/data/remote/dataset/face_detection/output_ckpt/mbet_1.8w_retinaface_ddp/mobile0.25_epoch_220.pth" \
--epochs=100 --batch_size=32 --num_workers=8 --learning-rate=1e-3 \
--use_apex=0 \