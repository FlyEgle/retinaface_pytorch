CUDA_VISIBLE_DEVICES=0 python train.py --training_dataset="/data/remote/dataset/face_detection/train_face_detection.log" \
--network="mobile0.25" --save_folder="/data/remote/dataset/face_detection/output_ckpt/mbet_1_5w_data" \
--pretrain=1 --log_dir="/data/remote/dataset/face_detection/output_log/mbet_1_5w_data"