CUDA_VISIBLE_DEVICES=GPU_NUM python main.py  --test_data_dir=Path_to_Testing_Dataset --batch_size=24 --workers=8  --height=32 --width=256 --arch=ResNet_ASTER --with_lstm --logs_dir=experiment_name --real_logs_dir=./aster.pytorch --max_len=200 --resume=Path_to_checkpoints --test