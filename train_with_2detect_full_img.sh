python main.py\
 --model_name corediff\
 --run_name test_training_with_2detect_100k_epochs_full_img_min_max_norm_fixed\
 --batch_size 8\
 --max_iter 100000\
 --image_size 1024\
 --train_dataset 2detect\
 --test_dataset 2detect\
 --test_id 9\
 --context\
 --only_adjust_two_step\
 --save_freq 2500\
 --context_mock_strategy_for_1st_and_last_frames copy_neighbor\
 --normalization_strategy min_max\


