python main.py\
 --model_name corediff\
 --run_name test_training_with_2detect_300k_epochs_random_crop_min_max_norm_fixed\
 --batch_size 32\
 --max_iter 300000\
 --image_size 512\
 --train_dataset 2detect\
 --test_dataset 2detect\
 --test_id 9\
 --context\
 --only_adjust_two_step\
 --save_freq 2500\
 --context_mock_strategy_for_1st_and_last_frames copy_neighbor\
 --normalization_strategy min_max\
 --train_set_crop_strategy random\
 --val_set_crop_strategy random\


