for i in {90000,110000}
  do 
     python3 main.py --mode test_brats --dataset BRATS --crop_size 500 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/test_results_$i \
                 --batch_size 1 --num_workers 4 --lambda_id 0.1 --test_iters $i
 done
