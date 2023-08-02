# JUST CHANGE last_epoch value


let last_epoch=230000 
let remaining_epochs=300000-$last_epoch

echo "                  "
echo "starting from epoch:" $last_epoch
echo "remaining epochs = " $remaining_epochs
echo "                  "

python3 main.py --mode train --dataset BRATS --crop_size 500 --image_size 256 --c_dim 1 \
                 --image_dir data/brats/syn \
                 --sample_dir brats_syn_256_lambda0.1/samples \
                 --log_dir brats_syn_256_lambda0.1/logs \
                 --model_save_dir brats_syn_256_lambda0.1/models \
                 --result_dir brats_syn_256_lambda0.1/results \
                 --batch_size 1 --num_workers 4 --lambda_id 0.1 \
                \
                \
                    --resume_iters $last_epoch \
                    --num_iters $remaining_epochs\
 