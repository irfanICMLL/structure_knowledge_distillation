is_pi_use=True
is_pa_use=True
is_ho_use=True
lambda_pi=10.0
lambda_d=0.1

# start kd from 0 step with loading the pretrain imgnet model on student 
CUDA_VISIBLE_DEVICES='3' python3 train_and_eval.py \
	--gpu 0 \
	--parallel False \
	--random-mirror \
	--random-scale \
	--weight-decay 5e-4 \
	--data-dir '../cityscapes' \
	--batch-size 8 \
	--num-steps 40000 \
	--is-student-load-imgnet True \
	--student-pretrain-model-imgnet ./dataset/resnet18-imagenet.pth \
	--pi ${is_pi_use} \
	--pa ${is_pa_use} \
	--ho ${is_ho_use} \
	--lambda-pa 0.5 \
	--pool-scale 0.5 \
	--lambda-pi ${lambda_pi} \
	--lambda-d ${lambda_d} \




