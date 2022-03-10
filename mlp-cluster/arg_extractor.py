import argparse
import ast

def get_args():	

	parser = argparse.ArgumentParser(
        description='Welcome to the MLP cluster\'s Pytorch training and inference helper script')
	###path to data###
	# if data is in same folder as py files you don't need to modify this argument
	parser.add_argument('--path', nargs="?", type=str, default="exp_1")
	#number of epochs to train
	parser.add_argument('--num_epochs', nargs="?", type=int, default=15)

	###type of otimizer###
	#it should be 'SGD' or'Adam'
	parser.add_argument('--optimizer_type', nargs="?", type=str, default='SGD')
	###otimizer hyperparameters, so far we used only 'SGD'###
	#see pytorch documentation for detailed description on hyperparameters
	#learning rate and weight decay can be used with both 'SGD' and 'Adam'
	parser.add_argument('--lr', nargs="?", type=float, default=0.006)
	parser.add_argument('--weight_decay', nargs="?", type=float, default=0.006)
	# momentum can only be used with 'SGD'
	parser.add_argument('--momentum', nargs="?", type=float, default=0.5)
	#beta1,beta2 and amsgrad can only be used with 'Adam'
	parser.add_argument('--beta1', nargs="?", type=float, default=0.9)
	parser.add_argument('--beta2', nargs="?", type=float, default=0.999)
	parser.add_argument('--amsgrad', nargs="?", type=bool, default=False)

	###type of Loss function###
	#It should be 'CEL' (cross entropy loss) or 'KLD' (Kullback-Leibler divergence)
	#We only used 'CEL' so far
	parser.add_argument('--loss_function', nargs="?", type=str, default='CEL')
	###Loss function hyperparameters###
	#Reduction to output in each batch should be 'mean' or 'sum' we only used 'mean' so far
	#Reduction can be applied to both 'CEL' and 'KLD'
	parser.add_argument('--reduction', nargs="?", type=str, default='mean')
	#Label smoothing punishes the labels that classified accurately, should range[0.0,1.0]
	#Label smoothing can only be applied to 'CEL'
	parser.add_argument('--label_smoothing', nargs="?", type=float, default=0.0)
	###Scheduler hyperparameters###
	#learning rate is decreased at multiples of step size (epochs) i.e(7,14,21), with cutout techniques the best so far is 4
	parser.add_argument('--step_size', nargs="?", type=int, default=7)
	#decrease the learning rate at step size with gamma
	parser.add_argument('--gamma', nargs="?", type=float, default=0.1)
	###The number of resnet blocks to freeze, should be between[1,10]
	parser.add_argument('--num_of_frozen_blocks', nargs="?", type=int, default=2)

	###Augmentation technique arguments###
	parser.add_argument('--exp_type', type=str)
	parser.add_argument('--exp_kwargs', type=lambda x: ast.literal_eval(x), default={})
	
	args = parser.parse_args()
	
	return args