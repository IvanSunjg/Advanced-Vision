import sys
import os
import argparse

def get_args():	

	parser = argparse.ArgumentParser(
        description='Welcome to the MLP cluster\'s Pytorch training and inference helper script')

	parser.add_argument('--path', nargs="?", type=str, default="exp_1")
	
	args = parser.parse_args()
	
	return args
