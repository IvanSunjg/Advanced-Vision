from arg_extractor import get_args
import experiment
import sys

# Script verify your command-line arguments passed to argparse
# Example script use: python arg_test.py --exp_type "mixup" --exp_kwargs "{'alpha': 1, 'min_lam': 0.2}"

print('RAW:', sys.argv[1:])
print()

args = get_args()
print('argparse')
print('-' * 10)
print(args)
print()

print(f'"{args.exp_type}" experiment, keyword args')
print('-' * 10)
print(args.exp_kwargs)
print()

exp = experiment.Experiment('train')
print('experiment')
print('-' * 10)
print(exp.construct_experiment(args.exp_type, **args.exp_kwargs))
