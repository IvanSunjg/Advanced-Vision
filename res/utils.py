import re

def parse_slurm_output(filepath):
    stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    num_rgx = '''\b\d+\.\d+\b'''

    with open(filepath, 'r') as f:
        for line in f:
            for phase in ['train', 'val']:
                if line.startswith(phase):
                    loss, acc = re.findall(num_rgx, line)
                    stats[f'{phase}_loss'] = float(loss)
                    stats[f'{phase}_acc'] = float(acc)
                    break
    
    return stats
