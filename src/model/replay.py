import random
import numpy as np


def replay_node_selection(args, data, model=None):
    if args.replay_strategy == 'random':
        return random_sampling(data['train_x'].shape[2], args.replay_num_samples)
    else:
        args.logger.info("repaly node selection mode illegal!")

def random_sampling(data_size, num_samples):
    return np.random.choice(data_size, num_samples)
