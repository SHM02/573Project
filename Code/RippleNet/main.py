import argparse
import numpy as np
from train import train
from data_loader import load_data

np.random.seed(555)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='dataset to use')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--n_memory', type=int, default=32, help='ripple set size at every hop')
parser.add_argument('--n_hop', type=int, default=2, help='maximum number of hops to build ripple set')
parser.add_argument('--kge_weight', type=float, default=0.01, help='weight of the KGE term')
parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
parser.add_argument('--l2_weight', type=float, default=1e-7, help='L2 regularization weight')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate for the system')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',
                    help='item embedding updation mode at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='flag for last vs all hops for making predictions')


args = parser.parse_args()
data_info = load_data(args)
show_loss = False
train(args, data_info, show_loss)
