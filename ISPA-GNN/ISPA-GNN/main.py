import argparse
from train import *

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=12, help='Nums of Training Epoch')  # 12
parser.add_argument('--batch_size', type=int, default=128, help='Nums of Batched Graph')  #
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--density', type=float, default=0.005, help='QoS Matrix Density')  # 0.005  0.01  0.015  0.02
parser.add_argument('--dim', type=int, default=32, help='Dimensionality of Embeddings')  # zyx 16 32 64 128
parser.add_argument('--order', type=int, default=2, help='Nums of order')
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--datatype', type=int, default=4, help='Type of dataset')

parser.add_argument('--gpu', default=1, action='store_true', help='Enable CUDA')
parser.add_argument('--version', type=int, default=4, help='Model Version')
parser.add_argument('--ctx', action='store_true', help='enable context graph')
parser.add_argument('--boxcox', action='store_true', help='enable boxcox transformation')
args = parser.parse_args()

File_name = 'Result.txt'

def fengefu():
    Note = open(File_name, mode='a')
    Note.write(
        '------------------------------------------------------------------------------------------------------------------\n')
    Note.close()


def show():
    Note = open(File_name, mode='a')
    Note.writelines(
        'Density : ' + str(args.density) + '  order : ' + str(args.order) + ' dim : ' + str(args.dim) + '\n')
    Note.writelines('\n')
    Note.close()


show()
eval(f'EvaluateV{args.version}(args, 1)')
eval(f'EvaluateV{args.version}(args, 2)')
fengefu()
