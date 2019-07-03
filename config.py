import argparse
import os
import json


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--pretrained-model', type=bool, default=False)
    parser.add_argument('--stem', type=str, default='conv', help='conv, attention')

    args = parser.parse_args()

    if not os.path.isdir('config'):
        os.mkdir('config')

    with open('./config/config.txt', 'w') as f:
        json.dump(vars(args), f, indent=4)

    return args
