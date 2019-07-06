import argparse
import os
import logging
import logging.handlers


# DEBUG < INFO < WARNING < ERROR < CRITICAL
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s | %(filename)s:%(lineno)s] %(asctime)s: %(message)s')

    if not os.path.isdir('log'):
        os.mkdir('log')

    file_handler = logging.FileHandler('./log/' + filename + '.log')
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--model-name', type=str, default='ResNet26', help='ResNet26, ResNet38, ResNet50')
    parser.add_argument('--img-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--print-interval', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--pretrained-model', type=bool, default=False)
    parser.add_argument('--stem', type=bool, default=False, help='attention: True, conv: False')

    args = parser.parse_args()

    logger = get_logger('train')
    logger.info(vars(args))

    return args, logger
