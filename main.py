import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
from tqdm import tqdm

from config import get_args
from model import ResNet50, ResNet38, ResNet26
from preprocess import load_data


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()

    train_acc = 0.0
    step = 0
    for data, target in train_loader:
        adjust_learning_rate(optimizer, epoch, args)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % args.print_interval == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))


def eval(model, test_loader, args):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="eval", mininterval=1):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}'.format(acc))
    return acc


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main(args):
    train_loader, test_loader = load_data(args)

    if args.pretrained_model:
        model = ResNet26(args)
        filename = './checkpoint/best_model_ckpt.t7'
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        max_acc = checkpoint['acc']
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, max_acc))
    else:
        model = ResNet26(args)
        start_epoch = 1
        max_acc = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    print("Number of model parameters: ", get_model_parameters(model))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch, args)
        eval_acc = eval(model, test_loader, args)

        if max_acc < eval_acc:
            print('Model Saving ...')
            max_acc = eval_acc
            parameters = get_model_parameters(model)
            state = {
                'model': model.state_dict(),
                'acc': max_acc,
                'epoch': epoch,
                'parameters': parameters,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            filename = './checkpoint/best_model_ckpt.t7'
            torch.save(state, filename)


if __name__ == '__main__':
    args = get_args()
    main(args)
