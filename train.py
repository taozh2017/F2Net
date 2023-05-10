import argparse
import os
import torch
from networks.pvt_new import F2Net

from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='brats', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--save_dir', type=str,
                    default='./results', help='output dir')
parser.add_argument('--epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--modalities', type=str, default='all',
                    choices=['all', 't1', 't1ce', 'flair', 't2'],
                    help='modalities (default: all)')

parser.add_argument('--model', type=str, default='F2Net',
                    choices=['F2Net'],
                    help='model name (default: F2Net)')

# optimizer params
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='M',
                    help='w-decay (default: 0.0001)')

# input params
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')

parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.model == 'F2Net':
        model = F2Net(args.num_classes).cuda()
    trainer = Trainer(args, model)
    for epoch in range(0, args.epochs):
        trainer.training(epoch)
        if epoch % 10 == 9:
            trainer.validation(epoch)
    trainer.writer.close()
    torch.cuda.empty_cache()
