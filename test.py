import os
import sys
from tqdm import tqdm
import glob
import cv2
import argparse

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import torch
import torch.utils.data as data

from data import get_segmentation_dataset
from networks.pvt_new import F2Net

parser = argparse.ArgumentParser(description='Segmentation Training With Pytorch')

# model and dataset
parser.add_argument('--modalities', type=str, default='all',
                    choices=['all', 't1', 't1ce', 'flair', 't2'],
                    help='modalities (default: all)')

parser.add_argument('--model', type=str, default='F2Net',
                    choices=['F2Net'],
                    help='model name (default: F2Net)')

# checkpoint and log
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')

args = parser.parse_args()


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')

        # Define Dataloader
        img_paths = glob.glob("./data/test_data/*")
        gt_paths = glob.glob("./data/test_gt/*")

        whole_dataset = get_segmentation_dataset('new', img_paths=img_paths, mask_paths=gt_paths)

        whole_size = len(whole_dataset)
        print('test size: ', whole_size)

        self.test_loader = data.DataLoader(dataset=whole_dataset,
                                           batch_size=1,
                                           num_workers=4,
                                           pin_memory=True)

        self.model = F2Net(num_classes=3).cuda()

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print(checkpoint)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], False)
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.test_loader)

        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]

            with torch.no_grad():
                if self.args.model == 'F2Net':
                    output, _, _ = self.model(image)

            output = torch.sigmoid(output)
            pred = (output > 0.5).float()
            pred_show = torch.zeros_like(pred, dtype=torch.uint8)
            target_show = torch.zeros_like(target, dtype=torch.uint8)

            for i in range(pred.size(0)):
                for h in range(pred.size(2)):
                    for w in range(pred.size(3)):
                        if pred[i, 0, h, w] == 1:
                            pred_show[i, 0, h, w] = 255
                            pred_show[i, 1, h, w] = 201
                            pred_show[i, 2, h, w] = 14
                        if target[i, 0, h, w] == 1:
                            target_show[i, 0, h, w] = 255
                            target_show[i, 1, h, w] = 201
                            target_show[i, 2, h, w] = 14
                        if pred[i, 1, h, w] == 1:
                            pred_show[i, 0, h, w] = 153
                            pred_show[i, 1, h, w] = 217
                            pred_show[i, 2, h, w] = 234
                        if target[i, 1, h, w] == 1:
                            target_show[i, 0, h, w] = 153
                            target_show[i, 1, h, w] = 217
                            target_show[i, 2, h, w] = 234
                        if pred[i, 2, h, w] == 1:
                            pred_show[i, 0, h, w] = 185
                            pred_show[i, 1, h, w] = 122
                            pred_show[i, 2, h, w] = 87
                        if target[i, 2, h, w] == 1:
                            target_show[i, 0, h, w] = 185
                            target_show[i, 1, h, w] = 122
                            target_show[i, 2, h, w] = 87

                pred_save = pred_show[i].cpu().numpy().transpose((1, 2, 0))[::-1, :, ::-1]
                target_save = target_show[i].cpu().numpy().transpose((1, 2, 0))[::-1, :, ::-1]

                if not os.path.exists('./show_{}/{}'.format(self.args.model, fn[i])):
                    os.makedirs('./show_{}/{}'.format(self.args.model, fn[i]))
                cv2.imwrite(
                    './show_{}/{}/ours.png'.format(self.args.model, fn[i], self.args.model),
                    pred_save)
                cv2.imwrite(
                    './show_{}/{}/gt.png'.format(self.args.model, fn[i]),
                    target_save)


if __name__ == '__main__':
    trainer = Tester(args)
    trainer.validation(0)

    torch.cuda.empty_cache()
