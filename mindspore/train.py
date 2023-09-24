import os
import argparse
import pdb

import numpy as np
import logging
import glob
from datetime import datetime

import mindspore as ms
from mindspore import context
from mindspore import nn, ops

from F2Net import F2Net
from ms.utils.dataloader import get_loader, get_loader_test
from data import get_segmentation_dataset
import mindspore.dataset as ds

ms.set_context(device_target='GPU', mode=context.GRAPH_MODE)

# import cv2


class Trainer:
    def __init__(self, train_loader, test_loader, model, optimizer, opt):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.opt = opt
        self.total_step = self.train_loader.get_dataset_size()  # TODO

        self.grad_fn = ops.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.resize = nn.ResizeBilinear()
        self.eval_loss_func = nn.L1Loss()
        self.best_dice = 0
        self.best_epoch = 0
        self.decay_rate = 0.1
        self.decay_epoch = 30
        self.sigmoid = nn.Sigmoid()

    def forward_fn(self, images, gts):
        output, output1, output2 = self.model(images)
        loss1 = self.structure_loss(output, gts)
        loss2 = self.structure_loss(output1, gts)
        loss3 = self.structure_loss(output2, gts)
        loss_total = loss1 + loss2 + loss3
        return loss_total, loss1

    def train_step(self, images, gts):
        (loss, loss1), grads = self.grad_fn(images, gts)
        self.optimizer(grads)
        return loss, loss1

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            self.model.set_train(True)
            self.adjust_lr(epoch)
            for step, data_pack in enumerate(self.train_loader.create_tuple_iterator(), start=1):
                images, gts = data_pack
                loss, loss1 = self.train_step(images, gts)

                # -- output loss -- #
                if step % 10 == 0 or step == self.total_step:
                    print(
                        '[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_1: {:.4f} Loss_total: {:0.4f}]'.
                        format(datetime.now(), epoch, epochs, step, self.total_step, loss1.asnumpy(), loss.asnumpy()))

                    logging.info(
                        '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_1: {:0.4f} Loss_total: {:0.4f}'.
                        format(epoch, opt.epoch, step, self.total_step, loss1.asnumpy(), loss.asnumpy()))

            self.test(epoch)

            if epoch % self.opt.save_epoch == 0:
                ms.save_checkpoint(model, os.path.join(save_path, 'Net_%d.pth' % (epoch)))

    def test(self, epoch):
        self.model.set_train(False)
        metric = []
        for i in range(3):
            metric.append(SegMetrics(2))
        for i, pack in enumerate(self.test_loader.create_tuple_iterator(), start=1):
            # ---- data prepare ----
            image, gt = pack
            # image = ops.ExpandDims()(image, 0)

            # ---- forward ----
            res, _, _ = self.model(image)
            #res = self.resize(res, size=gt.shape, align_corners=False)
            # pdb.set_trace()
            gt /= (gt.max() + 1e-8)
            # res = nn.Sigmoid()(res[0][0])  # TODO
            res = nn.Sigmoid()(res)
            res = (res > 0.5).float()
            # pdb.set_trace()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # pdb.set_trace()
            # mae_sum += self.eval_loss_func(res, gt)
            # res, gt = res.asnumpy(), gt.asnumpy()
            for i in range(len(metric)):
                metric[i].update(res[:, i, :, :], gt[:, i, :, :])
            # pdb.set_trace()
            # mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            # ---- recording loss ----
        # mae = mae_sum / self.test_loader.get_dataset_size()
        
        dice = []
        iou = []
        for i in range(len(metric)):
            dice1, iou1 = metric[i].eval()
            dice.append(dice1)
            iou.append(iou1)
        dice_avg, iou_avg = sum(dice) / 3 , sum(iou) / 3
        
        if epoch == 1:
            self.best_dice = dice_avg
            self.best_epoch = epoch
        else:
            if dice_avg > self.best_dice:
                self.best_dice = dice_avg
                self.best_epoch = epoch

                ms.save_checkpoint(model, os.path.join(save_path, 'best.ckpt'))
                print('best epoch:{}'.format(epoch))
                
        print('Epoch: {} dice: {} ####  bestdice: {} bestEpoch: {}'.format(epoch, dice_avg, self.best_dice, self.best_epoch))


    def structure_loss(self, pred, mask):
        pred = nn.Sigmoid()(pred)
        weit = 1 + 5 * ops.Abs()(ops.AvgPool(kernel_size=31, strides=1, pad_mode='same')(mask) - mask)
        wbce = nn.BCELoss(reduction='none')(pred, mask)
        wbce = (weit * wbce).sum(axis=(2, 3)) / weit.sum(axis=(2, 3))

        inter = ((pred * mask) * weit).sum(axis=(2, 3))
        union = ((pred + mask) * weit).sum(axis=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

    def adjust_lr(self, epoch):
        decay = self.decay_rate ** (epoch // self.decay_epoch)  # TODO
        self.optimizer.get_lr().set_data(self.opt.lr * decay)

def load_dataset(batch_size):
    
    train_img_paths = glob.glob('/dataset/BraTS2020/train_data/*')
    train_gt = glob.glob('/dataset/BraTS2020/train_gt/*')
    
    val_img_paths = glob.glob("/dataset/BraTS2020/val_data/*")
    val_gt = glob.glob("/dataset/BraTS2020/val_gt/*")

    train_dataset = get_segmentation_dataset('new', img_paths=train_img_paths, mask_paths=train_gt)
    val_dataset = get_segmentation_dataset('new', img_paths=val_img_paths, mask_paths=val_gt)

    train_data = ds.GeneratorDataset(train_dataset, ['image', 'label'], num_parallel_workers=2, shuffle=True)
    val_data = ds.GeneratorDataset(val_dataset, ['image', 'label'], num_parallel_workers=2, shuffle=True)
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    return train_data, val_data
       
class SegMetrics(ms.train.Metric):
    def __init__(self, num_class=2):
        super(SegMetrics, self).__init__()
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def clear(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        self.confusion_matrix += self._generate_matrix(y, y_pred)

    def eval(self):
        TN = self.confusion_matrix[0][0]
        FN = self.confusion_matrix[1][0]
        TP = self.confusion_matrix[1][1]
        FP = self.confusion_matrix[0][1]
        dice = 2 * TP / (2 * TP + FP + FN)
        iou = TP / (TP + FP + FN)
        return dice, iou
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=12, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=5, help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/')

    parser.add_argument('--train_img_dir', type=str, default='/dataset/TrainDataset/TrainDataset/image/')
    parser.add_argument('--train_gt_dir', type=str, default='/dataset/TrainDataset/TrainDataset/mask/')


    opt = parser.parse_args()

    ms.set_context(device_id=opt.gpu)

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=opt.save_model + '/log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                        datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Start Training")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(opt.epoch,
                                                                                                            opt.lr,
                                                                                                            opt.batchsize,
                                                                                                            opt.trainsize,
                                                                                                            opt.clip,
                                                                                                            opt.decay_rate,
                                                                                                            opt.save_model,
                                                                                                            opt.decay_epoch))

    # TIPS: you also can use deeper network for better performance like channel=64
    model = F2Net(args=opt)

    total = sum([param.nelement() for param in model.get_parameters()])
    print('Number of parameter:%.2fM' % (total / 1e6))

    optimizer = nn.SGD(model.trainable_params(), learning_rate=opt.lr)

    train_loader, test_loader = load_dataset(opt.batchsize)
    total_step = train_loader.get_dataset_size()
    
    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                                opt.batchsize, opt.save_model, total_step), '-' * 30)

    train = Trainer(train_loader, test_loader, model, optimizer, opt)
    train.train(opt.epoch)