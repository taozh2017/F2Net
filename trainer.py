import os
import torch
import glob
import torch.utils.data as data
from tqdm import tqdm

from kits import metrics
from kits import SegMetrics
from kits import configure_loss
from kits import LR_Scheduler
from kits import Saver
from kits import TensorboardSummary
from data import get_segmentation_dataset

class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device('cuda')
        self.model = model
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        train_img_paths = glob.glob("./data/train_data/*")
        train_gt = glob.glob("./data/train_gt/*")

        val_img_paths = glob.glob("./data/val_data/*")
        val_gt = glob.glob("./data/val_gt/*")
        train_dataset = get_segmentation_dataset('new', img_paths=train_img_paths, mask_paths=train_gt)
        val_dataset = get_segmentation_dataset('new', img_paths=val_img_paths, mask_paths=val_gt)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            shuffle=True,
                                            pin_memory=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=4,
                                          pin_memory=True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # Define criterion
        self.criterion1 = configure_loss('dice')
        self.criterion2 = configure_loss('bce')

        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define Evaluator
        self.evaluator1 = SegMetrics(2)
        self.evaluator2 = SegMetrics(2)
        self.evaluator3 = SegMetrics(2)

        self.evaluator1_1 = SegMetrics(2)
        self.evaluator1_2 = SegMetrics(2)
        self.evaluator1_3 = SegMetrics(2)

        self.evaluator2_1 = SegMetrics(2)
        self.evaluator2_2 = SegMetrics(2)
        self.evaluator2_3 = SegMetrics(2)

        # Resuming checkpoint
        self.train_loss = float('inf')
        self.best_pred = 0.0

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            self.scheduler(self.optimizer, iteration, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.model == 'F2Net':
                output, output_m1, output_m2 = self.model(image)
            elif self.args.modalities == 'all':
                output = self.model(image)
            else:
                raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            if self.args.model == 'F2Net':
                output = torch.sigmoid(output)
                output1 = torch.sigmoid(output_m1)
                output2 = torch.sigmoid(output_m2)
                dice_loss_1 = self.criterion1(output, target)
                loss_1 = self.criterion2(output, target)
                dice_loss_2 = self.criterion1(output1, target)
                loss_2 = self.criterion2(output1, target)
                dice_loss_3 = self.criterion1(output2, target)
                loss_3 = self.criterion2(output2, target)
                loss1 = dice_loss_1 + loss_1
                loss2 = dice_loss_2 + loss_2
                loss3 = dice_loss_3 + loss_3
                loss = loss1 + loss2 + loss3
            else:
                output = torch.sigmoid(output)
                dice_loss_1 = self.criterion1(output, target)
                bce_loss_1 = self.criterion2(output, target)
                loss = dice_loss_1 + bce_loss_1

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (iteration + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), iteration + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if iteration % (num_img_tr // 10) == 0:
                global_step = iteration + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss / (iteration + 1)))

        if train_loss < self.train_loss:
            self.train_loss = train_loss
            is_best = True
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'train_loss': self.train_loss
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0

        hausdorff_WT_sum = 0.0
        hausdorff_TC_sum = 0.0
        hausdorff_ET_sum = 0.0

        iou_WT_sum = 0.0
        iou_TC_sum = 0.0
        iou_ET_sum = 0.0

        for iteration, sample in enumerate(tbar):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]

            with torch.no_grad():
                if self.args.model == 'F2Net':
                    output, output_m1, output_m2 = self.model(image)
                elif self.args.modalities == 'all':
                    output = self.model(image)
                else:
                    raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            output = torch.sigmoid(output)
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + bce_loss
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (iteration + 1)))
            target1 = target.cpu()
            target1 = target1.numpy()

            pred = (output > 0.5).float()
            pred = pred.long().cpu()
            pred = pred.numpy()
            hausdorff_WT = metrics.hausdorff_95(pred[:, 0, :, :], target1[:, 0, :, :])
            hausdorff_TC = metrics.hausdorff_95(pred[:, 1, :, :], target1[:, 1, :, :])
            hausdorff_ET = metrics.hausdorff_95(pred[:, 2, :, :], target1[:, 2, :, :])
            hausdorff_WT_sum += hausdorff_WT
            hausdorff_TC_sum += hausdorff_TC
            hausdorff_ET_sum += hausdorff_ET
            self.evaluator1.update(pred[:, 0, None, :, :], target1[:, 0, None, :, :])
            self.evaluator2.update(pred[:, 1, None, :, :], target1[:, 1, None, :, :])
            self.evaluator3.update(pred[:, 2, None, :, :], target1[:, 2, None, :, :])

            iou_wt = metrics.iou_score(pred[:, 0, :, :], target1[:, 0, :, :])
            iou_tc = metrics.iou_score(pred[:, 1, :, :], target1[:, 1, :, :])
            iou_et = metrics.iou_score(pred[:, 2, :, :], target1[:, 2, :, :])
            iou_WT_sum += iou_wt
            iou_TC_sum += iou_tc
            iou_ET_sum += iou_et

        # Fast test during the training
        dice_WT = self.evaluator1.dice()
        dice_TC = self.evaluator2.dice()
        dice_ET = self.evaluator3.dice()
        dice_avg = (dice_WT + dice_TC + dice_ET) / 3

        hausdorff_WT = hausdorff_WT_sum / (iteration + 1)
        hausdorff_TC = hausdorff_TC_sum / (iteration + 1)
        hausdorff_ET = hausdorff_ET_sum / (iteration + 1)
        hausdorff_avg = (hausdorff_WT + hausdorff_TC + hausdorff_ET) / 3

        iou_wt = iou_WT_sum / (iteration + 1)
        iou_tc = iou_TC_sum / (iteration + 1)
        iou_et = iou_ET_sum / (iteration + 1)
        iou_avg = (iou_wt + iou_tc + iou_et) / 3

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/dice_WT', dice_WT, epoch)
        self.writer.add_scalar('val/dice_TC', dice_TC, epoch)
        self.writer.add_scalar('val/dice_ET', dice_ET, epoch)
        self.writer.add_scalar('val/dice_avg', dice_avg, epoch)
        self.writer.add_scalar('val/hausdorff_WT', hausdorff_WT, epoch)
        self.writer.add_scalar('val/hausdorff_TC', hausdorff_TC, epoch)
        self.writer.add_scalar('val/hausdorff_ET', hausdorff_ET, epoch)
        self.writer.add_scalar('val/hausdorff_avg', hausdorff_avg, epoch)

        sensitivity_WT = self.evaluator1.sensitivity()
        sensitivity_TC = self.evaluator2.sensitivity()
        sensitivity_ET = self.evaluator3.sensitivity()
        sensitivity_avg = (sensitivity_WT + sensitivity_TC + sensitivity_ET) / 3

        specificity_WT = self.evaluator1.specificity()
        specificity_TC = self.evaluator2.specificity()
        specificity_ET = self.evaluator3.specificity()
        specificity_avg = (specificity_WT + specificity_TC + specificity_ET) / 3

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('dice: WT:{}, TC:{}, ET:{}, avg:{}'
              .format(dice_WT, dice_TC, dice_ET, dice_avg).encode('utf-8'))
        print('hausdorff: WT:{}, TC:{}, ET:{}, avg: {}'
              .format(hausdorff_WT, hausdorff_TC, hausdorff_ET, hausdorff_avg).encode('utf-8'))
        print('sensitivity: WT:{}, TC:{}, ET:{}, avg: {}'
              .format(sensitivity_WT, sensitivity_TC, sensitivity_ET, sensitivity_avg).encode('utf-8'))
        print('specificity: WT:{}, TC:{}, ET:{}, avg: {}'
              .format(specificity_WT, specificity_TC, specificity_ET, specificity_avg).encode('utf-8'))
        print('IoU: WT:{}, TC:{}, ET:{}, avg: {}'
              .format(iou_wt, iou_tc, iou_et, iou_avg).encode('utf-8'))
        print('Loss: %.3f' % (test_loss / (iteration + 1)))

        new_pred = dice_avg
        if new_pred > self.best_pred:
            self.best_pred = new_pred
