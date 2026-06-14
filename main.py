import argparse

import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
import random
from copy import deepcopy
import os


def get_kmeans_centroid_for_new_head(model, online_session_train_loader, args, device):
    model.to(device)
    model.eval()
    all_feats = []
    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model[0](images)   # backbone
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes+args.num_unlabeled_classes, random_state=0,n_init=10).fit(all_feats)
    #preds = kmeans.labels_
    centroids_np = kmeans.cluster_centers_   # (60, 768)
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)   # torch.Size([60, 768])
    #centroids = centroids.float()
    with torch.no_grad():
        _, logits = model[1](centroids)   # torch.Size([60, 50])
        max_logits, _ = torch.max(logits, dim=-1)   # torch.Size([60])
        _, proto_idx = torch.topk(max_logits, k=args.num_unlabeled_classes, largest=False)   # torch.Size([10])
        new_head = centroids[proto_idx]   # torch.Size([10, 768])
        # _, proto_idx2 = torch.topk(max_logits, k=args.num_labeled_classes, largest=True)   # torch.Size([10])

        
    # 获得最优匹配
    
    return new_head

def distribution_statistics_matching(logits, num_labeled_classes):
    """
    使用分布统计矩匹配技术实现类别平衡和类内均匀性
    
    通过控制以下三个方面实现与原熵方法相似的效果：
    1. 新旧类别组的一阶矩(均值)接近目标比例
    2. 旧类内部的二阶矩(方差)最大化，促进均匀分布
    3. 新类内部的二阶矩(方差)最大化，促进均匀分布




    """
    # 计算批次平均预测概率
    temp = 0.1  # 温度参数
    probs = (logits / temp).softmax(dim=1)
    avg_probs = probs.mean(dim=0)
    # 样本级分布约束（替代原样本级平衡）
    # 不要求每个样本的新旧类预测平衡，而是约束批次内的方差
    
    
    
    # 分离新旧类概率
    old_probs = avg_probs[:num_labeled_classes]
    new_probs = avg_probs[num_labeled_classes:]
    
    # 1. 组间一阶矩匹配 - 控制新旧类总体概率比例接近目标值
    old_mass = torch.sum(old_probs)
    new_mass = torch.sum(new_probs)
    # me_max_loss_old_new = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs))) 
             
    
 
    me_max_loss_old_new =  old_mass * torch.log(old_mass) + new_mass * torch.log(new_mass) + math.log(2)
    
    # 2. 旧类内部均匀性 - 通过最大化方差/最小化峰度
    # 归一化旧类概率
    if old_mass > 1e-10:
        old_probs_norm = old_probs / old_mass
        # 计算理想均匀分布
        uniform_old = torch.ones_like(old_probs_norm) / num_labeled_classes
        
        # 计算与均匀分布的差异 - L2距离
        old_uniformity_loss = torch.sum((old_probs_norm - uniform_old)**2)
    else:
        old_uniformity_loss = torch.tensor(0.0, device=logits.device)
    
    # 3. 新类内部均匀性
    num_new_classes = len(new_probs)
    if new_mass > 1e-10 and num_new_classes > 0:
        new_probs_norm = new_probs / new_mass
        # 计算理想均匀分布
        uniform_new = torch.ones_like(new_probs_norm) / num_new_classes
        
        # 计算与均匀分布的差异 - L2距离
        new_uniformity_loss = torch.sum((new_probs_norm - uniform_new)**2)
    else:
        new_uniformity_loss = torch.tensor(0.0, device=logits.device)
    
    # 加权组合损失
    group_weight = 3.0
    old_weight = 1
    new_weight = 1
    
    # 最终损失 - 所有损失项都是"距离"度量，越小越好
    print('me_max_loss_old_new:',me_max_loss_old_new,'old_uniformity_loss',old_uniformity_loss,'new_uniformity_loss',new_uniformity_loss)
    loss = group_weight * me_max_loss_old_new + old_weight * old_uniformity_loss + new_weight * new_uniformity_loss
    
    return loss


def train_offline(student, train_loader, test_loader, args):
    args.epochs_offline = 2
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs_offline,
            eta_min=args.lr * 1e-3,
        )

    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs_offline,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    # best acc log
    best_test_acc_old = 0

    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()

        student.train()
        for batch_idx, batch in enumerate(train_loader):

            images, class_labels, uq_idxs = batch   # NOTE!!! no mask lab in this setting
            mask_lab = torch.ones_like(class_labels)   # NOTE!!! all samples are labeled

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            student_proj, student_out = student(images)
            teacher_out = student_out.detach()

            # clustering, sup






            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

            # clustering, unsup

            '''
            【这种损失本质上是强制模型学习 “输入变换下的输出一致性”：
            对于同一样本的不同增强视图（如翻转、裁剪后的图像），模型应输出相似的预测分布，增强特征的鲁棒性。
            通过蒸馏机制，让学生模型从自身的 “历史稳定输出” 中学习，避免参数更新过程中的波动，提升聚类结果的稳定性。】

            '''


            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
            cluster_loss += args.memax_weight * me_max_loss

            # represent learning, unsup
            contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
            contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

            # representation learning, sup
            student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
            student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

            # Total loss
            loss = 0
            sup_weight = 0.65
            loss += (1 - sup_weight) * cluster_loss + sup_weight * cls_loss
            loss += (1 - sup_weight) * contrastive_loss + sup_weight * sup_con_loss

            # logs
            pstr = ''
            pstr += f'cls_loss: {cls_loss.item():.4f} '
            pstr += f'cluster_loss: {cluster_loss.item():.4f} '
            pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
            pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f}'.format(all_acc_test, old_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc_test > best_test_acc_old:

            args.logger.info(f'Best ACC on Old Classes on test set: {old_acc_test:.4f}...')

            torch.save(save_dict, args.model_path[:-3] + f'_best_old.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best_old.pt'))

            best_test_acc_old = old_acc_test

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test set: Old: {best_test_acc_old:.4f}')
        args.logger.info('\n')

'''====================================================================================================================='''
def test_offline(model, test_loader, epoch, save_name, args):
    model.eval()
    preds, targets = [], []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


def train(student, train_loader, test_loader, unlabelled_train_loader, args):
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )


    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )

    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()


        '''
        加目标检损失，变为不确定数目的聚类


        uq_idxs加损失或者加采样频率
        '''
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                # avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                # me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                # cluster_loss += args.memax_weight * me_max_loss
                cluster_loss += distribution_statistics_matching(student_out,args.num_labeled_classes)
                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                
            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)


        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # if old_acc_test > best_test_acc_lab:
        #     
        #     args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
        #     args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        #     
        #     torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
        #     args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
        #     
        #     # inductive
        #     best_test_acc_lab = old_acc_test
        #     # transductive            
        #     best_train_acc_lab = old_acc
        #     best_train_acc_ubl = new_acc
        #     best_train_acc_all = all_acc
        # 
        # args.logger.info(f'Exp Name: {args.exp_name}')
        # args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, test_loader, epoch, save_name, args):
    model.eval()
    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc

'''====================================================================================================================='''
def test_offline(model, test_loader, epoch, save_name, args):

    model.eval()

    preds, targets = [], []
    mask = np.array([])
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
   # parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='gc', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
   # parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=1, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='test', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    model_learnable_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print('Learnable Parameters: {}'.format(model_learnable_parameters))
    args.logger.info('model build')
    backbone_source = deepcopy(backbone)   # NOTE!!!

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader_labelled = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False, pin_memory=False)#Note
    train_loader_unlabelled = DataLoader(datasets["train_unlabelled"], num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False, pin_memory=False)#Note
    test_loader_all = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=512, shuffle=False, pin_memory=False)
    offline_session_train_loader = DataLoader(train_dataset.labelled_dataset, num_workers=args.num_workers,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    offline_session_test_loader = DataLoader(datasets["offline_test_dataset"], num_workers=args.num_workers,
                                                 batch_size=512, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
    model = nn.Sequential(backbone, projector).to(device)
    # ----------------------
    # TRAIN offline
    # ----------------------
    args.train_session = 'offline'
    if args.train_session == 'offline':
        args.logger.info('========== offline training with labeled old data (old) ==========')
        train_offline(model, offline_session_train_loader, offline_session_test_loader, args)
        new_head = get_kmeans_centroid_for_new_head(model, test_loader_unlabelled, args, device)
        torch.save(new_head, args.model_path[:-3] + f'_new_head.pt')
        args.logger.info("new_head saved to {}.".format(args.model_path[:-3] + f'_new_head.pt'))
    # ----------------------
    # TRAIN online
    # ----------------------
    args.train_session = 'online'
    '''load ckpts from last session (session>0) or offline session (session=0)'''
    args.logger.info('loading checkpoints of model_pre...')
    projector_pre = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
    model_pre = nn.Sequential(backbone_source, projector_pre)
    # load_dir_online = '/home/asus/SimGCD-zhouqi/check_point/xc_best_old.pt'
    # load_dir_online = '/home/asus/SimGCD-zhouqi/gc_checkpoint/model_best_old.pt'
    load_dir_online = args.model_path[:-3] + f'_best_old.pt'

    if load_dir_online is not None:                    
        args.logger.info('loading offline checkpoints from: ' + load_dir_online)
        load_dict = torch.load(load_dir_online)
        model_pre.load_state_dict(load_dict['model'])
        args.logger.info('successfully loaded checkpoints!')
    model_pre.to(device)
    args.logger.info('successfully loaded checkpoints!')
    args.logger.info('number of all class (old + all new) in current session: {}'.format(args.mlp_out_dim))
    projector_cur = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    args.init_new_head = True
    if args.init_new_head:
        new_head =None
        # load_newhead_online='/home/asus/SimGCD-me/dev_outputs/simgcd/log/xc_model_ln/checkpoints/model_new_head.pt'
        # load_newhead_online = '/home/asus/SimGCD-zhouqi/gc_checkpoint/gc_new_head.pt'
        load_newhead_online = args.model_path[:-3] + f'_new_head.pt'
        if load_newhead_online is not None:
            if not os.path.exists(load_newhead_online):
                new_head = get_kmeans_centroid_for_new_head(model_pre, test_loader_unlabelled, args, device)
                torch.save(new_head, load_newhead_online)
                args.logger.info("new_head saved to {}.".format(load_newhead_online))         
            args.logger.info('loading offline newhead from: ' + load_newhead_online)
            new_head = torch.load(load_newhead_online)
        if new_head is None:
            new_head = get_kmeans_centroid_for_new_head(model_pre, test_loader_unlabelled, args, device)   # torch.Size([10, 768])
        args.logger.info('transferring classification head of seen classes...')
        succeed_num = args.num_labeled_classes
        projector_cur.last_layer.weight_v.data[:succeed_num] = projector_pre.last_layer.weight_v.data[:succeed_num]   # NOTE!!!
        projector_cur.last_layer.weight_g.data[:succeed_num] = projector_pre.last_layer.weight_g.data[:succeed_num]   # NOTE!!!
        projector_cur.last_layer.weight.data[:succeed_num] = projector_pre.last_layer.weight.data[:succeed_num]   # NOTE!!!
        norm_new_head_weight_v = torch.norm(projector_cur.last_layer.weight_v.data[succeed_num:], dim=-1).mean()
        norm_new_head_weight = torch.norm(projector_cur.last_layer.weight.data[succeed_num:], dim=-1).mean()
        new_head_weight_v = new_head * norm_new_head_weight_v
        new_head_weight = new_head * norm_new_head_weight
        args.logger.info('initializing classification head of unseen novel classes...')
        projector_cur.last_layer.weight_v.data[succeed_num:] = new_head_weight_v.data   # NOTE!!!   # copy
        projector_cur.last_layer.weight.data[succeed_num:] = new_head_weight.data   # NOTE!!!
        ##############################################
    '''incremental parametric classifier in SimGCD'''
    ####################################################################################################################   
    backbone_cur = deepcopy(backbone_source)   # NOTE!!!
    backbone_cur.load_state_dict(model_pre[0].state_dict())   # NOTE!!!
    model_cur = nn.Sequential(backbone_cur, projector_cur)   # NOTE!!! backbone_cur
    model_cur.to(device)

    for m in model_pre.parameters():
        m.requires_grad = False

    train(model_cur, train_loader_labelled, None, test_loader_unlabelled, args)
    #给损失加cos值，后期逐渐回归正常水平