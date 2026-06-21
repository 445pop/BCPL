# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
完整的推理脚本：训练全部类，然后推理全部数据，拿到每个类别的边缘数据与中心数据，可以直接运行
"""
##import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import math

from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from torch.optim import SGD, lr_scheduler
from copy import deepcopy
from cluster import cubian_cluster
from cluster_dbscan import cubian_dbscan_cluster
def detect_and_record_unknown_images(model, data_loader, dataset, args, 
                                      threshold=0.5, output_file='unknown_images.jsonl'):
    model.eval()
    unknown_images = []
    
    print(f"\n{'='*80}")
    print(f"开始检测边缘类...")
    print(f"阈值: {threshold}")
    print(f"{'='*80}\n")
    
    # 确保数据集有samples属性
    has_samples = hasattr(dataset, 'samples')
    has_uq_idxs = hasattr(dataset, 'uq_idxs')
    
    global_idx = 0
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="检测中")):
        images, labels, uq_idxs,_ ,image_root= batch
        #统一使用第一个view
        images = images[0].cuda(non_blocking=True) 
        with torch.no_grad():
            _, logits = model(images)
            

            probs = torch.softmax(logits, dim=1)
            
            known_probs = probs[:, :args.num_labeled_classes]
            unknown_probs = probs[:, args.num_labeled_classes:]
            
            max_known_prob, _ = torch.max(known_probs, dim=1)
            max_unknown_prob, unknown_class_idx = torch.max(unknown_probs, dim=1)
            
            #可选是否设置阈值,可以进行选择
            #is_unknown = (max_unknown_prob > threshold) & (max_unknown_prob > max_known_prob)
            is_unknown=max_unknown_prob > max_known_prob
            unknown_indices = is_unknown.nonzero(as_tuple=True)[0].cpu().numpy()
            

            for idx in unknown_indices:
                batch_idx_offset = global_idx
                

                if has_samples:
                    if has_uq_idxs and int(uq_idxs[idx].item()) < len(dataset.samples):
                        real_idx = int(uq_idxs[idx].item())
                    else:
                        real_idx = batch_idx_offset + int(idx)
                    
                    if real_idx < len(dataset.samples):
                        image_path = dataset.samples[real_idx][0]
                    else:
                        image_path = f"batch_{batch_idx}_idx_{idx}"
                else:
                    image_path = f"batch_{batch_idx}_idx_{idx}"

                pred_class = unknown_class_idx[idx].item() + args.num_labeled_classes
                pred_prob = max_unknown_prob[idx].item()
                
                record = {
                    'image_path': image_path,
                    'predicted_unknown_class': int(pred_class),
                    'unknown_class_prob': float(pred_prob),
                    'max_known_prob': float(max_known_prob[idx].item()),
                    'batch_idx': batch_idx,
                    'idx_in_batch': int(idx),
                    'global_idx': batch_idx_offset + int(idx),
                    'image_root':image_root[int(idx)]
                }
                unknown_images.append(record)
        
        global_idx += images.size(0)
    

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in unknown_images:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*80}")
    print(f"检测完成！")
    print(f"总共处理了 {len(data_loader.dataset)} 张图片")
    print(f"发现 {len(unknown_images)} 张边缘类图片 ({len(unknown_images)/len(data_loader.dataset)*100:.2f}%)")
    print(f"结果已保存到: {output_file}")
    

    if unknown_images:
        unknown_probs = [r['unknown_class_prob'] for r in unknown_images]
        print(f"\n边缘类统计：")
        print(f"  - 平均概率: {np.mean(unknown_probs):.4f}")
        print(f"  - 最大概率: {np.max(unknown_probs):.4f}")
        print(f"  - 最小概率: {np.min(unknown_probs):.4f}")

        from collections import Counter
        class_counts = Counter([r['predicted_unknown_class'] for r in unknown_images])
        print(f"\n边缘类别分布：")
        for cls, count in sorted(class_counts.items()):
            print(f"  类别 {cls}: {count} 张 ({count/len(unknown_images)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return unknown_images


def train_offline(student, train_loader, test_loader, args):
    """只训练offline阶段（类别中心）"""
    args.epochs_offline = 100
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
    
    best_test_acc_original = 0
    
    for epoch in range(args.epochs_offline):
        loss_record = AverageMeter()
        student.train()
        for batch_idx, batch in enumerate(train_loader):
            #应用了多视角数据增强，所以维度变为了两倍
            if len(batch) == 5:
                images, class_labels, uq_idxs, mask_lab,_ = batch
                mask_lab = torch.ones_like(class_labels)
                class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

            else:
                    images, class_labels, uq_idxs,_= batch
                    mask_lab = torch.ones_like(class_labels)
                    
                    class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()

            images = torch.cat(images, dim=0).cuda(non_blocking=True)
            student_proj, student_out = student(images)
            teacher_out = student_out.detach()
            
            # clustering, sup
            sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
            sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
            cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
            
            # clustering, unsup
            cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
            avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
            me_max_loss = -torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
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
            
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        

            
        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        
        args.logger.info('Testing on test set...')
        all_acc_test, original_acc_test, _ = test_offline(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)
        args.logger.info('Test Accuracies: All {:.4f} | 原有类 {:.4f}'.format(all_acc_test, original_acc_test))
        
        exp_lr_scheduler.step()
        
        # 保存模型
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'num_labeled_classes': args.num_labeled_classes,
            'num_mlp_layers': args.num_mlp_layers,
        }
        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))
        
        if original_acc_test > best_test_acc_original:
            args.logger.info(f'Best ACC on 原有类: {original_acc_test:.4f}...')
            torch.save(save_dict, args.model_path[:-3] + f'_best_original.pt')
            args.logger.info("Best model saved.")
            best_test_acc_original = original_acc_test

    args.logger.info(f'\n训练完成！最佳准确率: {best_test_acc_original:.4f}\n')
    return student


def test_offline(model, test_loader, epoch, save_name, args):
    """测试函数"""
    model.eval()
    preds, targets = [], []
    mask = np.array([])
    
    for batch_idx, (images, label, _,_) in enumerate(tqdm(test_loader, desc="测试中", leave=False)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
    
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    all_acc, original_acc, edge_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, original_acc, edge_acc


def main():
    parser = argparse.ArgumentParser(description='只训练类别中心，然后检测边缘类')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='xc', help='xc, gc, etc.')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float)
    parser.add_argument('--teacher_temp', default=0.04, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int)
    
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='detect_cluster', type=str)
    
    parser.add_argument('--skip_training', action='store_true', default=True, 
                       help='跳过训练，直接使用已有模型推理')
    # parser.add_argument('--model_path', type=str, default='/root/data1/HDAR-main/dev_outputs/inference_only/log/detect_cluster_(28.11.2025_|_32.903)/checkpoints/model_best_old.pt', 
    #                    help='用于推理的模型路径（如果跳过训练）')
    parser.add_argument('--threshold', type=float, default=0.6, 
                       help='判定为边缘类的概率阈值')
    
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args.device = device
    
    # 获取类别分割
    args = get_class_splits(args)
    #类别数量
    args.num_labeled_classes =21
    args.num_unlabeled_classes=0
    # 初始化实验
    init_experiment(args, runner_name=['inference_only'])
    #东北大学模型
    #args.model_path='/root/data1/HDAR-main/dev_outputs/inference_only/log/detect_cluster_(23.01.2026_|_31.551)/checkpoints/model_best_old.pt'
    #马钢模型
    #args.model_path='/root/data1/HDAR-main/dev_outputs/inference_only/log/detect_cluster_(25.01.2026_|_53.364)/checkpoints/model_best_old.pt'
    #seversteel 模型
    #args.model_path=r'/root/data1/HDAR-main/dev_outputs/inference_only/log/detect_cluster_(26.01.2026_|_37.791)/checkpoints/model_best_old.pt'
    #南南铝模型
    args.model_path=r'/root/data1/HDAR-main/dev_outputs/inference_only/log/detect_cluster_(26.04.2026_|_40.983)/checkpoints/model.pt'
    args.logger.info(f'使用评估函数 {args.eval_funcs[0]} 打印结果')
    torch.backends.cudnn.benchmark = True
    # ======================================================================
    # 构建模型（只包含类别中心）
    # ======================================================================
    args.interpolation = 3
    args.crop_pct = 0.875
    
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    
    if args.warmup_model_dir is not None:
        args.logger.info(f'从 {args.warmup_model_dir} 加载权重')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    
    # 只微调部分层
    for m in backbone.parameters():
        m.requires_grad = False
    
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    
    # ======================================================================
    # 数据加载  数据集统一按照规格要求命名为x'c
    # ======================================================================
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args)

    offline_session_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    
    offline_session_test_loader = DataLoader(
        test_dataset, 
        num_workers=args.num_workers,
        batch_size=8, 
        shuffle=False, 
        pin_memory=False
    )
    
    # ======================================================================
    # 训练或加载模型
    # ======================================================================
    if args.skip_training:
        args.logger.info("跳过训练，加载已有模型...")
        if args.model_path is None:
            args.logger.error("必须指定 --model_path 参数")
            return
        
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
        model = nn.Sequential(backbone, projector).to(device)
        
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args.logger.info(f"模型加载完成: {args.model_path}")
    else:
        args.logger.info("="*80)
        args.logger.info("开始训练offline阶段（只训练类别中心）")
        args.logger.info("="*80)
        # 创建只包含类别中心的projector
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=args.num_mlp_layers)
        model = nn.Sequential(backbone, projector).to(device)
        
        # 训练
        model = train_offline(model, offline_session_train_loader, offline_session_test_loader, args)
        
        # 加载最佳模型
        best_model_path = args.model_path[:-3] + f'_best.pt'
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model'])
        args.logger.info(f"加载最佳模型: {best_model_path}")
    # ======================================================================
    # 扩展模型以包含边缘类
    # ======================================================================
    # 为了能够在推理时判断边缘类，需要重新定义包含已知+边缘类的模型
    args.logger.info("\n扩展模型以支持边缘类检测...")
    # ======================================================================
    # 步骤1: 提取特征并计算类别原型（prototype）
    # ======================================================================
    args.logger.info("提取特征并计算类别原型...")
    model_combined = nn.Sequential(backbone, projector).to(device)
    model_combined.eval()
    all_feats = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(DataLoader(
                train_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4,
                pin_memory=False
            ), desc="提取特征")):
            images, labels, _, _, _ = batch
            # 统一使用第一个view（避免batch_idx%2的不一致性）
            images = images[0].cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # 提取backbone特征
            feats = model_combined[0](images)  # backbone features
            feats = torch.nn.functional.normalize(feats, dim=-1)
            # 获取预测类别（用于计算类别原型）
            _, logits = model_combined(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_feats.append(feats.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_feats_tensor = torch.cat(all_feats, dim=0).to(device)  # [N, feat_dim]
    all_preds_tensor = torch.cat(all_preds, dim=0)  # [N]
    all_labels_tensor = torch.cat(all_labels, dim=0)  # [N]
    args.logger.info(f"提取到 {len(all_feats_tensor)} 个特征向量")
    #对所有的标签进行去重
    unique_labels = torch.unique(all_labels_tensor)
    unique_pred=torch.unique(all_preds_tensor)
    print('##############################################3')
    print(unique_labels)
    print(unique_pred)
    print('##############################################3')
    # 计算每个类别的原型（prototype）：按预测类别分组求均值
    class_prototypes = []
    for cls_id in unique_labels:
        mask = (all_preds_tensor == cls_id)
        if mask.sum() > 0:
            # 计算该类所有样本的特征均值并归一化
            cls_feats = all_feats_tensor[mask.cuda()]
            prototype = cls_feats.mean(dim=0)
            prototype = torch.nn.functional.normalize(prototype.unsqueeze(0), dim=-1).squeeze(0)
            class_prototypes.append(prototype)
        else:
            # 如果某个类别没有样本，使用随机初始化
            args.logger.warning(f"类别 {cls_id} 没有样本，使用随机初始化")
            prototype = torch.randn(args.feat_dim, device=device)
            prototype = torch.nn.functional.normalize(prototype, dim=-1)
            class_prototypes.append(prototype)
    class_prototypes = torch.stack(class_prototypes)  # [num_labeled_classes, feat_dim]
    args.logger.info(f"计算得到 {len(class_prototypes)} 个类别原型")
    # 步骤2: 使用分类不确定性筛选边界点
    args.logger.info("使用分类不确定性筛选边界点...")
    # 重新推理获取所有样本的logits和概率
    all_logits = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(DataLoader(
                train_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4,
                pin_memory=False
            ), desc="计算分类不确定性")):
            images, _, _, _, _ = batch
            images = images[0].cuda(non_blocking=True)
            _, logits = model_combined(images)
            all_logits.append(logits.cpu())
    all_logits_tensor = torch.cat(all_logits, dim=0).to(device)  # [N, num_labeled_classes]
    all_probs = torch.softmax(all_logits_tensor, dim=1)  # [N, num_labeled_classes]
    # 方法1: 使用margin（top1和top2概率差）筛选边界点
    top2_probs, top2_indices = torch.topk(all_probs, k=2, dim=1)  # [N, 2]
    margins = top2_probs[:, 0] - top2_probs[:, 1]  # [N]，margin越小越不确定
    # 方法2: 使用entropy筛选边界点
    entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=1)  # [N]，entropy越大越不确定
    # 结合margin和entropy筛选边界点（选择margin小且entropy大的样本）
    # 使用分位数阈值：选择margin最小的10%或entropy最大的10%
    margin_threshold = torch.quantile(margins, 0.1)  # 最小的10%
    entropy_threshold = torch.quantile(entropy, 0.1)  # 最大的10%
    boundary_mask = (margins < margin_threshold) & (entropy > entropy_threshold)
    boundary_feats = all_feats_tensor[boundary_mask]  # [M, feat_dim]
    boundary_margins = margins[boundary_mask]
    boundary_entropy = entropy[boundary_mask]
    args.logger.info(f"筛选出 {len(boundary_feats)} 个边界样本 (占总数的 {len(boundary_feats)/len(all_feats_tensor)*100:.2f}%)")
    if len(boundary_feats) < 2:
        args.logger.warning("边界样本数量不足，无法进行聚类！")
        # 兜底：使用entropy最大的样本
        _, top_entropy_indices = torch.topk(entropy, k=min(10, len(entropy)), largest=True)
        boundary_feats = all_feats_tensor[top_entropy_indices]
        args.logger.info(f"使用entropy最大的 {len(boundary_feats)} 个样本作为边界样本")
    # 步骤3: 对边界样本进行DBSCAN聚类
    args.logger.info("对边界样本进行DBSCAN聚类...")
    unknown_centroids = cubian_dbscan_cluster(boundary_feats, class_prototypes)
    num_centroids = unknown_centroids.size(0)
    # 步骤4: 创建扩展的检测头并初始化权重 (修正版)
    args.logger.info("创建扩展的检测头并初始化权重...")
    # 1. 创建扩展头
    projector_extended = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes + num_centroids, 
                                   nlayers=args.num_mlp_layers)
    # 2. 复制原有类的参数 (v, g, 和 computed weight)
    # 注意：如果使用了weight_norm，起作用的是 weight_v 和 weight_g，weight 只是计算结果
    with torch.no_grad():
        # 复制原有类别的权重
        projector_extended.last_layer.weight_v.data[:args.num_labeled_classes] = projector.last_layer.weight_v.data
        projector_extended.last_layer.weight_g.data[:args.num_labeled_classes] = projector.last_layer.weight_g.data
        # 获取原有类别的 weight_g 的平均值
        mean_g = projector.last_layer.weight_g.data.view(-1).mean()
        # C. 初始化边缘类别的权重
        # unknown_centroids是归一化的纯粹的方向
        # 设置边缘类别的方向 (weight_v)
        mean_v_norm = projector.last_layer.weight_v.data.norm(dim=1, keepdim=True).mean()
        projector_extended.last_layer.weight_v.data[args.num_labeled_classes:] = unknown_centroids * mean_v_norm
        # 让边缘类别的 Logit 放大倍数与原有类别一致
        projector_extended.last_layer.weight_g.data[args.num_labeled_classes:] = mean_g

    args.logger.info(f"扩展检测头初始化完成：原有类 {args.num_labeled_classes} 个，边缘类 {num_centroids} 个")
    args.logger.info(f"边缘类权重初始化使用了均值 Gain: {mean_g.item():.4f}")
    model_extended = nn.Sequential(deepcopy(backbone), projector_extended).to(device)
    # 加载backbone权重
    model_extended[0].load_state_dict(model[0].state_dict())
    # 推理：检测边缘类
    args.logger.info("\n" + "="*80)
    args.logger.info("开始检测边缘类")
    args.logger.info("="*80)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'dev_outputs/inference_only/log/new_unknown_{args.dataset_name}_{timestamp}.jsonl'
    # 在所有测试数据上检测
    unknown_images = detect_and_record_unknown_images(
        model_extended,
        data_loader=DataLoader(
            train_dataset,
            num_workers=args.num_workers,
            batch_size=8,
            shuffle=False,
            pin_memory=False
        ),
        dataset=train_dataset,
        args=args,
        threshold=0.15,
        output_file=output_file
    )
    
    args.logger.info("\n" + "="*80)
    args.logger.info("全部完成！")
    args.logger.info("="*80)
if __name__ == "__main__":
    main()
