import torchvision
import numpy as np
from PIL import Image
import os
import torch
from copy import deepcopy
from config import xc_root
from config import mg_root
xc_train_root = os.path.join(xc_root, 'train')
xc_test_root = os.path.join(xc_root, 'test')
mg_train_root = os.path.join(mg_root, 'train')
mg_test_root = os.path.join(mg_root, 'test')
def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices
class XCBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform,is_train=True,size_transform=None):

        super(XCBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))
        self.size_transform = size_transform
        self.is_train = is_train

    def __getitem__(self, item):

        img, label = super().__getitem__(item)

        # 获取原始图像路径
        path, target = self.samples[item]
        # 加载原始图像
        original_image = self.loader(path)
        # 获取原始尺寸
        original_size = original_image.size
        
        # areas = original_size[0] * original_size[1] /30000
        # 计算长宽比
        # aspect_ratios = original_size[0]  / original_size[1] /20
        scaled_size_tensor = torch.tensor([original_size[0]/200 , original_size[1]/200])
        if self.size_transform is not None and self.is_train:
            scaled_size_tensor = self.size_transform(scaled_size_tensor)
        uq_idx = self.uq_idxs[item]
        image_name=self.imgs[item][0]
        return img, label, uq_idx,image_name


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_xc_datasets(train_transform, test_transform,size_transform=None, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    # subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    # subsampled_100_classes = range
    # print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    # cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    whole_training_set = XCBase(root=xc_train_root, transform=train_transform,size_transform = size_transform)
    # whole_training_set = subsample_classes(xc_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    # whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    #计算unseen类索引
    # unseen_cls_idxs = [x for x, t in enumerate(whole_training_set.targets) if t in unseen_classes]
    # unseen_uq_idxs = whole_training_set.uq_idxs[unseen_cls_idxs]

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs) 
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))
    # 计算验证数据集
    # val_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    # val_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(val_indices)))
    # Get test set for all classes  未用到
    test_dataset = XCBase(root=xc_test_root, transform=test_transform)
    # test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(train_classes))


    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
        'offline_test_dataset': offline_test_dataset
    }
 
    return all_datasets


if __name__ == '__main__':

    x = get_xc_datasets(None, None, split_train_val=False,size_transform=None,
                               train_classes=range(6), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')