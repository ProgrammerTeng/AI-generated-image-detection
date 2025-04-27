import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir='D:\pythonProject\AIdetection\Crawled', batch_size=16):
    # ImageNet标准归一化参数
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # 分割数据集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # 修改验证集transform
    val_subset.dataset.transform = val_transform

    # 创建加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader