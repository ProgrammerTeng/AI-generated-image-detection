import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from imblearn.over_sampling import RandomOverSampler
import torch.cuda.amp as amp  # 混合精度训练

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


# 1. 数据集准备
class ArtPhotoDataset(Dataset):
    def __init__(self, photo_dir, art_dir, transform=None):
        self.photo_paths = []
        self.art_paths = []
        self.transform = transform

        # 收集照片路径 (标签0)
        for root, _, files in os.walk(photo_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.photo_paths.append(os.path.join(root, file))

        # 收集艺术画路径 (标签1)
        for root, _, files in os.walk(art_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.art_paths.append(os.path.join(root, file))

        # 组合路径和标签
        self.image_paths = self.photo_paths + self.art_paths
        self.labels = [0] * len(self.photo_paths) + [1] * len(self.art_paths)

        # 过采样艺术画样本
        ros = RandomOverSampler(random_state=42)
        indices = np.arange(len(self.image_paths)).reshape(-1, 1)
        resampled_indices, _ = ros.fit_resample(indices, self.labels)

        self.image_paths = [self.image_paths[i[0]] for i in resampled_indices]
        self.labels = [self.labels[i[0]] for i in resampled_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像作为占位符
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


# 2. 数据增强 - 简化增强操作
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),  # 保留水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 3. 模型定义 - 使用更小的模型
def create_model():
    # 使用更小的EfficientNet-B1模型
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)

    # 冻结特征提取层
    for param in model.parameters():
        param.requires_grad = False

    # 简化分类器
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),  # 减少中间层大小
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    return model.to(device)


# 4. 训练函数 - 添加混合精度训练和早停
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, phase1_epochs=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scaler = amp.GradScaler()  # 混合精度训练
    patience = 3  # 早停耐心值
    no_improve = 0  # 无改善计数器

    # 第一阶段：只训练分类器
    print("\n=== Phase 1: Training Classifier ===")
    for epoch in range(phase1_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{phase1_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Phase1 Epoch {epoch + 1}/{phase1_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 第二阶段：训练所有层
    print("\n=== Phase 2: Full Network Training ===")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader,
                                   desc=f"Epoch {phase1_epochs + epoch + 1}/{phase1_epochs + num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 混合精度训练
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # 验证集评估
        val_loss, val_acc, f1, cm = evaluate(model, criterion, val_loader)
        print(f'Epoch {phase1_epochs + epoch + 1}/{phase1_epochs + num_epochs} | '
              f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {f1:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0  # 重置计数器
        else:
            no_improve += 1
            print(f'No improvement for {no_improve} epochs')

            # 早停检查
            if no_improve >= patience:
                print(f'Early stopping at epoch {phase1_epochs + epoch + 1}')
                break

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


# 5. 评估函数
def evaluate(model, criterion, dataloader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return loss, acc, f1, cm


# 6. 主执行流程
def main():
    # 配置参数 - 减少epoch和增加batch size
    BATCH_SIZE = 128  # 增加批处理大小
    NUM_EPOCHS = 8  # 减少第二阶段轮次
    PHASE1_EPOCHS = 3
    LR = 0.001
    N_SPLITS = 5

    # 创建数据集
    dataset = ArtPhotoDataset(
        photo_dir='GenImage',
        art_dir='Crawled',
        transform=None
    )

    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), dataset.labels)):
        print(f"\n{'=' * 40}")
        print(f"Fold {fold + 1}/{N_SPLITS}")
        print(f"{'=' * 40}")

        # 创建子集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # 应用不同的变换
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        # 创建数据加载器
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True, persistent_workers=True)

        # 初始化模型
        model = create_model()

        # 损失函数（带类别权重）
        class_counts = np.bincount(dataset.labels)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 分离参数
        feature_extractor_params = []
        classifier_params = []

        for name, param in model.named_parameters():
            if 'classifier' not in name:
                feature_extractor_params.append(param)
            else:
                classifier_params.append(param)

        # 设置优化器
        optimizer = optim.Adam([
            {'params': feature_extractor_params, 'lr': LR / 10},
            {'params': classifier_params, 'lr': LR}
        ], weight_decay=1e-4)

        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1, verbose=True
        )

        # 训练模型
        model = train_model(
            model, criterion, optimizer,
            train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            phase1_epochs=PHASE1_EPOCHS
        )

        # 评估模型
        _, val_acc, f1, cm = evaluate(model, criterion, val_loader)
        fold_results.append({
            'accuracy': val_acc.item(),
            'f1_score': f1,
            'confusion_matrix': cm
        })
        print(f"Fold {fold + 1} Results: Acc={val_acc:.4f}, F1={f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        # 保存每个fold的模型
        torch.save(model.state_dict(), f'model_fold_{fold + 1}.pth')

    # 打印最终结果
    avg_acc = np.mean([res['accuracy'] for res in fold_results])
    avg_f1 = np.mean([res['f1_score'] for res in fold_results])
    print(f"\n{'=' * 40}")
    print(f"Final Results: Average Acc={avg_acc:.4f}, Average F1={avg_f1:.4f}")
    print(f"{'=' * 40}")

    # 保存平均模型
    torch.save(model.state_dict(), 'final_model-0.9996.pth')


if __name__ == "__main__":
    main()