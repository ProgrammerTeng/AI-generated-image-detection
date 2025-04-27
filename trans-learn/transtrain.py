import torch
from transmodel import create_model
from transdata_loader import get_data_loaders
from torch import optim, nn
from tqdm import tqdm


def main():
    # 设备配置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数配置
    config = {
        'batch_size': 16,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 30,
        'patience': 5,
        'freeze_epochs': 2  # 前几轮只训练分类头
    }

    # 加载数据
    train_loader, val_loader = get_data_loaders(batch_size=config['batch_size'])

    # 初始化模型（前两轮冻结主干）
    model = create_model(device, freeze_backbone=True)

    # 优化器和损失函数
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.2)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(config['num_epochs']):
        # 第二阶段解冻主干
        if epoch == config['freeze_epochs']:
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config['lr'] / 10,  # 降低学习率
                weight_decay=config['weight_decay']
            )
            print("\nUnfreezing backbone layers")

        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'Loss': loss.item()})

        # 验证阶段
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / len(val_loader.dataset)
        scheduler.step(val_acc)

        # 打印日志
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停与保存
        if val_acc >= best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_resnet_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()