import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import argparse
from transmodel import ResNetClassifier

# ========================== 配置 ==========================
DEFAULT_INPUT_DIR = "./predict"
DEFAULT_MODEL_PATH = "best_resnet_model-9466.pth"
DEFAULT_HEATMAP_DIR = "heatmaps"


# =========================================================

def load_model(model_path, device='cuda'):
    model = ResNetClassifier(num_classes=2, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def create_data_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# def generate_heatmap(image_path, model, transform, device, heatmap_dir):
#     try:
#         # 图像预处理
#         image = Image.open(image_path).convert('RGB')
#         tensor = transform(image).unsqueeze(0).to(device)
#
#         # 注册hook获取特征图和梯度
#         features = None
#         gradients = None
#
#         def forward_hook(module, input, output):
#             nonlocal features
#             features = output.detach()
#
#         def backward_hook(module, grad_input, grad_output):
#             nonlocal gradients
#             gradients = grad_output[0].detach()
#
#         handle_forward = model.backbone.layer4.register_forward_hook(forward_hook)
#         handle_backward = model.backbone.layer4.register_backward_hook(backward_hook)
#
#         # 前向传播
#         output = model(tensor)
#         pred_class = output.argmax().item()
#
#         # 反向传播
#         model.zero_grad()
#         output[0, pred_class].backward()
#
#         # 确保特征和梯度维度匹配
#         if features is not None and gradients is not None:
#             # 计算权重（全局平均池化梯度）
#             weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
#
#             # 生成热图
#             heatmap = torch.sum(features * weights, dim=1).squeeze()
#             heatmap = nn.functional.relu(heatmap).cpu().numpy()
#             heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
#
#             # 调整热图尺寸
#             heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
#             heatmap = (heatmap * 255).astype(np.uint8)
#             heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#             # 保存热图
#             os.makedirs(heatmap_dir, exist_ok=True)
#             base_name = os.path.basename(image_path)
#             cv2.imwrite(os.path.join(heatmap_dir, f'heatmap_{base_name}'), heatmap)
#
#         handle_forward.remove()
#         handle_backward.remove()
#         return True
#
#     except Exception as e:
#         print(f"Error processing {image_path}: {str(e)}")
#         return False

def generate_heatmap(image_path, model, transform, device, heatmap_dir):
    try:
        # 原图预处理
        orig_image = Image.open(image_path).convert('RGB')
        image = orig_image.copy()
        tensor = transform(image).unsqueeze(0).to(device)

        # Hook相关变量
        features = None
        gradients = None

        # 注册hook
        def forward_hook(module, input, output):
            nonlocal features
            features = output.detach()

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        handle_forward = model.backbone.layer4.register_forward_hook(forward_hook)
        handle_backward = model.backbone.layer4.register_backward_hook(backward_hook)

        # 前向传播
        output = model(tensor)
        pred_class = output.argmax().item()

        # 反向传播
        model.zero_grad()
        output[0, pred_class].backward()

        if features is not None and gradients is not None:
            # 生成热力图
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            heatmap = torch.sum(features * weights, dim=1).squeeze()
            heatmap = nn.functional.relu(heatmap).cpu().numpy()

            # 归一化处理
            heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
            heatmap_8bit = (heatmap_norm * 255).astype(np.uint8)

            # 调整到原图尺寸
            heatmap_resized = cv2.resize(heatmap_8bit, orig_image.size)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

            # 创建可绘制图像
            overlay = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(overlay, 0.5, heatmap_color, 0.5, 0)

            # 多区域检测逻辑
            _, thresh = cv2.threshold(heatmap_resized, 200, 255, cv2.THRESH_BINARY)  # 自适应阈值
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 过滤小区域（面积>原图的0.5%）
            min_area = orig_image.size[0] * orig_image.size[1] * 0.005
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            # 动态绘制所有有效区域
            for cnt in valid_contours:
                # 获取旋转矩形框
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)

                # 绘制半透明填充+边框
                cv2.drawContours(overlay, [box], 0, (0, 0, 255), 2)
                cv2.drawContours(blended, [box], 0, (0, 0, 255), 2)

            # 保存结果
            os.makedirs(heatmap_dir, exist_ok=True)
            base_name = os.path.basename(image_path)

            # 保存带标注的融合图像
            cv2.imwrite(os.path.join(heatmap_dir, f'highlight_{base_name}'), blended)
            # 保存带标注的原图
            cv2.imwrite(os.path.join(heatmap_dir, f'bbox_{base_name}'), overlay)
            # 保存原始热力图
            cv2.imwrite(os.path.join(heatmap_dir, f'heatmap_{base_name}'), heatmap_color)

        handle_forward.remove()
        handle_backward.remove()
        return True

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=DEFAULT_INPUT_DIR)
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH)
    parser.add_argument('--heatmap_dir', default=DEFAULT_HEATMAP_DIR)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    transform = create_data_transform()

    # 获取图片列表
    valid_ext = ('png', 'jpg', 'jpeg')
    image_files = [f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(valid_ext)]

    # 生成热图
    for filename in image_files:
        img_path = os.path.join(args.input_dir, filename)
        generate_heatmap(img_path, model, transform, device, args.heatmap_dir)


if __name__ == '__main__':
    main()