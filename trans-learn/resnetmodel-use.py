import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import csv
from tqdm import tqdm
from transmodel import ResNetClassifier  # 确保model.py在相同目录

# ========================== 默认路径配置 ==========================
DEFAULT_INPUT_DIR = "./predict"
DEFAULT_MODEL_PATH = "best_resnet_model-9466.pth"
DEFAULT_OUTPUT_CSV = "predictions.csv"
# ===============================================================

def load_model(model_path, device='cuda'):
    """加载训练好的模型"""
    # 初始化模型结构（必须与训练时完全一致）
    model = ResNetClassifier(num_classes=2, pretrained=False)

    # 加载训练权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    # 设备设置
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model


def create_data_transform():
    """创建与验证集一致的数据预处理流程"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict_image(image_path, model, transform, device):
    """单张图片预测"""
    try:
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)  # 添加批次维度

        # 执行预测
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # 获取结果
        pred_prob, pred_class = torch.max(probabilities, dim=1)
        return {
            'class': pred_class.item(),
            'prob': pred_prob.item(),
            'status': 'success'
        }
    except Exception as e:
        return {
            'class': None,
            'prob': None,
            'status': f'error: {str(e)}'
        }


def main():
    # 参数解析器配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        type=str,
                        default=DEFAULT_INPUT_DIR,
                        help=f'Input image directory (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--model_path',
                        type=str,
                        default=DEFAULT_MODEL_PATH,
                        help=f'Model path (default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--output_csv',
                        type=str,
                        default=DEFAULT_OUTPUT_CSV,
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT_CSV})')
    args = parser.parse_args()

    # 路径规范化处理
    args.input_dir = os.path.normpath(args.input_dir)
    args.model_path = os.path.normpath(args.model_path)
    args.output_csv = os.path.normpath(args.output_csv)

    # 设备检测
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 初始化组件
    model = load_model(args.model_path, device)
    transform = create_data_transform()

    # 获取图片列表
    image_files = [f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    print(f'Found {len(image_files)} images to classify')

    # 批量预测
    results = []
    for filename in tqdm(image_files, desc='Processing'):
        img_path = os.path.join(args.input_dir, filename)
        result = predict_image(img_path, model, transform, device)
        results.append({
            'filename': filename,
            'pred_class': 'AI' if result['class'] == 0 else 'Non-AI',  # 根据训练时的类别顺序调整
            'confidence': f"{result['prob']:.4f}",
            'status': result['status']
        })

    # 保存结果
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'pred_class', 'confidence', 'status'])
        writer.writeheader()
        writer.writerows(results)
    print(f'Predictions saved to {args.output_csv}')


if __name__ == '__main__':
    main()