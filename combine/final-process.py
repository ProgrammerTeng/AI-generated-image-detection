import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import shutil
import argparse
import csv
from tqdm import tqdm
import cv2
import numpy as np
import requests
import time
from transformers import BlipProcessor, BlipForConditionalGeneration

# ======================== 设备配置 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================== 目录配置 ========================
DEFAULT_INPUT_DIR = "./input_images"
DEFAULT_OUTPUT_ROOT = "./output_results"
DIR_STRUCTURE = {
    "classified": ["art", "photo"],
    "art_analysis": ["heatmaps"],
    "photo_analysis": ["resnet", "meaning", "lightangle"],
    "final_result": ["AI", "Non-AI"]
}


# ======================== 模型路径 ========================
CLASSIFY_MODEL_PATH = "final_model-0.9976.pth"
ART_MODEL_PATH = "best_resnet_model-9466.pth"
PHOTO_MODEL_PATH = "best_resnet_model-9805.pth"


# ======================== 模型定义 ========================
# ART鉴定模型
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self._init_weights(self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)

    def _init_weights(self, module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)



# ======================== 工具函数 ========================
def setup_directories(base_path):
    """创建所有需要的目录"""
    for category, subdirs in DIR_STRUCTURE.items():
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(category_path, subdir), exist_ok=True)
    return {
        "classified": os.path.join(base_path, "classified"),
        "art_analysis": os.path.join(base_path, "art_analysis"),
        "photo_analysis": os.path.join(base_path, "photo_analysis"),
        "final_result": os.path.join(base_path, "final_result")
    }


def get_image_files(input_dir):
    """获取所有支持的图片文件"""
    valid_ext = ('.png', '.jpg', '.jpeg')
    return [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]


# ======================== 核心功能模块 ========================
class ImageClassifier:
    def __init__(self):
        self.model = models.efficientnet_b1(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.model.load_state_dict(torch.load(CLASSIFY_MODEL_PATH, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def classify_image(self, image_path, output_dirs):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = self.model(input_tensor)
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()
                class_folder = "photo" if class_idx == 0 else "art"
                filename = os.path.basename(image_path)
                output_path = os.path.join(output_dirs["classified"], class_folder, filename)
                shutil.copy2(image_path, output_path)
                print(f"{os.path.basename(image_path)} -> Predicted class: {class_idx} -> {class_folder}")
                return class_folder, output_path
        except Exception as e:
            print(f"Error classifying {image_path}: {str(e)}")
            return None, None


class ArtAnalyzer:
    def __init__(self):
        self.model = ResNetClassifier(num_classes=2, pretrained=False)
        self.model.load_state_dict(torch.load(ART_MODEL_PATH, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def generate_heatmap(self, image_path, output_dir):
        try:
            orig_image = Image.open(image_path).convert('RGB')
            image = orig_image.copy()
            tensor = self.transform(image).unsqueeze(0).to(device)

            features = None
            gradients = None

            def forward_hook(module, input, output):
                nonlocal features
                features = output.detach()

            def backward_hook(module, grad_input, grad_output):
                nonlocal gradients
                gradients = grad_output[0].detach()

            handle_forward = self.model.backbone.layer4.register_forward_hook(forward_hook)
            handle_backward = self.model.backbone.layer4.register_backward_hook(backward_hook)

            output = self.model(tensor)
            pred_class = output.argmax().item()
            pred_label = "AI" if pred_class == 0 else "Non-AI"

            self.model.zero_grad()
            output[0, pred_class].backward()

            if features is not None and gradients is not None:
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                heatmap = torch.sum(features * weights, dim=1).squeeze()
                heatmap = nn.functional.relu(heatmap).cpu().numpy()
                heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
                heatmap_8bit = (heatmap_norm * 255).astype(np.uint8)
                heatmap_resized = cv2.resize(heatmap_8bit, orig_image.size)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

                overlay = cv2.cvtColor(np.array(orig_image), cv2.COLOR_RGB2BGR)
                blended = cv2.addWeighted(overlay, 0.5, heatmap_color, 0.5, 0)

                # 框选高激活区域
                _, thresh = cv2.threshold(heatmap_resized, 200, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = orig_image.size[0] * orig_image.size[1] * 0.005
                valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

                for cnt in valid_contours:
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    cv2.drawContours(overlay, [box], 0, (0, 0, 255), 2)
                    cv2.drawContours(blended, [box], 0, (0, 0, 255), 2)

                base_name = os.path.basename(image_path)
                os.makedirs(output_dir, exist_ok=True)
                #cv2.imwrite(os.path.join(output_dir, f'heatmap_{base_name}'), heatmap_color)
                cv2.imwrite(os.path.join(output_dir, f'bbox_{base_name}'), overlay)
                cv2.imwrite(os.path.join(output_dir, f'highlight_{base_name}'), blended)

            handle_forward.remove()
            handle_backward.remove()
            return pred_label

        except Exception as e:
            print(f"Error generating heatmap for {image_path}: {str(e)}")
            return "Error"


class PhotoAnalyzer:
    def __init__(self):
        # Photo鉴定模型
        self.photo_model = ResNetClassifier(num_classes=2, pretrained=False)
        self.photo_model.load_state_dict(torch.load(PHOTO_MODEL_PATH, map_location=device))
        self.photo_model = self.photo_model.to(device)
        self.photo_model.eval()

        # BLIP模型初始化
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(
            device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def resnet_predict(self, image_path):
        """使用ResNet模型预测图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = self.photo_model(tensor)
                probabilities = torch.softmax(outputs, dim=1)

            pred_prob, pred_class = torch.max(probabilities, dim=1)
            return "Non-AI" if pred_class.item() == 1 else "AI", pred_prob.item()
        except Exception as e:
            print(f"ResNet prediction error for {image_path}: {str(e)}")
            return "Error", 0.0

    def analyze_light_angle(self, image_path, output_dir):
        """分析光线角度一致性"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "Error", 0.0

            h, w = image.shape[:2]
            region_split = 2
            region_angles = []
            region_brightness = []
            region_size_h = h // region_split
            region_size_w = w // region_split

            arrow_img = image.copy()

            for i in range(region_split):
                for j in range(region_split):
                    y1, y2 = i * region_size_h, (i + 1) * region_size_h
                    x1, x2 = j * region_size_w, (j + 1) * region_size_w
                    region = image[y1:y2, x1:x2]
                    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=21)
                    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=21)
                    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
                    hist_angle = cv2.calcHist([angle.astype(np.uint8)], [0], None, [180], [0, 180])
                    main_angle = int(np.argmax(hist_angle))
                    region_angles.append(main_angle)

                    center = (x1 + region_size_w // 2, y1 + region_size_h // 2)
                    mean_brightness = np.mean(gray)
                    region_brightness.append(mean_brightness)

                    length = min(region_size_h, region_size_w) // 3
                    angle_rad = np.deg2rad(main_angle)
                    end_point = (int(center[0] + length * np.cos(angle_rad)),
                                 int(center[1] - length * np.sin(angle_rad)))
                    color = (0, 255, 0) if mean_brightness == max(region_brightness) else (0, 0, 255)
                    cv2.arrowedLine(arrow_img, center, end_point, color, 2, tipLength=0.3)
                    cv2.putText(arrow_img, f"{main_angle}", (center[0] - 10, center[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            region_angles = np.array(region_angles)
            std_angle = np.std(region_angles)
            tolerance = 30
            is_consistent = std_angle <= tolerance
            ai_judgement = "AI" if not is_consistent else "Non-AI"

            # 保存结果图片
            base_name = os.path.basename(image_path)
            cv2.imwrite(os.path.join(output_dir, f'light_analysis_{base_name}'), arrow_img)

            return ai_judgement, std_angle
        except Exception as e:
            print(f"Light analysis error for {image_path}: {str(e)}")
            return "Error", 0.0

    def analyze_caption_reasonableness(self, caption):
        """
        使用 Llama-4-Maverick-17B-128E-Instruct API 分析合理性
        直接来自 try-meaning.py
        """
        api_key = "02aafa4b-8f43-4583-9afe-89d18827252d"
        url = "https://api.sambanova.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        prompt = f"Is the following description reasonable in the real world? Answer only 'Reasonable' or 'Unreasonable'. Description: '{caption}'"
        data = {
            "stream": False,
            "model": "Llama-4-Maverick-17B-128E-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=20)
            response.raise_for_status()
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            # 只保留 Reasonable 或 Unreasonable
            if "unreasonable" in answer.lower():
                return "Unreasonable"
            else:
                return "Reasonable"
        except Exception as e:
            print(f"API error: {e}")
            return "Unknown"

    def visualize_caption(self, image_path, caption, output_dir):
        """
        可视化描述和合理性分析结果
        直接来自 try-meaning.py
        """
        image = Image.open(image_path).convert("RGB")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)
        margin = 10
        max_width = w - 2 * margin

        # 文本换行处理
        words = caption.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            (line_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if line_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        y = margin + 25
        for line in lines:
            cv2.putText(img_cv, line, (margin, y), font, font_scale, color, thickness)
            y += int(30 * font_scale) + 5

        # 分析和可视化合理性
        reasonableness = self.analyze_caption_reasonableness(caption)
        result_color = (0, 255, 0) if reasonableness == "Reasonable" else (0, 0, 255)
        cv2.putText(img_cv, reasonableness, (margin, h - 20), font, 0.8, result_color, 2)

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_dir, f'caption_{base_name}'), img_cv)

        return reasonableness

    def analyze_meaning(self, image_path, output_dir):
        """分析图片语义合理性"""
        try:
            # 生成描述
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            out = self.caption_model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # 可视化并分析合理性
            reason = self.visualize_caption(image_path, caption, output_dir)

            return reason, caption
        except Exception as e:
            print(f"Meaning analysis error for {image_path}: {str(e)}")
            return "Error", ""


# ======================== 主流程 ========================
def main(input_dir, output_root):
    # 设置目录结构
    dirs = setup_directories(output_root)

    # 初始化分析器
    classifier = ImageClassifier()
    art_analyzer = ArtAnalyzer()
    photo_analyzer = PhotoAnalyzer()

    # 获取所有图片
    image_files = get_image_files(input_dir)
    print(f"Found {len(image_files)} images to process")

    # 结果记录
    results = []

    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, filename)
        file_result = {"filename": filename}

        # 步骤1: 分类图片 (art/photo)
        img_type, output_path = classifier.classify_image(img_path, dirs)
        file_result["type"] = img_type

        if img_type == "art":
            # 步骤2: ART图片分析
            heatmap_dir = os.path.join(dirs["art_analysis"], "heatmaps")
            prediction = art_analyzer.generate_heatmap(img_path, heatmap_dir)
            file_result["art_prediction"] = prediction
            file_result["final_decision"] = prediction

            # 复制到最终结果目录
            dest_dir = os.path.join(dirs["final_result"], prediction)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

        elif img_type == "photo":
            # 步骤3: PHOTO图片分析 (三个方法)
            # 3.1 ResNet模型预测
            resnet_pred, resnet_conf = photo_analyzer.resnet_predict(img_path)
            file_result["resnet_pred"] = resnet_pred
            file_result["resnet_conf"] = f"{resnet_conf:.4f}"

            # 3.2 光线角度分析
            light_dir = os.path.join(dirs["photo_analysis"], "lightangle")
            light_pred, light_std = photo_analyzer.analyze_light_angle(img_path, light_dir)
            file_result["light_pred"] = light_pred
            file_result["light_std"] = f"{light_std:.2f}"

            # 3.3 语义合理性分析 (使用完整API方法)
            meaning_dir = os.path.join(dirs["photo_analysis"], "meaning")
            reason, caption = photo_analyzer.analyze_meaning(img_path, meaning_dir)
            file_result["reason"] = reason
            file_result["caption"] = caption

            # 综合决策 (所有方法必须一致认为Non-AI)
            if (resnet_pred == "Non-AI" and
                    light_pred == "Non-AI" and
                    reason == "Reasonable"):
                final_decision = "Non-AI"
            else:
                final_decision = "AI"

            file_result["final_decision"] = final_decision

            # 复制到最终结果目录
            dest_dir = os.path.join(dirs["final_result"], final_decision)
            shutil.copy2(img_path, os.path.join(dest_dir, filename))

        results.append(file_result)
        time.sleep(0.5)  # 避免API请求过快

    # 保存结果报告
    report_path = os.path.join(output_root, "analysis_report.csv")
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["filename", "type", "art_prediction", "resnet_pred", "resnet_conf",
                      "light_pred", "light_std", "reason", "caption", "final_decision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Processing completed! Report saved to: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Image Authenticity Analyzer")
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f"Input directory containing images (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root directory for output results (default: {DEFAULT_OUTPUT_ROOT})")

    args = parser.parse_args()

    # 规范化路径
    input_dir = os.path.normpath(args.input_dir)
    output_root = os.path.normpath(args.output_root)

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"Created input directory: {input_dir}")
        print("Please add images to this directory and rerun the script.")
        exit()

    main(input_dir, output_root)