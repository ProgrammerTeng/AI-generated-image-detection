import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import shutil

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义与训练时相同的预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# 定义模型结构（必须与训练时完全相同）
def create_model():
    model = models.efficientnet_b1(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )
    return model.to(device)


# 加载训练好的模型
model = create_model()
model.load_state_dict(torch.load('final_model-0.9996.pth', map_location=device))
model.eval()  # 设置为评估模式

# 创建输出目录
output_dir = "outputclassify"
os.makedirs(os.path.join(output_dir, "photo"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "art"), exist_ok=True)

# 处理测试图像
test_dir = "testclassify"
for filename in os.listdir(test_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_dir, filename)

        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)  # 增加批次维度

            # 预测
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                class_idx = predicted.item()

            # 复制到相应类别文件夹
            class_folder = "photo" if class_idx == 0 else "art"
            output_path = os.path.join(output_dir, class_folder, filename)
            shutil.copy2(img_path, output_path)

            print(f"Image: {filename} -> Class: {class_folder}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("\nClassification completed! Results saved to:", output_dir)