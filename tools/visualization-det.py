import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models, transforms
from PIL import Image

# 1. 加载预训练的 Faster R-CNN 模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # 加载预训练的模型
model.eval()  # 设置模型为评估模式


# 2. 加载和预处理图像
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def transform_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # 增加 batch 维度


# 3. 可视化目标检测结果
def plot_bboxes(image, boxes, labels, scores, class_names, threshold=0.5):
    """
    在图像上绘制边界框，类别标签和置信度
    :param image: 输入图像
    :param boxes: 目标检测框 (num_boxes, 4)
    :param labels: 目标的分类标签 (num_boxes,)
    :param scores: 每个框的置信度 (num_boxes,)
    :param class_names: 类别名称列表
    :param threshold: 置信度阈值，低于此阈值的框会被过滤
    """
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]
        score = scores[i]

        # 如果置信度低于阈值，则跳过
        if score < threshold:
            continue

        # 计算边界框的坐标
        x_min, y_min, x_max, y_max = box

        # 确保坐标是整数类型
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # 绿色框

        # 显示类别标签和置信度
        label_name = class_names[label]  # 获取类别名称
        text = f'{label_name}: {score:.2f}'  # 文本标签，包含类别和置信度

        # 在框上方显示文本
        cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


# 4. 目标检测主流程
def detect_objects(image_path, model, class_names):
    # 加载和预处理图像
    image = load_image(image_path)
    transformed_image = transform_image(image)

    # 将图像输入模型
    with torch.no_grad():
        prediction = model(transformed_image)  # 获取模型输出

    # 解析输出
    boxes = prediction[0]['boxes'].cpu().numpy()  # 获取边界框
    labels = prediction[0]['labels'].cpu().numpy()  # 获取标签
    scores = prediction[0]['scores'].cpu().numpy()  # 获取置信度

    # 将原图转换为可视化所用的格式
    image = np.array(image)

    # 绘制边界框、类别标签和置信度
    output_image = plot_bboxes(image, boxes, labels, scores, class_names, threshold=0.5)

    # 显示检测结果
    save_path = './img/2007_000925_visualization.jpg' # 替换为你的图像路径
    cv2.imwrite(save_path, image)
    print(f"save: {save_path}")
    plt.imshow(output_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


# 5. 设置 VOC07+12 类别名称
class_names = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'tvmonitor', 'sofa', 'train',
    'sheep'
]

# 6. 输入图像路径并执行检测
image_path = './img/2007_000925.jpg'  # 替换为你的图像路径
detect_objects(image_path, model, class_names)
