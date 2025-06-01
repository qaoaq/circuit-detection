import cv2
import numpy as np
import os
from detect_circuit_combined_models import detect_circuit_combined_models
import math


def generate_binary_circuit_with_component_removal(image_path, output_path=None, radius=65,
                                                   standard_model_path=None, pose_model_path=None):
    """
    生成已移除元件区域的电路线路二值图像。

    参数：
    - image_path：输入图像路径
    - output_path：保存二值输出的路径
    - radius：端点连接半径
    - standard_model_path：标准 YOLO 模型路径
    - pose_model_path：YOLO-pose 模型路径

    返回：
    - binary_image：电路原始二值图像
    - connected_image：包含连接线的二值图像
    - components_image：显示检测到的元件的图像
    - final_image：移除元件后的二值图像
    -detected_components：检测到的元件数据字典
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    # 得到图像维度
    h, w = img.shape[:2]

    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)  # 边缘检测

    # 获取原始霍夫线
    min_line_length = 50
    max_line_gap = 50
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    if lines is None or len(lines) == 0:
        print("图片中没有线。")
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), img, np.zeros((h, w),
                                                                                                 dtype=np.uint8), {}

    # 用这些检测到的线创建一个图像
    binary_image = np.zeros((h, w), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(binary_image, (x1, y1), (x2, y2), 255, 1)

    # 保存初始二进制图像
    cv2.imwrite("original_binary.png", binary_image)

    # 提取这些线的端点
    endpoints = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))

    print(f"使用固定的精细化半径： {radius}")

    # 创建连通的二进制图像
    connected_image = create_connected_binary(binary_image, endpoints, radius)
    cv2.imwrite("connected_binary.png", connected_image)

    # 清理连接的图像
    cleaned_image = clean_binary_image(connected_image)
    cv2.imwrite("cleaned_binary.png", cleaned_image)

    # 确认模型路径存在
    if standard_model_path and pose_model_path and os.path.exists(standard_model_path) and os.path.exists(
            pose_model_path):
        try:
            print("使用 YOLO 模型检测组件...")
            detected_components, _ = detect_circuit_combined_models(standard_model_path, pose_model_path, image_path)
        except Exception as e:
            print(f"组件检测错误： {str(e)}")
            detected_components = {}
    else:
        print("未提供或未找到 YOLO 模型路径，跳过组件检测。")
        detected_components = {}

    # 创建原件可视化
    components_image = img.copy()
    component_mask = np.zeros((h, w), dtype=np.uint8)

    for comp_id, comp_data in detected_components.items():
        x1, y1, x2, y2 = comp_data['box']

        color = (0, 255, 0) if comp_data['model'] == 'standard' else (0, 0, 255)
        cv2.rectangle(components_image, (x1, y1), (x2, y2), color, 2)

        label = f"{comp_data['class_name']} ({comp_id})"
        cv2.putText(components_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        if comp_data['direction']:
            start_pt = comp_data['direction']['start']
            end_pt = comp_data['direction']['end']
            cv2.arrowedLine(components_image, start_pt, end_pt, (255, 0, 255), 2)

        cv2.rectangle(component_mask, (x1, y1), (x2, y2), 255, -1)

    cv2.imwrite("detected_components.png", components_image)
    cv2.imwrite("component_mask.png", component_mask)

    final_image = cleaned_image.copy()
    final_image[component_mask > 0] = 0

    # 保存最终图片
    if output_path:
        cv2.imwrite(output_path, final_image)

    # 创建增强版本来展示
    enhanced_binary = enhance_binary_image(binary_image)
    enhanced_connected = enhance_binary_image(cleaned_image)
    enhanced_final = enhance_binary_image(final_image)

    # 保存增强版本
    cv2.imwrite("enhanced_binary.png", enhanced_binary)
    cv2.imwrite("enhanced_connected.png", enhanced_connected)
    cv2.imwrite("enhanced_final.png", enhanced_final)

    return binary_image, cleaned_image, components_image, final_image, detected_components


def create_connected_binary(binary_image, endpoints, radius):
    """创建具有连接端点的二进制图像"""
    h, w = binary_image.shape[:2]
    result = binary_image.copy()

    # 创建基本端点可视化（可选，用于调试）
    endpoint_viz = np.zeros((h, w), dtype=np.uint8)
    for x, y in endpoints:
        cv2.circle(endpoint_viz, (x, y), 2, 255, -1)

    cv2.imwrite("endpoints_viz.png", endpoint_viz)

    # 查找半径内的连接
    for i in range(len(endpoints)):
        x1, y1 = endpoints[i]
        for j in range(i + 1, len(endpoints)):
            x2, y2 = endpoints[j]
            # 检查端点是否属于不同的线段
            if (i // 2) != (j // 2):  # 不同线段
                # 计算距离
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist <= radius:
                    # 画一条线连接这些端点
                    cv2.line(result, (x1, y1), (x2, y2), 255, 1)

    return result


def clean_binary_image(binary_image, min_line_length=15):
    """通过去除孤立的小线段清理二值图像"""
    """
        清理二值图像中孤立的小线段或噪声，保留满足最小长度要求的线段。
        通过检测图像中的轮廓，计算每个轮廓的近似长度，并滤除长度小于设定阈值的轮廓，最终生成干净的二值图像。
    """
    # 查找所有轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(binary_image)

    for contour in contours:
        # 近似轮廓
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, False)

        # 计算轮廓的总长度
        total_length = 0
        for i in range(len(approx) - 1):
            x1, y1 = approx[i][0]
            x2, y2 = approx[i + 1][0]
            total_length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 仅保留长度大于最小长度的轮廓
        if total_length >= min_line_length:
            cv2.drawContours(cleaned, [contour], 0, 255, 1)

    return cleaned


def enhance_binary_image(binary_image):
    """增强二值图像，使线条更加清晰"""
    """
        使用形态学膨胀操作增强二值图像中线条的可见性。
        此操作会扩展图像中的白色区域（前景），填充细小的断点或间隙，
        使线条更粗、更连续，从而在视觉上更清晰。
        此功能适用于需要提升二值图像可视化效果的场景，
        例如显示或初步预处理。
    """
    # 为了显示目的，创建一个稍微扩大的版本
    kernel = np.ones((3, 3), np.uint8)
    enhanced = cv2.dilate(binary_image, kernel, iterations=1)

    return enhanced
