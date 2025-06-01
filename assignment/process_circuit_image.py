import cv2
import matplotlib.pyplot as plt
import numpy as np
from generate_binary_circuit_with_component_removal import generate_binary_circuit_with_component_removal

standard_model_path = r"E:\ppt\graphtheory\assignment\runs\circuit_components_detector3\weights\best.pt"
pose_model_path = r"E:\ppt\graphtheory\assignment\runs\circuit_components_detector_pose2\weights\best.pt"


def process_circuit_image(image_path):
    """步骤1：处理电路图像以生成二值图像"""
    # 固定半径按指定
    radius = 65

    # 生成去除组件的二进制图像
    print(f"生成细调整半径为{radius}的二进制电路图像（如果元件个数过多，可以把该值调小；反之，调大。根据检测结果准确度决定。）...")
    binary_image, connected_image, components_image, final_image, detected_components = \
        generate_binary_circuit_with_component_removal(
            image_path,
            "final_binary_circuit.png",
            radius,
            standard_model_path,
            pose_model_path
        )

    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(231)
    img = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 原始二进制线
    plt.subplot(232)
    plt.imshow(enhance_binary_image(binary_image), cmap='gray')
    plt.title('Original Binary Lines')
    plt.axis('off')

    # 连通二值图像
    plt.subplot(233)
    plt.imshow(enhance_binary_image(connected_image), cmap='gray')
    plt.title(f'Connected Binary (r={radius})')
    plt.axis('off')

    # 原件检测
    plt.subplot(234)
    plt.imshow(cv2.cvtColor(components_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Components ({len(detected_components)})')
    plt.axis('off')

    # 最终元件移去图像
    plt.subplot(235)
    plt.imshow(enhance_binary_image(final_image), cmap='gray')
    plt.title('Final Binary Circuit')
    plt.axis('off')

    # 元件掩码
    component_mask = cv2.imread("component_mask.png", 0)
    if component_mask is not None:
        plt.subplot(236)
        plt.imshow(component_mask, cmap='gray')
        plt.title('Component Mask')
        plt.axis('off')

    plt.savefig("circuit_processing_stages.png", dpi=300)

    # 保存最终的二进制图像以供下一步使用
    cv2.imwrite("final_binary_circuit.png", final_image)

    print(f"检测到的 {len(detected_components)} 个组件")

    # 保存组件数据以供下一步使用
    np.save("detected_components.npy", detected_components)

    return final_image, detected_components


def enhance_binary_image(binary_image):
    kernel = np.ones((3, 3), np.uint8)
    enhanced = cv2.dilate(binary_image, kernel, iterations=1)

    return enhanced
