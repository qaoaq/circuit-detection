# from ultralytics import YOLO
# import cv2
# import numpy as np


# def detect_circuit_combined_models(standard_model_path, pose_model_path, image_path, confidence_threshold=0.5):
#     """
#     Detect circuit components using both standard YOLOv8 and YOLOv8-pose models.
#     ALWAYS prioritizes standard model detections over pose model when there's overlap.
#     Prompts user to confirm directions for components missing directional data.
#
#     Args:
#         standard_model_path: Path to the standard YOLOv8 model (for all components)
#         pose_model_path: Path to the YOLOv8-pose model (for directional info)
#         image_path: Path to the circuit diagram image
#         confidence_threshold: Minimum confidence threshold for detections
#     """
#     print("\nDetecting circuit components...")
#
#     # 加载两个YOLO模型
#     standard_model = YOLO(standard_model_path)
#     pose_model = YOLO(pose_model_path)
#
#     # 加载照片
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: 找不到 {image_path}")
#         return None, None
#
#     # 复制照片
#     result_image = image.copy()
#
#     # 定义不同元件识别的颜色
#     colors = {
#         'voltage_source': (0, 0, 255),  # 红色
#         'controlled_voltage_source': (255, 0, 0),  # 蓝色
#         'current_source': (0, 255, 0),  # 绿色
#         'controlled_current_source': (0, 255, 255),  # 黄色
#         'resistor': (255, 0, 255),  # 洋红
#         'inductor': (255, 255, 0),  # 青色
#         'capacitor': (0, 165, 255),  # 橙色
#         'diode': (128, 0, 128),  # 紫色
#         'ground': (128, 128, 128)  # 灰色
#     }
#
#     # 创建检测元件信息的字典
#     detected_components = {}
#
#     # 跟踪标准模型检测到的元件范围
#     standard_boxes = []
#
#     # 记录方向元件是否需要人工判断方向
#     directional_components_needing_direction = []
#
#     # 方向元件分类
#     directional_classes = ['voltage_source', 'controlled_voltage_source',
#                            'current_source', 'controlled_current_source', 'diode']
#
#     print("检测所有元件")
#     standard_results = standard_model(image, conf=confidence_threshold)
#
#     # 处理结果，记录特性
#     for result in standard_results:
#         boxes = result.boxes
#         for i, box in enumerate(boxes):
#             class_id = int(box.cls)
#             class_name = standard_model.names[class_id]
#             confidence = float(box.conf)
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # 转成整数
#             center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
#
#             # 创建独特的元件ID
#             component_id = f"std_{class_name}_{i}"
#             detected_components[component_id] = {
#                 "class_name": class_name,
#                 "confidence": confidence,
#                 "box": (x1, y1, x2, y2),
#                 "center": center,
#                 "model": "standard",
#                 "direction": None
#             }
#             standard_boxes.append((x1, y1, x2, y2, class_name, component_id))
#
#             # 如果是方向元件，那么其需要方向标注
#             if class_name in directional_classes:
#                 directional_components_needing_direction.append(component_id)
#
#             # 得到元件对应颜色
#             color = colors.get(class_name, (0, 0, 0))
#
#             # 标出元件所在位置和标签
#             cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
#             label = f"{class_name}: {confidence:.2f}"
#             text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#             cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
#             cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#     print("检测方向信息")
#     pose_results = pose_model(image, conf=confidence_threshold)
#
#     # 处理方向信息
#     for result in pose_results:
#         boxes = result.boxes
#         keypoints = result.keypoints
#
#         for i, box in enumerate(boxes):
#             class_id = int(box.cls)
#             class_name = pose_model.names[class_id]
#             confidence = float(box.conf)
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#
#             # 检测方向检测结果中的元件本身是否被标准模型检测到
#             matched_component = None
#             matched_iou = 0
#
#             for box_info in standard_boxes:
#                 std_x1, std_y1, std_x2, std_y2, std_class, std_id = box_info
#
#                 # 计算四边位置
#                 x_left = max(x1, std_x1)
#                 y_top = max(y1, std_y1)
#                 x_right = min(x2, std_x2)
#                 y_bottom = min(y2, std_y2)
#
#                 # 如果二者面积重合超过百分之五十，则认为是同一个元件
#                 if x_right > x_left and y_bottom > y_top:
#                     intersection = (x_right - x_left) * (y_bottom - y_top)
#                     area1 = (x2 - x1) * (y2 - y1)
#                     area2 = (std_x2 - std_x1) * (std_y2 - std_y1)
#                     union = area1 + area2 - intersection
#                     iou = intersection / union
#                     if iou > 0.5 and iou > matched_iou:
#                         matched_component = std_id
#                         matched_iou = iou
#
#             # 如果有方向信息则记录
#             direction_info = None
#             if hasattr(keypoints, 'xy') and i < len(keypoints):
#                 kpts = keypoints[i].xy.cpu().numpy()[0]
#                 kpts_visible = keypoints[i].conf.cpu().numpy()[0] > 0.5
#
#                 if len(kpts) >= 2 and kpts_visible[0] and kpts_visible[1]:
#                     direction_info = {
#                         "start": (int(kpts[0][0]), int(kpts[0][1])),
#                         "end": (int(kpts[1][0]), int(kpts[1][1]))
#                     }
#
#             # 如果两个模型检测结果匹配
#             if matched_component:
#                 # 只用对以往的方向检测元件添加其方向，不用改变记录的元件种类
#                 if direction_info:
#                     detected_components[matched_component]["direction"] = direction_info
#
#                     # 从记录追踪的对应列表中删除，防止重复操作
#                     if matched_component in directional_components_needing_direction:
#                         directional_components_needing_direction.remove(matched_component)
#
#                     print(f"增加方向到该元件 {matched_component}")
#
#                     # 画出该元件的方向，红色为起始，绿色中终点
#                     cv2.circle(result_image, direction_info["start"], 5, (0, 255, 0), -1)
#                     cv2.circle(result_image, direction_info["end"], 5, (0, 0, 255), -1)
#                     cv2.arrowedLine(result_image,
#                                     direction_info["start"],
#                                     direction_info["end"],
#                                     (255, 255, 0), 2, tipLength=0.3)
#
#                     # 为电压源添加极性标签
#                     std_class = detected_components[matched_component]["class_name"]
#                     if 'voltage_source' in std_class:
#                         cv2.putText(result_image, "-",
#                                     (direction_info["start"][0] - 5, direction_info["start"][1] - 5),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                         cv2.putText(result_image, "+",
#                                     (direction_info["end"][0] - 5, direction_info["end"][1] - 5),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#                 # 若两个模型检测结果种类不同
#                 if detected_components[matched_component]["class_name"] != class_name:
#                     print(f"注意：标准模型检测到了 {detected_components[matched_component]['class_name']} " +
#                           f" 但姿势模型检测到了同一组件的 {class_name}。" +
#                           "使用标准模型分类。")
#
#             # 这是标准模型未检测到的新组件
#             else:
#                 # 添加相应信息
#                 if direction_info:
#                     component_id = f"pose_{class_name}_{i}"
#                     detected_components[component_id] = {
#                         "class_name": class_name,
#                         "confidence": confidence,
#                         "box": (x1, y1, x2, y2),
#                         "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
#                         "model": "pose",
#                         "direction": direction_info
#                     }
#
#                     color = colors.get(class_name, (0, 0, 0))
#                     cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
#                     label = f"{class_name}: {confidence:.2f} (pose)"
#                     text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#                     cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
#                     cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#
#                     cv2.circle(result_image, direction_info["start"], 5, (0, 255, 0), -1)
#                     cv2.circle(result_image, direction_info["end"], 5, (0, 0, 255), -1)
#                     cv2.arrowedLine(result_image,
#                                     direction_info["start"],
#                                     direction_info["end"],
#                                     (255, 255, 0), 2, tipLength=0.3)
#
#                     if 'voltage_source' in class_name:
#                         cv2.putText(result_image, "-",
#                                     (direction_info["start"][0] - 5, direction_info["start"][1] - 5),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                         cv2.putText(result_image, "+",
#                                     (direction_info["end"][0] - 5, direction_info["end"][1] - 5),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#                     print(f"从姿态模型增加新元件{component_id}(标准模型不识别)")
#
#     # 创建原始图像和结果图像并排的组合可视化
#     h, w = image.shape[:2]
#     combined_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
#     combined_image[:, :w] = image
#     combined_image[:, w:] = result_image
#     cv2.putText(combined_image, "Original Image", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#     cv2.putText(combined_image, "Standard Model Priority Detection", (w + 10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#     # 保存结果
#     cv2.imwrite("combined_model_results.jpg", combined_image)
#     cv2.imwrite("detection_result.jpg", result_image)
#
#     # 为需要的组件设置手动方向输入
#     if directional_components_needing_direction:
#         print("\n*** 需要方向信息 ***")
#         print(
#             f"发现{len(directional_components_needing_direction)}个没有方向信息的方向分量：")
#
#         # 创建结果图像的副本，用于可视化手动指示
#         manual_direction_image = result_image.copy()
#
#         for comp_id in directional_components_needing_direction:
#             comp_data = detected_components[comp_id]
#             comp_type = comp_data['class_name']
#             x1, y1, x2, y2 = comp_data['box']
#             center_x = (x1 + x2) // 2
#             center_y = (y1 + y2) // 2
#
#             # 确定组件是水平的还是垂直的
#             width = x2 - x1
#             height = y2 - y1
#             orientation = "horizontal" if width > height else "vertical"
#
#             print(f"\n元件{comp_id} ({comp_type}):")
#             print(f"方向为{orientation}.")
#
#             if 'voltage_source' in comp_type:
#                 if orientation == "horizontal":
#                     print("对于水平电压源，指定方向：")
#                     print("1. 从左到右 (左负极，右正极)")
#                     print("2. 从右到左 (右负极，左正极)")
#                     direction = input("选择1或2:").strip()
#
#                     if direction == '1':
#                         start_point = (x1 - width // 4, center_y)
#                         end_point = (x2 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从左到右")
#                     elif direction == '2':
#                         start_point = (x2 - width // 4, center_y)
#                         end_point = (x1 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从右到左")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#                 else:
#                     print("对于垂直电压源，请指定方向：")
#                     print("1. 从上到下 (上负极，下正极)")
#                     print("2. 从下到上 (下负极，上正极)")
#                     direction = input("选择1或2: ").strip()
#
#                     if direction == '1':
#                         start_point = (center_x, y1 + height // 4)
#                         end_point = (center_x, y2 - height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从上到下")
#                     elif direction == '2':
#                         start_point = (center_x, y2 - height // 4)
#                         end_point = (center_x, y1 + height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从下到上")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#
#             # For current sources and controlled current sources
#             elif 'current_source' in comp_type:
#                 if orientation == "horizontal":
#                     print("对于水平电流源，指定方向：")
#                     print("1. 从左到右 (左负极，右正极)")
#                     print("2. 从右到左 (右负极，左正极)")
#                     direction = input("选择1或2:").strip()
#
#                     if direction == '1':
#                         start_point = (x1 - width // 4, center_y)
#                         end_point = (x2 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从左到右")
#                     elif direction == '2':
#                         start_point = (x2 - width // 4, center_y)
#                         end_point = (x1 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从右到左")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#                 else:
#                     print("对于垂直电流源，请指定方向：")
#                     print("1. 从上到下 (上负极，下正极)")
#                     print("2. 从下到上 (下负极，上正极)")
#                     direction = input("选择1或2: ").strip()
#
#                     if direction == '1':
#                         start_point = (center_x, y1 + height // 4)
#                         end_point = (center_x, y2 - height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从上到下")
#                     elif direction == '2':
#                         start_point = (center_x, y2 - height // 4)
#                         end_point = (center_x, y1 + height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从下到上")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#
#             # For diodes
#             elif comp_type == 'diode':
#                 if orientation == "horizontal":
#                     print("对于水平二极管，指定方向：")
#                     print("1. 从左到右 (通过电流方向)")
#                     print("2. 从右到左 (通过电流方向)")
#                     direction = input("选择1或2:").strip()
#
#                     if direction == '1':
#                         start_point = (x1 - width // 4, center_y)
#                         end_point = (x2 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从左到右")
#                     elif direction == '2':
#                         start_point = (x2 - width // 4, center_y)
#                         end_point = (x1 + width // 4, center_y)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从右到左")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#                 else:
#                     print("对于垂直二极管，请指定方向：")
#                     print("1. 从上到下 (通过电流方向)")
#                     print("2. 从下到上 (通过电流方向)")
#                     direction = input("选择1或2: ").strip()
#
#                     if direction == '1':
#                         start_point = (center_x, y1 + height // 4)
#                         end_point = (center_x, y2 - height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从上到下")
#                     elif direction == '2':
#                         start_point = (center_x, y2 - height // 4)
#                         end_point = (center_x, y1 + height // 4)
#                         detected_components[comp_id]['direction'] = {
#                             'start': start_point,
#                             'end': end_point
#                         }
#                         print("方向设置：从下到上")
#                     else:
#                         print("选择无效，跳过此组件的方向。")
#                         continue
#
#             # 有有效的方向信息可以可视化
#             if 'direction' in detected_components[comp_id] and detected_components[comp_id]['direction']:
#                 direction_data = detected_components[comp_id]['direction']
#                 start_point = direction_data['start']
#                 end_point = direction_data['end']
#
#                 cv2.circle(manual_direction_image, start_point, 5, (0, 255, 0), -1)
#                 cv2.circle(manual_direction_image, end_point, 5, (0, 0, 255), -1)
#                 cv2.arrowedLine(manual_direction_image, start_point, end_point, (255, 255, 0), 2, tipLength=0.3)
#
#                 if 'voltage_source' in comp_type:
#                     cv2.putText(manual_direction_image, "+", (start_point[0] - 5, start_point[1] - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                     cv2.putText(manual_direction_image, "-", (end_point[0] - 5, end_point[1] - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#         cv2.imwrite("detection_with_manual_directions.jpg", manual_direction_image)
#         print("\n人工辅助标注保存在'detection_with_manual_directions.jpg'")
#
#     print("\nDetection Results:")
#     print(f"总检测元件: {len(detected_components)}")
#
#     standard_count = sum(1 for info in detected_components.values() if info['model'] == 'standard')
#     pose_count = sum(1 for info in detected_components.values() if info['model'] == 'pose')
#     print(f"从标准模型来看：{standard_count}")
#     print(f"从姿态模型{pose_count}")
#
#     directional_count = sum(1 for info in detected_components.values() if info['direction'] is not None)
#     print(f"具有方向信息的组件：{directional_count}")
#
#     for component_id, info in detected_components.items():
#         print(f"检测到{info['class_name']}置信度为{info['confidence']:.2f} (使用{info['model']}模型)")
#         if info["direction"]:
#             start = info["direction"]["start"]
#             end = info["direction"]["end"]
#             print(f" Direction: from {start} to {end}")
#
#     return detected_components, result_image


from ultralytics import YOLO
import cv2
import numpy as np


def detect_circuit_combined_models(standard_model_path, pose_model_path, image_path, confidence_threshold=0.5):
    """
    使用标准 YOLOv8 和 YOLOv8-pose 模型检测电路组件。当存在重叠时，始终优先使用标准模型检测，而不是姿势模型。提示用户确认缺少方向数据的组件的方向。

    参数：
    standard_model_path：标准 YOLOv8 模型的路径（用于所有组件）
    pose_model_path：YOLOv8-pose 模型的路径（用于方向信息）
    image_path：电路图图像的路径
    confidence_threshold：检测的最小置信度阈值
    """
    print("\n检测电路元件...")

    standard_model = YOLO(standard_model_path)
    pose_model = YOLO(pose_model_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：不能从{image_path}加载图像")
        return None, None

    result_image = image.copy()

    # 对不同元件类型定义不同颜色用于可视化
    colors = {
        'voltage_source': (0, 0, 255),  # 红
        'controlled_voltage_source': (255, 0, 0),  # 蓝
        'current_source': (0, 255, 0),  # 绿
        'controlled_current_source': (0, 255, 255),  # 黄
        'resistor': (255, 0, 255),  # 洋红
        'inductor': (255, 255, 0),  # 青
        'capacitor': (0, 165, 255),  # 橘
        'diode': (128, 0, 128),  # 紫
        'ground': (128, 128, 128)  # 灰
    }

    # 追踪检测元件的字典
    detected_components = {}

    # 跟踪从标准模型检测到的边界框
    standard_boxes = []

    # 列出需要用户输入的方向组件
    directional_components_needing_direction = []

    # 方向元件分类（需要方向信息）
    directional_classes = ['voltage_source', 'controlled_voltage_source',
                           'current_source', 'controlled_current_source', 'diode']

    print("运行标准模型检测...")
    standard_results = standard_model(image, conf=confidence_threshold)

    # 处理标准模型检测结果
    for result in standard_results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            class_name = standard_model.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # 创建独特的元件ID
            component_id = f"std_{class_name}_{i}"

            # 添加到检测元件列表中
            detected_components[component_id] = {
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": center,
                "model": "standard",
                "direction": None
            }
            standard_boxes.append((x1, y1, x2, y2, class_name, component_id))

            # 检查这是否是需要方向的定向组件
            if class_name in directional_classes:
                directional_components_needing_direction.append(component_id)

            color = colors.get(class_name, (0, 0, 0))

            # 画元件范围和标签
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    print("运行姿态模型检测方向信息...")
    pose_results = pose_model(image, conf=confidence_threshold)

    # 处理姿势模型结果以获取方向信息或新元件
    for result in pose_results:
        boxes = result.boxes
        keypoints = result.keypoints

        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            class_name = pose_model.names[class_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # 检查此姿势检测是否与任何标准模型检测重叠
            matched_component = None
            matched_iou = 0

            for box_info in standard_boxes:
                std_x1, std_y1, std_x2, std_y2, std_class, std_id = box_info

                # 计算并集交集
                x_left = max(x1, std_x1)
                y_top = max(y1, std_y1)
                x_right = min(x2, std_x2)
                y_bottom = min(y2, std_y2)

                # 检测是否重叠
                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (std_x2 - std_x1) * (std_y2 - std_y1)
                    union = area1 + area2 - intersection
                    iou = intersection / union

                    # 如果重叠足够大，认为是同一元件
                    if iou > 0.5 and iou > matched_iou:
                        matched_component = std_id
                        matched_iou = iou

            # 获得方向信息
            direction_info = None
            if hasattr(keypoints, 'xy') and i < len(keypoints):
                kpts = keypoints[i].xy.cpu().numpy()[0]
                kpts_visible = keypoints[i].conf.cpu().numpy()[0] > 0.5

                if len(kpts) >= 2 and kpts_visible[0] and kpts_visible[1]:
                    direction_info = {
                        "start": (int(kpts[0][0]), int(kpts[0][1])),
                        "end": (int(kpts[1][0]), int(kpts[1][1]))
                    }

            # 案例 1：此姿势检测与标准模型检测匹配
            if matched_component:
                # 创建临时图像来显示检测到的方向
                if direction_info:
                    component_type = detected_components[matched_component]["class_name"]

                    temp_dir_img = result_image.copy()
                    cv2.circle(temp_dir_img, direction_info["start"], 5, (0, 255, 0), -1)
                    cv2.circle(temp_dir_img, direction_info["end"], 5, (0, 0, 255), -1)
                    cv2.arrowedLine(temp_dir_img, direction_info["start"], direction_info["end"],
                                    (255, 255, 0), 2, tipLength=0.3)

                    window_name = f"Verify Direction: {component_type}"
                    cv2.imshow(window_name, temp_dir_img)
                    cv2.waitKey(500)

                    # 对于电压源，请标记端子以便清晰
                    if 'voltage_source' in component_type:
                        # 验证模型的方向判断是否正确
                        print(f"\n验证{component_type}的检测 (元件ID: {matched_component})")
                        print("模型检测到方向分量。请验证方向：")
                        print("1. 接受模型的方向判断")
                        print("2. 反转方向（交换起点/终点）")
                        print("3. 手动指定方向")

                        choice = input("输入你的选择(1/2/3): ").strip()

                        if choice == '1':
                            # 接受模型的判断
                            print("接受模型的判断")
                        elif choice == '2':
                            # 反转方向
                            print("反转方向")
                            direction_info = {
                                "start": direction_info["end"],
                                "end": direction_info["start"]
                            }
                        elif choice == '3':
                            # 手工标注方向
                            print("请手工指定方向：")
                            width = x2 - x1
                            height = y2 - y1
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            orientation = "horizontal" if width > height else "vertical"

                            # Follow the same manual direction input process as below
                            if orientation == "horizontal":
                                print("对于水平电压源，指定方向：")
                                print("1. 从左到右（左负极，右正极）")
                                print("2. 从右到左（右负极，左正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    # 从左到右
                                    start_point = (x1 + width // 4, center_y)
                                    end_point = (x2 - width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从左到右")
                                elif dir_choice == '2':
                                    start_point = (x2 - width // 4, center_y)
                                    end_point = (x1 + width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从右到左")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                            else:  # Vertical
                                print("对于垂直电压源，请指定方向：")
                                print("1. 从上到下（上负极，下正极）")
                                print("2. 从下到上（下负极，上正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    start_point = (center_x, y1 + height // 4)
                                    end_point = (center_x, y2 - height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从上到下")
                                elif dir_choice == '2':
                                    start_point = (center_x, y2 - height // 4)
                                    end_point = (center_x, y1 + height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从下到上")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                        else:
                            print("选择无效。使用模型的原始方向。")

                    # 对于电流源，询问是否检验
                    elif 'current_source' in component_type:
                        # 验证模型的方向判断是否正确
                        print(f"\n验证{component_type}的检测 (元件ID: {matched_component})")
                        print("模型检测到方向分量。请验证方向：")
                        print("1. 接受模型的方向判断")
                        print("2. 反转方向（交换起点/终点）")
                        print("3. 手动指定方向")

                        choice = input("输入你的选择(1/2/3): ").strip()

                        if choice == '1':
                            # 接受模型的判断
                            print("接受模型的判断")
                        elif choice == '2':
                            # 反转方向
                            print("反转方向")
                            direction_info = {
                                "start": direction_info["end"],
                                "end": direction_info["start"]
                            }
                        elif choice == '3':
                            # 手工标注方向
                            print("请手工指定方向：")
                            width = x2 - x1
                            height = y2 - y1
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            orientation = "horizontal" if width > height else "vertical"

                            # Follow the same manual direction input process as below
                            if orientation == "horizontal":
                                print("对于水平电压源，指定方向：")
                                print("1. 从左到右（左负极，右正极）")
                                print("2. 从右到左（右负极，左正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    # 从左到右
                                    start_point = (x1 + width // 4, center_y)
                                    end_point = (x2 - width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从左到右")
                                elif dir_choice == '2':
                                    start_point = (x2 - width // 4, center_y)
                                    end_point = (x1 + width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从右到左")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                            else:  # Vertical
                                print("对于垂直电压源，请指定方向：")
                                print("1. 从上到下（上负极，下正极）")
                                print("2. 从下到上（下负极，上正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    start_point = (center_x, y1 + height // 4)
                                    end_point = (center_x, y2 - height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从上到下")
                                elif dir_choice == '2':
                                    start_point = (center_x, y2 - height // 4)
                                    end_point = (center_x, y1 + height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从下到上")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                        else:
                            print("选择无效。使用模型的原始方向。")

                    # 对于二极管询问检验
                    elif component_type == 'diode':
                        # 验证模型的方向判断是否正确
                        print(f"\n验证{component_type}的检测 (元件ID: {matched_component})")
                        print("模型检测到方向分量。请验证方向：")
                        print("1. 接受模型的方向判断")
                        print("2. 反转方向（交换起点/终点）")
                        print("3. 手动指定方向")

                        choice = input("输入你的选择(1/2/3): ").strip()

                        if choice == '1':
                            # 接受模型的判断
                            print("接受模型的判断")
                        elif choice == '2':
                            # 反转方向
                            print("反转方向")
                            direction_info = {
                                "start": direction_info["end"],
                                "end": direction_info["start"]
                            }
                        elif choice == '3':
                            # 手工标注方向
                            print("请手工指定方向：")
                            width = x2 - x1
                            height = y2 - y1
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2

                            orientation = "horizontal" if width > height else "vertical"

                            # Follow the same manual direction input process as below
                            if orientation == "horizontal":
                                print("对于水平电压源，指定方向：")
                                print("1. 从左到右（左负极，右正极）")
                                print("2. 从右到左（右负极，左正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    # 从左到右
                                    start_point = (x1 + width // 4, center_y)
                                    end_point = (x2 - width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从左到右")
                                elif dir_choice == '2':
                                    start_point = (x2 - width // 4, center_y)
                                    end_point = (x1 + width // 4, center_y)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从右到左")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                            else:  # Vertical
                                print("对于垂直电压源，请指定方向：")
                                print("1. 从上到下（上负极，下正极）")
                                print("2. 从下到上（下负极，上正极）")
                                dir_choice = input("输入 1 或 2: ").strip()

                                if dir_choice == '1':
                                    start_point = (center_x, y1 + height // 4)
                                    end_point = (center_x, y2 - height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从上到下")
                                elif dir_choice == '2':
                                    start_point = (center_x, y2 - height // 4)
                                    end_point = (center_x, y1 + height // 4)
                                    direction_info = {
                                        'start': start_point,
                                        'end': end_point
                                    }
                                    print("  方向设置：从下到上")
                                else:
                                    print("  选择无效。使用模型的原始方向。")
                        else:
                            print("选择无效。使用模型的原始方向。")
                    cv2.destroyWindow(window_name)
                    detected_components[matched_component]["direction"] = direction_info

                    # 从需要指导的组件列表中删除
                    if matched_component in directional_components_needing_direction:
                        directional_components_needing_direction.remove(matched_component)

                    print(f"添加方向信息到元件 {matched_component}")

                    cv2.circle(result_image, direction_info["start"], 5, (0, 255, 0), -1)
                    cv2.circle(result_image, direction_info["end"], 5, (0, 0, 255), -1)
                    cv2.arrowedLine(result_image,
                                    direction_info["start"],
                                    direction_info["end"],
                                    (255, 255, 0), 2, tipLength=0.3)

                    std_class = detected_components[matched_component]["class_name"]
                    if 'voltage_source' in std_class:
                        cv2.putText(result_image, "-",
                                    (direction_info["start"][0] - 5, direction_info["start"][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(result_image, "+",
                                    (direction_info["end"][0] - 5, direction_info["end"][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 当姿势和标准模型检测到不同组件类型时进行记录
                if detected_components[matched_component]["class_name"] != class_name:
                    print(f"Note: Standard model detected {detected_components[matched_component]['class_name']} " +
                          f"but pose model detected {class_name} for the same component. " +
                          "Using standard model classification.")

            # 情况 2：这是标准模型未检测到的新组件
            else:
                # 仅当我们有方向信息时才添加
                if direction_info:
                    temp_dir_img = result_image.copy()
                    cv2.circle(temp_dir_img, direction_info["start"], 5, (0, 255, 0), -1)
                    cv2.circle(temp_dir_img, direction_info["end"], 5, (0, 0, 255), -1)
                    cv2.arrowedLine(temp_dir_img, direction_info["start"], direction_info["end"],
                                    (255, 255, 0), 2, tipLength=0.3)

                    window_name = f"Verify Direction: {class_name} (new pose detection)"
                    cv2.imshow(window_name, temp_dir_img)
                    cv2.waitKey(500)  # Show briefly

                    # 验证模型的方向判断是否正确
                    print(f"\n正在验证新姿态检测的 {class_name} 的方向")
                    print("模型检测到一个方向分量。请验证方向：")
                    print("1. 接受模型的方向判断")
                    print("2. 反转方向（交换起点/终点）")

                    choice = input("输入选择(1/2): ").strip()

                    if choice == '2':
                        print("反转方向")
                        direction_info = {
                            "start": direction_info["end"],
                            "end": direction_info["start"]
                        }
                    else:
                        print("接受模型方向判断")

                    cv2.destroyWindow(window_name)

                    component_id = f"pose_{class_name}_{i}"
                    detected_components[component_id] = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                        "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        "model": "pose",
                        "direction": direction_info
                    }

                    color = colors.get(class_name, (0, 0, 0))
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f} (pose)"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    cv2.circle(result_image, direction_info["start"], 5, (0, 255, 0), -1)
                    cv2.circle(result_image, direction_info["end"], 5, (0, 0, 255), -1)
                    cv2.arrowedLine(result_image,
                                    direction_info["start"],
                                    direction_info["end"],
                                    (255, 255, 0), 2, tipLength=0.3)

                    # 为电压源添加极性标签
                    if 'voltage_source' in class_name:
                        cv2.putText(result_image, "-",
                                    (direction_info["start"][0] - 5, direction_info["start"][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(result_image, "+",
                                    (direction_info["end"][0] - 5, direction_info["end"][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    print(f"增添新元件{component_id} 从姿态模型")

    # 创建原始图像和结果图像并排的组合可视化
    h, w = image.shape[:2]
    combined_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_image[:, :w] = image
    combined_image[:, w:] = result_image

    cv2.putText(combined_image, "Original Image", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_image, "Standard Model Priority Detection", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite("combined_model_results.jpg", combined_image)
    cv2.imwrite("detection_result.jpg", result_image)

    # 为需要的组件设置手动方向输入
    if directional_components_needing_direction:
        print("\n*** 需要方向信息 ***")
        print(
            f"发现{len(directional_components_needing_direction)}方向元件没有方向信息")

        manual_direction_image = result_image.copy()

        for comp_id in directional_components_needing_direction:
            comp_data = detected_components[comp_id]
            comp_type = comp_data['class_name']
            x1, y1, x2, y2 = comp_data['box']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 确定元件是水平的还是垂直的
            width = x2 - x1
            height = y2 - y1
            orientation = "horizontal" if width > height else "vertical"

            print(f"\n元件{comp_id} ({comp_type}):")
            print(f"元件为{orientation}.")

            # 对于电压源和受控电压源
            if 'voltage_source' in comp_type:
                if orientation == "horizontal":
                    print("对于水平电压源，请指定方向：")
                    print("1. 从左到右（左负极，右正极）")
                    print("2. 从右到左（右负极，左正极）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (x1 + width // 4, center_y)
                        end_point = (x2 - width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测从左到右")
                    elif direction == '2':
                        start_point = (x2 - width // 4, center_y)
                        end_point = (x1 + width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向设置：从右到左")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue
                else:  # Vertical orientation
                    print("对于垂直电压源，请指定方向：")
                    print("1. 从上到下（负极在上，正极在下）")
                    print("2. 从下到上（负极在下，正极在上）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (center_x, y1 + height // 4)
                        end_point = (center_x, y2 - height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从上到下")
                    elif direction == '2':
                        start_point = (center_x, y2 - height // 4)
                        end_point = (center_x, y1 + height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从下到上")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue

            # 对于电流源和受控电流源
            elif 'current_source' in comp_type:
                if orientation == "horizontal":
                    print("对于水平电压源，请指定方向：")
                    print("1. 从左到右（左负极，右正极）")
                    print("2. 从右到左（右负极，左正极）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (x1 + width // 4, center_y)
                        end_point = (x2 - width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测从左到右")
                    elif direction == '2':
                        start_point = (x2 - width // 4, center_y)
                        end_point = (x1 + width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向设置：从右到左")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue
                else:  # Vertical orientation
                    print("对于垂直电压源，请指定方向：")
                    print("1. 从上到下（负极在上，正极在下）")
                    print("2. 从下到上（负极在下，正极在上）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (center_x, y1 + height // 4)
                        end_point = (center_x, y2 - height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从上到下")
                    elif direction == '2':
                        start_point = (center_x, y2 - height // 4)
                        end_point = (center_x, y1 + height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从下到上")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue

            # 对二极管
            elif comp_type == 'diode':
                if orientation == "horizontal":
                    print("对于水平电压源，请指定方向：")
                    print("1. 从左到右（左负极，右正极）")
                    print("2. 从右到左（右负极，左正极）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (x1 + width // 4, center_y)
                        end_point = (x2 - width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测从左到右")
                    elif direction == '2':
                        start_point = (x2 - width // 4, center_y)
                        end_point = (x1 + width // 4, center_y)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向设置：从右到左")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue
                else:  # Vertical orientation
                    print("对于垂直电压源，请指定方向：")
                    print("1. 从上到下（负极在上，正极在下）")
                    print("2. 从下到上（负极在下，正极在上）")
                    direction = input("输入 1 或 2: ").strip()

                    if direction == '1':
                        start_point = (center_x, y1 + height // 4)
                        end_point = (center_x, y2 - height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从上到下")
                    elif direction == '2':
                        start_point = (center_x, y2 - height // 4)
                        end_point = (center_x, y1 + height // 4)
                        detected_components[comp_id]['direction'] = {
                            'start': start_point,
                            'end': end_point
                        }
                        print("  方向检测：从下到上")
                    else:
                        print("  选择无效。跳过此组件的方向。")
                        continue

            if 'direction' in detected_components[comp_id] and detected_components[comp_id]['direction']:
                direction_data = detected_components[comp_id]['direction']
                start_point = direction_data['start']
                end_point = direction_data['end']

                cv2.circle(manual_direction_image, start_point, 5, (0, 255, 0), -1)
                cv2.circle(manual_direction_image, end_point, 5, (0, 0, 255), -1)
                cv2.arrowedLine(manual_direction_image, start_point, end_point, (255, 255, 0), 2, tipLength=0.3)

                if 'voltage_source' in comp_type:
                    cv2.putText(manual_direction_image, "-", (start_point[0] - 5, start_point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(manual_direction_image, "+", (end_point[0] - 5, end_point[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imwrite("detection_with_manual_directions.jpg", manual_direction_image)
        print("\n带有人工标记的图像保存在'detection_with_manual_directions.jpg'")

    print("\n检测结果：")
    print(f"总的被检测到的元件{len(detected_components)}")

    standard_count = sum(1 for info in detected_components.values() if info['model'] == 'standard')
    pose_count = sum(1 for info in detected_components.values() if info['model'] == 'pose')
    print(f"从标准模型: {standard_count}")
    print(f"只从姿态模型: {pose_count}")

    directional_count = sum(1 for info in detected_components.values() if info['direction'] is not None)
    print(f"有方向信息的元件: {directional_count}")

    for component_id, info in detected_components.items():
        print(f"检测到{info['class_name']}有着置信度{info['confidence']:.2f} (使用{info['model']}模型)")
        if info["direction"]:
            start = info["direction"]["start"]
            end = info["direction"]["end"]
            print(f" 方向：从{start}到{end}")

    return detected_components, result_image