import cv2
import numpy as np
import networkx as nx


def analyze_component_connectivity_fast(binary_image, components_data, output_path=None):
    """
    使用基于连通组件标记而非路径追踪的更快方法分析组件之间的连通性，连接边由组件方向决定。
    特殊情况：接地组件可以从各个方向连接。
    """

    # 创建二进制图像的副本以进行可视化
    h, w = binary_image.shape
    connectivity_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # 定义组件的侧面
    SIDES = ['top', 'right', 'bottom', 'left']

    # 创建组件掩模并确定方向
    component_masks = {}
    component_orientation = {}
    side_padded_masks = {}

    for comp_id, comp_data in components_data.items():
        # 基本组件掩码
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = comp_data['box']
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        component_masks[comp_id] = mask

        # 检查这是否是接地组件
        is_ground = comp_data['class_name'].lower() == 'ground'

        # 根据宽度与高度或特殊情况确定方向
        width = x2 - x1
        height = y2 - y1

        if is_ground:
            # 特殊情况：地线可以从任意一侧连接
            orientation = "any"
            active_sides = ['top', 'right', 'bottom', 'left']
        elif width > height:
            orientation = "horizontal"
            # 对于水平组件，我们主要使用左侧和右侧
            active_sides = ['left', 'right']
        else:
            orientation = "vertical"
            # 对于垂直组件，我们主要使用顶部和底部
            active_sides = ['top', 'bottom']

        component_orientation[comp_id] = {
            'orientation': orientation,
            'active_sides': active_sides,
            'is_ground': is_ground
        }

        # 创建特定于侧面的填充面罩，但仅适用于活动侧
        side_padded_masks[comp_id] = {}

        # 填充参数
        padding = 10

        # 根据方向或特殊情况为活动面创建遮罩
        for side in active_sides:
            if side == 'top':
                top_mask = np.zeros((h, w), dtype=np.uint8)
                top_x1 = max(0, x1 - padding)
                top_x2 = min(w, x2 + padding)
                top_y1 = max(0, y1 - padding)
                top_y2 = y1
                cv2.rectangle(top_mask, (top_x1, top_y1), (top_x2, top_y2), 255, -1)
                side_padded_masks[comp_id]['top'] = top_mask

            elif side == 'bottom':
                bottom_mask = np.zeros((h, w), dtype=np.uint8)
                bottom_x1 = max(0, x1 - padding)
                bottom_x2 = min(w, x2 + padding)
                bottom_y1 = y2
                bottom_y2 = min(h, y2 + padding)
                cv2.rectangle(bottom_mask, (bottom_x1, bottom_y1), (bottom_x2, bottom_y2), 255, -1)
                side_padded_masks[comp_id]['bottom'] = bottom_mask

            elif side == 'left':
                left_mask = np.zeros((h, w), dtype=np.uint8)
                left_x1 = max(0, x1 - padding)
                left_x2 = x1
                left_y1 = max(0, y1 - padding)
                left_y2 = min(h, y2 + padding)
                cv2.rectangle(left_mask, (left_x1, left_y1), (left_x2, left_y2), 255, -1)
                side_padded_masks[comp_id]['left'] = left_mask

            elif side == 'right':
                right_mask = np.zeros((h, w), dtype=np.uint8)
                right_x1 = x2
                right_x2 = min(w, x2 + padding)
                right_y1 = max(0, y1 - padding)
                right_y2 = min(h, y2 + padding)
                cv2.rectangle(right_mask, (right_x1, right_y1), (right_x2, right_y2), 255, -1)
                side_padded_masks[comp_id]['right'] = right_mask

    # 创建方向和活动侧的可视化
    orientation_viz = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        'top': (255, 0, 0),  # 蓝
        'right': (0, 255, 0),  # 绿
        'bottom': (0, 0, 255),  # 红
        'left': (255, 255, 0)  # 青
    }

    # 绘制按方向着色的组件框
    for comp_id, comp_data in components_data.items():
        x1, y1, x2, y2 = comp_data['box']
        orientation_info = component_orientation[comp_id]
        orientation = orientation_info['orientation']
        is_ground = orientation_info['is_ground']

        # 绘制元件盒
        if is_ground:
            # 地面组件获得特殊颜色
            color = (0, 255, 255)  # 黄
        else:
            color = (0, 0, 255) if orientation == 'vertical' else (255, 0, 0)

        cv2.rectangle(orientation_viz, (x1, y1), (x2, y2), color, 2)

        # 绘制活动边
        active_sides = orientation_info['active_sides']
        for side in active_sides:
            if side in side_padded_masks[comp_id]:
                mask = side_padded_masks[comp_id][side]
                orientation_viz[mask > 0] = colors[side]

    cv2.imwrite("component_orientation.png", orientation_viz)

    # 创建带标签的图像，其中每条电路走线都标有唯一的 ID
    circuit_traces = binary_image.copy()

    # 标记电路走线中每个连接的元件
    num_labels, labels = cv2.connectedComponents(circuit_traces)

    print(f"发现{num_labels - 1}单独的电路走线")

    # 创建标记轨迹的可视化
    trace_viz = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, num_labels):
        color = np.random.randint(50, 255, size=3).tolist()
        trace_viz[labels == i] = color
    cv2.imwrite("labeled_traces.png", trace_viz)

    # 查找哪些走线连接到哪些组件以及在哪一侧
    component_side_traces = {}
    for comp_id in side_padded_masks:
        component_side_traces[comp_id] = {}
        active_sides = component_orientation[comp_id]['active_sides']

        for side in active_sides:
            if side in side_padded_masks[comp_id]:
                component_side_traces[comp_id][side] = set()

                # 拿到这边的掩码
                side_mask = side_padded_masks[comp_id][side]

                # 检查哪些迹线与此侧相交
                for label_id in range(1, num_labels):
                    # 为该轨迹创建蒙版
                    trace_mask = (labels == label_id).astype(np.uint8) * 255

                    # 检查此走线是否与组件的侧面相交
                    intersection = cv2.bitwise_and(side_mask, trace_mask)
                    if np.any(intersection):
                        component_side_traces[comp_id][side].add(label_id)

    # 查找组件之间的连接点
    connectivity_graph = nx.MultiGraph()  # 使用MultiGraph支持多个连接

    # 将所有组件添加到图表中
    for comp_id in components_data:
        connectivity_graph.add_node(comp_id, **components_data[comp_id])

    # 跟踪详细的连接信息
    detailed_connections = []

    # 根据共享轨迹检查组件之间的连接
    for comp_id1 in component_side_traces:
        for comp_id2 in component_side_traces:
            if comp_id1 < comp_id2:  # 避免检查同一对两次
                # 检查两侧之间的连接
                active_sides1 = component_orientation[comp_id1]['active_sides']
                active_sides2 = component_orientation[comp_id2]['active_sides']

                for side1 in active_sides1:
                    if side1 not in component_side_traces[comp_id1]:
                        continue

                    for side2 in active_sides2:
                        if side2 not in component_side_traces[comp_id2]:
                            continue

                        # 获取每一侧的痕迹
                        traces1 = component_side_traces[comp_id1][side1]
                        traces2 = component_side_traces[comp_id2][side2]

                        # 查找共享痕迹
                        shared_traces = traces1.intersection(traces2)

                        if shared_traces:
                            # 组件连接在这些侧面
                            connection_info = {
                                'comp1': comp_id1,
                                'side1': side1,
                                'comp2': comp_id2,
                                'side2': side2,
                                'traces': list(shared_traces)
                            }
                            detailed_connections.append(connection_info)

    # 对接地连接进行滤波
    filtered_connections = filter_ground_connections(detailed_connections, components_data)

    # 根据过滤的连接创建连接图和可视化
    connectivity_graph = nx.MultiGraph()

    # 将所有组件添加到图表中
    for comp_id in components_data:
        connectivity_graph.add_node(comp_id, **components_data[comp_id])

    # 退货简化清单
    connection_points = []

    # 处理已过滤的连接
    for conn in filtered_connections:
        comp1 = conn['comp1']
        comp2 = conn['comp2']
        side1 = conn['side1']
        side2 = conn['side2']

        # 向图中添加带有边信息的边
        connectivity_graph.add_edge(
            comp1,
            comp2,
            side1=side1,
            side2=side2,
            traces=conn['traces']
        )

        # 添加到简单连接列表
        connection_points.append((comp1, comp2))

        # 可视化：获取组件中心和框信息
        box1 = components_data[comp1]['box']
        box2 = components_data[comp2]['box']

        # 根据边计算连接点
        conn_pt1 = get_connection_point(box1, side1)
        conn_pt2 = get_connection_point(box2, side2)

        # 用侧面颜色绘制连接
        cv2.line(connectivity_image, conn_pt1, conn_pt2, colors[side1], 2)

        # 在连接点处画小圆圈
        cv2.circle(connectivity_image, conn_pt1, 4, colors[side1], -1)
        cv2.circle(connectivity_image, conn_pt2, 4, colors[side2], -1)

        # 添加显示中点处连接边的文本
        mid_point = ((conn_pt1[0] + conn_pt2[0]) // 2, (conn_pt1[1] + conn_pt2[1]) // 2)

        # 接地连接专用标签
        is_ground1 = component_orientation[comp1].get('is_ground', False)
        is_ground2 = component_orientation[comp2].get('is_ground', False)

        if is_ground1 or is_ground2:
            conn_text = "GND"
        else:
            conn_text = f"{side1[0]}-{side2[0]}"  # 为了简洁，只使用首字母

        cv2.putText(connectivity_image, conn_text, mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 绘制带有方向信息的组件
    for comp_id, comp_data in components_data.items():
        x1, y1, x2, y2 = comp_data['box']
        center = comp_data['center']

        # 获取方向信息
        orientation_info = component_orientation[comp_id]
        orientation = orientation_info['orientation']
        is_ground = orientation_info.get('is_ground', False)

        # 绘制元件盒
        color = (0, 255, 0) if connectivity_graph.degree(comp_id) > 0 else (100, 100, 100)
        cv2.rectangle(connectivity_image, (x1, y1), (x2, y2), color, 2)

        # 添加方向文本
        if is_ground:
            orientation_text = "G"  # Ground
        else:
            orientation_text = "H" if orientation == "horizontal" else "V"

        cv2.putText(connectivity_image, orientation_text, (center[0] - 5, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加组件标签
        cv2.putText(connectivity_image, comp_id, (center[0] - 15, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 添加有关接地过滤的信息文本
    cv2.putText(connectivity_image, "每个组件的接地组件仅连接一次",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    # 保存连接可视化
    if output_path:
        cv2.imwrite(output_path, connectivity_image)

    # 对每个元件创建连接信息
    sorted_connections = {}
    for comp_id in components_data:
        sorted_connections[comp_id] = {}
        for neighbor in connectivity_graph.neighbors(comp_id):
            # 获取此组件与邻居之间的所有边
            edges = connectivity_graph.get_edge_data(comp_id, neighbor)
            # 存储每个连接的辅助信息
            sides = []
            for edge_key, edge_data in edges.items():
                # 查找哪一侧属于此组件
                if 'side1' in edge_data and comp_id < neighbor:
                    sides.append(edge_data['side1'])
                elif 'side2' in edge_data and comp_id > neighbor:
                    sides.append(edge_data['side2'])

            sorted_connections[comp_id][neighbor] = sides

    print(f"过滤后识别出 {len(filtered_connections)} 个连接")

    return connection_points, connectivity_graph, connectivity_image, sorted_connections, detailed_connections


def get_connection_point(box, side):
    """计算组件盒指定边上的一个点"""
    x1, y1, x2, y2 = box

    if side == 'top':
        return (x1 + x2) // 2, y1
    elif side == 'bottom':
        return (x1 + x2) // 2, y2
    elif side == 'left':
        return x1, (y1 + y2) // 2
    elif side == 'right':
        return x2, (y1 + y2) // 2
    else:
        return (x1 + x2) // 2, (y1 + y2) // 2  # 返回中心点


def filter_ground_connections(detailed_connections, components_data):
    """
    对连接进行后处理，以确保接地组件彼此之间仅连接一次，无论使用哪一侧。
    参数：
    detailed_connections：详细连接信息列表
    components_data：组件数据字典
    返回：
    filtered_connections：已筛选的连接列表
    """
    print("过滤多余的接地连接...")

    # 跟踪具有接地连接的唯一组件对
    ground_connections = {}
    non_ground_connections = []

    # 首先确定地面组件
    ground_components = set()
    for comp_id, comp_data in components_data.items():
        if comp_data['class_name'].lower() == 'ground':
            ground_components.add(comp_id)

    # 处理所有连接
    for conn in detailed_connections:
        comp1 = conn['comp1']
        comp2 = conn['comp2']

        # 检查任一组件是否接地
        if comp1 in ground_components or comp2 in ground_components:
            # 创建一致的密钥对
            pair = tuple(sorted([comp1, comp2]))

            if pair not in ground_connections:
                # 该接地元件对的第一个连接，保持
                ground_connections[pair] = conn
        else:
            # 非接地连接，请保留
            non_ground_connections.append(conn)

    # 合并已过滤的列表
    filtered_connections = non_ground_connections + list(ground_connections.values())

    # 打印统计数据
    removed = len(detailed_connections) - len(filtered_connections)
    print(f"移除了{removed}个冗余接地连接，保留{len(filtered_connections)}个总连接数")

    return filtered_connections
