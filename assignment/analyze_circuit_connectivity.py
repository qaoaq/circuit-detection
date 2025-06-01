import os
import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from analyse_circuit_values import analyze_circuit_values
from analyze_component_connectivity_fast import analyze_component_connectivity_fast


def analyze_circuit_connectivity(image_path, final_image, detected_components):
    """步骤2：使用二值图像分析电路的连通性"""
    try:
        binary_image = final_image
        print("分析元件连通性...")
        connections, connectivity_graph, connectivity_image, sorted_connections, detailed_connections = \
            analyze_component_connectivity_fast(
                binary_image,
                detected_components,
                "component_connectivity_fast.png"
            )

        # 创建示意图表示 - 更新以传递detailed_connections
        print("创建电路原理图...")
        circuit_graph = create_circuit_schematic(
            detected_components,
            connections,
            detailed_connections,
            "circuit_schematic.png"
        )

        plt.figure(figsize=(15, 10))

        plt.subplot(221)
        img = cv2.imread(image_path) if os.path.exists(image_path) else np.zeros((400, 400, 3), dtype=np.uint8)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(222)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Circuit')
        plt.axis('off')

        plt.subplot(223)
        plt.imshow(cv2.cvtColor(connectivity_image, cv2.COLOR_BGR2RGB))
        plt.title('Component Connectivity')
        plt.axis('off')

        plt.subplot(224)
        schematic_img = cv2.imread("circuit_schematic.png")
        if schematic_img is not None:
            plt.imshow(cv2.cvtColor(schematic_img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(np.zeros_like(img))
        plt.title('Circuit Schematic')
        plt.axis('off')

        plt.savefig("connectivity_analysis_results.png", dpi=300)

        print("\nComponent Connections:")
        for comp_id, connected_comps in sorted_connections.items():
            if connected_comps:
                print(
                    f" {comp_id} ({detected_components[comp_id]['class_name']})被连接于:{', '.join(connected_comps)}")
            else:
                print(f" {comp_id} ({detected_components[comp_id]['class_name']}) 没有连接")

        print(f"\n发现{len(connections)}个连接在{len(detected_components)}个元件中")

        results = analyze_circuit_values(connectivity_graph, detected_components, detailed_connections)

        plt.show()

    except Exception as e:
        print(f"连通性分析期间发生错误： {str(e)}")
        import traceback
        traceback.print_exc()


def create_circuit_schematic(components_data, connections, detailed_connections=None,
                             output_path='circuit_schematic.png'):
    """
    使用 NetworkX 和 matplotlib 创建简化的电路原理图，其连接边由元件方向决定。
    """
    G = nx.Graph()

    # 添加节点（组件）
    for comp_id, comp_data in components_data.items():
        G.add_node(comp_id, **comp_data)

    # 添加边（连接）
    for comp1_id, comp2_id in connections:
        G.add_edge(comp1_id, comp2_id)

    # 创建一个图形
    plt.figure(figsize=(12, 10))

    # 为组件创建位置（使用 spring 布局）
    pos = nx.spring_layout(G, seed=42)

    # 画元件
    component_types = {}
    for comp_id, comp_data in components_data.items():
        component_types[comp_id] = comp_data['class_name']

    # 获取用于着色的独特组件类型
    unique_types = set(component_types.values())
    color_map = {}
    for i, t in enumerate(unique_types):
        color_map[t] = plt.cm.tab20(i / len(unique_types))

    # 根据组件类型绘制具有颜色的节点
    for comp_type in unique_types:
        node_list = [node for node in G.nodes() if component_types[node] == comp_type]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=node_list,
                               node_color=[color_map[comp_type]],
                               node_size=500,
                               alpha=0.8)

    # 如果可用，绘制带有连接信息的边缘
    if detailed_connections:
        # 创建字典以将组件对映射到它们的连接详细信息
        connection_details = {}
        for conn in detailed_connections:
            comp1 = conn['comp1']
            comp2 = conn['comp2']
            side1 = conn['side1']
            side2 = conn['side2']

            # 创建排序对键
            pair = tuple(sorted([comp1, comp2]))
            if pair not in connection_details:
                connection_details[pair] = []

            connection_details[pair].append((side1, side2))

        # 用标签绘制边
        for (comp1, comp2), details in connection_details.items():
            # 将详细信息转换为字符串标签
            sides_str = ", ".join([f"{s1}-{s2}" for s1, s2 in details])

            # 画边
            nx.draw_networkx_edges(G, pos,
                                   edgelist=[(comp1, comp2)],
                                   width=1.5, alpha=0.7)

            # 增加边的标签
            edge_label = {(comp1, comp2): sides_str}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_label, font_size=7)
    else:
        # 绘制简单的边缘
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)

    # 添加节点标签（组件 ID）
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")

    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[t], markersize=10,
                                  label=t) for t in unique_types]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('Circuit Schematic')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    return G