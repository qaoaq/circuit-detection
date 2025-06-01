import networkx as nx
from matplotlib import pyplot as plt

from analyze_component_connectivity_fast import get_connection_point
from convert_to_directed_graph import create_node_based_circuit_graph
from perform_graph_based_circuit_analysis import perform_graph_based_circuit_analysis, visualize_circuit_with_values


def analyze_circuit_values(connectivity_graph, components_data, detailed_connections):
    """
    分析电路值以计算元件间的电流和电压。
    此函数允许手动输入元件值并处理受控源。

    参数：
    connectivity_graph：表示元件连接的图表
    components_data：元件数据字典
    detailed_connections：详细连接信息列表

    返回：
    results：包含计算出的电流和电压的字典
    """
    print("\n=== 电路值分析 ===")

    print("\n创建基于节点的电路表示...")
    node_based_graph, nodes_mapping, component_nodes = create_node_based_circuit_graph(
        components_data,
        detailed_connections
    )

    # 创建简化的组件列表以供显示
    component_info = []
    for comp_id, comp_data in components_data.items():
        component_info.append({
            'id': comp_id,
            'type': comp_data['class_name'],
            'neighbors': list(connectivity_graph.neighbors(comp_id))
        })

    # 显示组件信息
    print("\nComponent Information:")
    for comp in component_info:
        conn_str = ", ".join(comp['neighbors']) if comp['neighbors'] else "None"
        print(f"  {comp['id']} ({comp['type']}) - Connected to: {conn_str}")

    # 创建用于电路分析的有向图
    circuit_graph = nx.DiGraph()

    # 跟踪组件值和控制关系
    component_values = {}
    controlled_sources = {}
    component_directions = {}  # 跟踪元件的方向

    # 将组件节点添加到电路图
    for comp_id, comp_data in components_data.items():
        circuit_graph.add_node(comp_id, component_type=comp_data['class_name'])

    # 从基于节点的图中添加电路节点
    for node_id, node_data in node_based_graph['nodes'].items():
        circuit_graph.add_node(node_id, type='node')

        # 将组件连接到其节点
        for comp_id in node_data['connected_components']:
            # 找到此组件连接到此节点的适当边
            if comp_id in component_nodes:
                for side, connected_node in component_nodes[comp_id].items():
                    if connected_node == node_id:
                        circuit_graph.add_edge(comp_id, node_id, side=side)

    # 过程组件值（手动输入）
    print("\n输入组件值：")

    # 首先处理受控源以确保它们可用于其他组件
    for comp_id, comp_data in sorted([(cid, cdata) for cid, cdata in components_data.items()
                                      if 'controlled' in cdata['class_name'].lower()],
                                     key=lambda x: x[0]):
        comp_type = comp_data['class_name']

        if 'controlled_current_source' in comp_type.lower():
            # 对于受控电流源
            print(f" 受控电流源{comp_id}:")

            # 列出潜在的控制组件（除此以外）
            available_components = [c_id for c_id in components_data.keys() if c_id != comp_id]
            print("    可用组件： " + ", ".join(available_components))

            # 获取控制组件
            controlling_comp = input("    控制元件 ID: ").strip()
            while controlling_comp not in available_components and controlling_comp:
                print("    组件无效，请从可用组件中选择。")
                controlling_comp = input("    控制元件 ID: ").strip()

            if not controlling_comp:
                # 如果没有指定控制组件，则默认行为
                print("    未指定控制组件。使用默认的 0.01A 恒流源。")
                component_values[comp_id] = 0.01
            else:
                # 确定控制类型（电压或电流）
                control_type = input("     控制类型（v 代表电压，i 代表电流）：").strip().lower()
                control_type = 'voltage' if control_type == 'v' else 'current'

                # 获取跨导/电流增益
                gain = input(f"    {' 跨导(A/V)' if control_type == 'voltage' else '电流增益'}: ")

                try:
                    gain_value = float(gain) if gain.strip() else (0.01 if control_type == 'voltage' else 2.0)

                    # 对于VCCS，确定控制组件的参考方向
                    if control_type == 'voltage':
                        # 检查控制组件是水平还是垂直
                        ctrl_orientation = "horizontal"  # 默认
                        if 'box' in components_data.get(controlling_comp, {}):
                            x1, y1, x2, y2 = components_data[controlling_comp]['box']
                            ctrl_orientation = "horizontal" if (x2 - x1) > (y2 - y1) else "vertical"

                        # 问参考方向
                        if ctrl_orientation == "horizontal":
                            dir_choice = input(
                                f"     指定 {controlling_comp} 的参考方向（1=从左到右（左负极右正极），2=从右到左（右负极左正极））：")
                            left_to_right = dir_choice != '2'

                            # 在 component_directions 中记录组件端子
                            if controlling_comp in component_nodes and len(component_nodes[controlling_comp]) >= 2:
                                # 尝试识别左右终端
                                left_node = right_node = None
                                for s, n in component_nodes[controlling_comp].items():
                                    if s == 'left':
                                        left_node = n
                                    elif s == 'right':
                                        right_node = n

                                if left_node and right_node:
                                    if left_to_right:
                                        # 从左到右：左边是负极，右边是正极
                                        component_directions[controlling_comp] = (left_node, right_node)
                                    else:
                                        # 从右到左：右边是负极，左边是正极
                                        component_directions[controlling_comp] = (right_node, left_node)
                        else:  # vertical
                            dir_choice = input(
                                f"     指定 {controlling_comp} 的参考方向（1=从上到下（上负极下正极），2=从下到上（下负极上正极））：")
                            top_to_bottom = dir_choice != '2'

                            # 在 component_directions 中记录组件端子
                            if controlling_comp in component_nodes and len(component_nodes[controlling_comp]) >= 2:
                                # 尝试识别顶部和底部端子
                                top_node = bottom_node = None
                                for s, n in component_nodes[controlling_comp].items():
                                    if s == 'top':
                                        top_node = n
                                    elif s == 'bottom':
                                        bottom_node = n

                                if top_node and bottom_node:
                                    if top_to_bottom:
                                        # 从上到下：顶部为负极，底部为正极
                                        component_directions[controlling_comp] = (top_node, bottom_node)
                                    else:
                                        # 从下到上：底部为负极，顶部为正极
                                        component_directions[controlling_comp] = (bottom_node, top_node)

                    controlled_sources[comp_id] = {
                        'controlling_component': controlling_comp,
                        'control_type': control_type,
                        'gain': gain_value
                    }

                    # 如果有的话，添加参考方向
                    if controlling_comp in component_directions:
                        neg_node, pos_node = component_directions[controlling_comp]
                        controlled_sources[comp_id]['reference'] = {
                            'negative': neg_node,
                            'positive': pos_node
                        }
                        print(f"    使用参考方向： {neg_node}(-) to {pos_node}(+)")

                except ValueError:
                    default_gain = 0.01 if control_type == 'voltage' else 2.0
                    print(f"    无效增益，则使用{default_gain}")
                    controlled_sources[comp_id] = {
                        'controlling_component': controlling_comp,
                        'control_type': control_type,
                        'gain': default_gain
                    }

        elif 'controlled_voltage_source' in comp_type.lower():
            # 对于受控电压源
            print(f"  受控电压源 {comp_id}:")
            available_components = [c_id for c_id in components_data.keys() if c_id != comp_id]
            print("    有效元件： " + ", ".join(available_components))
            controlling_comp = input("    控制元件 ID: ").strip()
            while controlling_comp not in available_components and controlling_comp:
                print("    组件无效。请从可用组件中选择。")
                controlling_comp = input("    控制元件: ").strip()
            if not controlling_comp:
                print("    未指定控制元件。使用默认的5V恒流源。")
                component_values[comp_id] = 5.0
            else:
                # 添加控制类型选择（电压或电流）
                control_type = input("    控制类型（v 代表电压，i 代表电流）：").strip().lower()
                control_type = 'voltage' if control_type == 'v' else 'current'
                # 根据控制类型提供不同的增益提示
                if control_type == 'voltage':
                    gain_prompt = "    增益因子（例如，2表示2倍电压放大）："
                else:
                    gain_prompt = "    跨阻增益 (V/A)："
                gain = input(gain_prompt)
                try:
                    gain_value = float(gain) if gain.strip() else (2.0 if control_type == 'voltage' else 1000.0)
                    ctrl_orientation = "horizontal"
                    if 'box' in components_data.get(controlling_comp, {}):
                        x1, y1, x2, y2 = components_data[controlling_comp]['box']
                        ctrl_orientation = "horizontal" if (x2 - x1) > (y2 - y1) else "vertical"
                        # 问参考方向
                        if ctrl_orientation == "horizontal":
                            dir_choice = input(
                                f"     指定 {controlling_comp} 的参考方向（1=从左到右（左负极右正极），2=从右到左（右负极左正极））：")
                            left_to_right = dir_choice != '2'
                            # 在 component_directions 中记录组件端子
                            if controlling_comp in component_nodes and len(component_nodes[controlling_comp]) >= 2:
                                # 尝试识别左右终端
                                left_node = right_node = None
                                for s, n in component_nodes[controlling_comp].items():
                                    if s == 'left':
                                        left_node = n
                                    elif s == 'right':
                                        right_node = n
                                if left_node and right_node:
                                    if left_to_right:
                                        # 从左到右：左边是负极，右边是正极
                                        component_directions[controlling_comp] = (left_node, right_node)
                                    else:
                                        # 从右到左：右边是负极，左边是正极
                                        component_directions[controlling_comp] = (right_node, left_node)
                        else:  # vertical
                            dir_choice = input(
                                f"     指定 {controlling_comp} 的参考方向（1=从上到下（上负极下正极），2=从下到上（下负极上正极））：")
                            top_to_bottom = dir_choice != '2'
                            # 在 component_directions 中记录组件端子
                            if controlling_comp in component_nodes and len(component_nodes[controlling_comp]) >= 2:
                                # 尝试识别顶部和底部端子
                                top_node = bottom_node = None
                                for s, n in component_nodes[controlling_comp].items():
                                    if s == 'top':
                                        top_node = n
                                    elif s == 'bottom':
                                        bottom_node = n
                                if top_node and bottom_node:
                                    if top_to_bottom:
                                        # 从上到下：顶部为负极，底部为正极
                                        component_directions[controlling_comp] = (top_node, bottom_node)
                                    else:
                                        # 从下到上：底部为负极，顶部为正极
                                        component_directions[controlling_comp] = (bottom_node, top_node)
                    controlled_sources[comp_id] = {
                        'controlling_component': controlling_comp,
                        'control_type': control_type,  # Now using the selected control type
                        'gain': gain_value
                    }

                    # 增加参考方向
                    if controlling_comp in component_directions:
                        neg_node, pos_node = component_directions[controlling_comp]
                        controlled_sources[comp_id]['reference'] = {
                            'negative': neg_node,
                            'positive': pos_node
                        }
                        print(f"    使用以下参考方向 {neg_node}(-) to {pos_node}(+)")
                        print(f"    控制类型：{control_type}，增益：{gain_value}")

                except ValueError:
                    default_gain = 2.0 if control_type == 'voltage' else 1000.0
                    print(f"    无效增益，使用默认值 {default_gain}")
                    controlled_sources[comp_id] = {
                        'controlling_component': controlling_comp,
                        'control_type': control_type,  # Now using the selected control type
                        'gain': default_gain
                    }

    # 现在处理其他组件
    for comp_id, comp_data in sorted([(cid, cdata) for cid, cdata in components_data.items()
                                      if 'controlled' not in cdata['class_name'].lower()],
                                     key=lambda x: x[0]):
        comp_type = comp_data['class_name']

        if comp_type == 'resistor':
            value = input(f"  电阻{comp_id}值是多少欧姆: ")
            if value.strip():
                try:
                    component_values[comp_id] = float(value)
                except ValueError:
                    print("    无效值，使用默认的1000欧姆")
                    component_values[comp_id] = 1000.0
            else:
                component_values[comp_id] = 1000.0

        elif comp_type == 'voltage_source':
            value = input(f"  电压源{comp_id}值是多少V ")
            if value.strip():
                try:
                    component_values[comp_id] = float(value)
                except ValueError:
                    print("    无效值，使用默认的5V")
                    component_values[comp_id] = 5.0
            else:
                component_values[comp_id] = 5.0

            # 对于具有方向信息的电压源，识别端子
            if 'direction' in comp_data and comp_id in component_nodes and len(component_nodes[comp_id]) >= 2:
                print(f"  对于具有方向信息的电压源 {comp_id}：")
                print(f"    方向表示电流流动")
                print(f"    起点 = 负极 (-) 端子，终点 = 正极 (+) 端子")

                direction = comp_data['direction']
                start_pt = direction.get('start')  # 负极
                end_pt = direction.get('end')  # 正极

                if start_pt and end_pt:
                    neg_node = None
                    pos_node = None
                    min_neg_dist = float('inf')
                    min_pos_dist = float('inf')

                    # 找到最接近起始（负极）点的节点
                    for side, node_id in component_nodes[comp_id].items():
                        terminal_pos = get_connection_point(comp_data['box'], side)

                        # 距起始点（负极）的距离
                        neg_dist = ((terminal_pos[0] - start_pt[0]) ** 2 +
                                    (terminal_pos[1] - start_pt[1]) ** 2) ** 0.5

                        # 距起始点（正极）的距离
                        pos_dist = ((terminal_pos[0] - end_pt[0]) ** 2 +
                                    (terminal_pos[1] - end_pt[1]) ** 2) ** 0.5

                        if neg_dist < min_neg_dist:
                            min_neg_dist = neg_dist
                            neg_node = node_id

                        if pos_dist < min_pos_dist:
                            min_pos_dist = pos_dist
                            pos_node = node_id

                    if neg_node and pos_node:
                        component_directions[comp_id] = (neg_node, pos_node)
                        print(f"    已识别的终端：{neg_node}(-)至{pos_node}(+)")

        elif comp_type == 'current_source':
            value = input(f"  电流源{comp_id}值为多少A： ")
            if value.strip():
                try:
                    component_values[comp_id] = float(value)
                except ValueError:
                    print("    无效值，使用默认的0.01A")
                    component_values[comp_id] = 0.01
            else:
                component_values[comp_id] = 0.01

        elif comp_type == 'capacitor':
            value = input(f"  电容{comp_id}值为多少F")
            if value.strip():
                try:
                    component_values[comp_id] = float(value)
                except ValueError:
                    print("    无效值，使用默认的1e-6F")
                    component_values[comp_id] = 1e-6
            else:
                component_values[comp_id] = 1e-6

        elif comp_type == 'inductor':
            value = input(f"  电感{comp_id}值为多少H ")
            if value.strip():
                try:
                    component_values[comp_id] = float(value)
                except ValueError:
                    print("    无效值，使用默认的0.001H")
                    component_values[comp_id] = 1e-3
            else:
                component_values[comp_id] = 1e-3

    # 识别地面/参考节点
    ground_nodes = []

    # 首先检查明确的地面组件
    for comp_id, comp_data in components_data.items():
        if comp_data['class_name'].lower() == 'ground':
            # 查找连接到地面组件的节点
            if comp_id in component_nodes:
                for side, node_id in component_nodes[comp_id].items():
                    ground_nodes.append(node_id)
                    print(f"Found ground component {comp_id} connected to node {node_id}")

    # 如果没有发现明确的接地，则使用电压源负极
    if not ground_nodes:
        for comp_id in component_directions:
            comp_type = components_data.get(comp_id, {}).get('class_name', '').lower()

            if 'voltage_source' in comp_type:
                # 使用第一个已识别的节点（按照惯例为负极）
                neg_node, _ = component_directions[comp_id]
                ground_nodes.append(neg_node)
                print(f"用电压源{comp_id}负极端点(节点{neg_node})作为参考点")
                break

        # 如果仍然没有合适的理由，则采用其他策略
        if not ground_nodes:
            # 寻找具有方向信息的电压源
            for comp_id, comp_data in components_data.items():
                if 'voltage_source' in comp_data.get('class_name', '').lower() and 'direction' in comp_data:
                    direction = comp_data['direction']
                    start_pt = direction.get('start')

                    if start_pt and comp_id in component_nodes:
                        # 找到最接近起点的终端（负极）
                        closest_node = None
                        min_dist = float('inf')

                        for side, node_id in component_nodes[comp_id].items():
                            terminal_pos = get_connection_point(comp_data['box'], side)
                            dist = ((terminal_pos[0] - start_pt[0]) ** 2 +
                                    (terminal_pos[1] - start_pt[1]) ** 2) ** 0.5

                            if dist < min_dist:
                                min_dist = dist
                                closest_node = node_id

                        if closest_node:
                            ground_nodes.append(closest_node)
                            print(
                                f"使用电压源 {comp_id} 负极 (节点 {closest_node}) 作为参考 - 基于起点")
                            break

    # 如果仍然没有接地，则使用具有最多连接的节点
    if not ground_nodes and node_based_graph['nodes']:
        max_connections = 0
        reference_node = None

        for node_id, node_data in node_based_graph['nodes'].items():
            if len(node_data['connected_components']) > max_connections:
                max_connections = len(node_data['connected_components'])
                reference_node = node_id

        if reference_node:
            ground_nodes.append(reference_node)
            print(f"未识别地面。使用连接最紧密的节点 {reference_node} 作为参考")

    print(f"使用以下接地节点： {ground_nodes}")

    # 用于调试的印刷电路节点连接
    print("\nCircuit Nodes:")
    for node_id, node_data in node_based_graph['nodes'].items():
        components_str = ', '.join(node_data['connected_components'])
        print(f"  {node_id}:被连接于元件{components_str}")

        # 对于电压源，指出哪一侧连接到哪个节点
        for comp_id in node_data['connected_components']:
            comp_type = components_data.get(comp_id, {}).get('class_name', '')
            if 'voltage_source' in comp_type.lower():
                for side, conn_node in component_nodes[comp_id].items():
                    if conn_node == node_id:
                        print(f"    {comp_id} 通过 {side} 端连接")

                        # 如果该电压源在 component_directions 中已识别终端
                        if comp_id in component_directions:
                            neg_node, pos_node = component_directions[comp_id]

                            # 确定这是负极还是正极
                            if node_id == neg_node:
                                print(f"      这是负极（-）端子")
                            elif node_id == pos_node:
                                print(f"      这是正极 (+) 端子")

    # 应用MNA分析电路
    try:
        print("\n执行修改后的节点分析...")
        results = perform_graph_based_circuit_analysis(
            circuit_graph,
            component_values,
            controlled_sources,
            ground_nodes,
            node_based_graph,
            component_nodes,
            components_data
        )

        # 使用计算值创建电路的可视化
        # visualize_circuit_solution(
        #     circuit_graph,
        #     connectivity_graph,
        #     components_data,
        #     component_values,
        #     controlled_sources,
        #     results,
        #     "circuit_component_solution.png"
        # )

        visualize_circuit_with_values(
            components_data,
            node_based_graph,
            component_nodes,
            results,
            "circuit_with_node_values.png"
        )
        return results

    except Exception as e:
        print(f"Error during circuit analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# def verify_kcl(circuit_graph, results, node_based_graph, component_nodes):
#     """
#     在每个节点上正确验证基尔霍夫电流定律。
#
#     KCL 指出：进入节点的电流总和 = 离开节点的电流总和
#     """
#     print("\n--- 验证每个节点的 KCL ---")
#
#     for node_id, node_data in node_based_graph['nodes'].items():
#         if node_id not in results.get('node_voltages', {}):
#             continue
#
#         total_current = 0.0
#         connected_components = node_data['connected_components']
#         currents = []
#
#         for comp_id in connected_components:
#             if comp_id not in results.get('branch_currents', {}):
#                 continue
#
#             branch_current = results['branch_currents'][comp_id]
#
#             if component_nodes and comp_id in component_nodes:
#                 comp_terminals = list(component_nodes[comp_id].items())
#
#                 sign = 1
#
#                 node_terminal_idx = -1
#                 for i, (side, connected_node) in enumerate(comp_terminals):
#                     if connected_node == node_id:
#                         node_terminal_idx = i
#                         break
#
#                 if node_terminal_idx != -1:
#                     comp_type = circuit_graph.nodes[comp_id]['component_type'].lower()
#
#                     if 'resistor' in comp_type:
#                         # 对于电阻：电流从高电位流向低电位
#                         # 参考方向是从第一个端子流向第二个端子
#                         # 如果此节点是第一个端子：
#                         # - 如果电流 > 0：电流流出（负）
#                         # - 如果电流 < 0：电流流入（正）
#                         # 如果此节点是第二个端子：
#                         # - 如果电流 > 0：电流流入（正）
#                         # - 如果电流 < 0：电流流出（负）
#                         if node_terminal_idx == 0:
#                             sign = -1 if branch_current > 0 else 1
#                         else:  # Second terminal
#                             sign = 1 if branch_current > 0 else -1
#
#                     elif 'voltage_source' in comp_type:
#                         # 对于电压源：来自 MNA 的电流已经具有正确的符号
#                         # 只需确定它是进入还是离开此节点
#                         if node_terminal_idx == 0:
#                             sign = 1
#                         else:
#                             sign = -1
#
#                     elif 'current_source' in comp_type:
#                         if node_terminal_idx == 0:
#                             sign = -1
#                         else:
#                             sign = 1
#
#                 comp_current = sign * abs(branch_current)
#                 total_current += comp_current
#                 currents.append(f"{comp_id}: {comp_current:.4f}A")
#
#             else:
#                 sign = 1 if node_id < comp_id else -1
#                 comp_current = sign * branch_current
#                 total_current += comp_current
#                 currents.append(f"{comp_id}: {comp_current:.4f}A")
#
#         if abs(total_current) > 1e-5:
#             print(f"警告：节点 {node_id} 处的 KCL 不满足要求。净电流 = {total_current:.6f}A")
#             if currents:
#                 print(f"  组件电流：{', '.join(currents)}")
#         else:
#             print(f"节点 {node_id} 处的 KCL 满足要求。净电流 = {total_current:.6f}A")
#             if currents:
#                 print(f"  元件电流：{', '.join(currents)}")


# def visualize_circuit_solution(circuit_graph, connectivity_graph, components_data,
#                                component_values, controlled_sources, results, output_path=None):
#     """
#     Create a visualization of the circuit solution.
#
#     Args:
#         circuit_graph: NetworkX directed graph representing the circuit
#         connectivity_graph: Original undirected MultiGraph
#         components_data: Dictionary of component data
#         component_values: Dictionary of component values
#         controlled_sources: Dictionary of controlled source information
#         results: Results from circuit analysis
#         output_path: Path to save the visualization (optional)
#     """
#     plt.figure(figsize=(12, 10))
#
#     # Create a layout for the circuit graph
#     pos = nx.spring_layout(connectivity_graph, seed=42)
#
#     # Draw nodes with different colors based on type
#     component_types = {}
#     for node in connectivity_graph.nodes():
#         comp_data = components_data.get(node, {})
#         comp_type = comp_data.get('class_name', 'unknown').lower()
#
#         if comp_type not in component_types:
#             component_types[comp_type] = []
#         component_types[comp_type].append(node)
#
#     # Draw each component type with a different color
#     colors = plt.cm.tab10.colors
#     for i, (comp_type, nodes) in enumerate(component_types.items()):
#         nx.draw_networkx_nodes(connectivity_graph, pos,
#                                nodelist=nodes,
#                                node_color=[colors[i % len(colors)]] * len(nodes),
#                                node_size=500,
#                                label=comp_type)
#
#     # Draw edges
#     nx.draw_networkx_edges(connectivity_graph, pos, width=1.5, alpha=0.7)
#
#     # Add node labels with component ID and voltage
#     node_labels = {}
#     for node in connectivity_graph.nodes():
#         label = node
#
#         # Add voltage if available
#         if node in results.get('node_voltages', {}):
#             voltage = results['node_voltages'][node]
#             node_labels[node] = f"{label}\n{voltage:.2f}V"
#         elif node in results.get('component_voltages', {}):
#             voltage = results['component_voltages'][node]
#             node_labels[node] = f"{label}\n{voltage:.2f}V"
#         else:
#             node_labels[node] = label
#
#     nx.draw_networkx_labels(connectivity_graph, pos, labels=node_labels, font_size=8)
#
#     # Add edge labels with current flow
#     edge_labels = {}
#     for u, v in connectivity_graph.edges():
#         # Check if we have current for component u or v
#         if u in results.get('branch_currents', {}):
#             current = results['branch_currents'][u]
#             edge_labels[(u, v)] = f"{abs(current):.2f}A"
#         elif v in results.get('branch_currents', {}):
#             current = results['branch_currents'][v]
#             edge_labels[(u, v)] = f"{abs(current):.2f}A"
#
#     nx.draw_networkx_edge_labels(connectivity_graph, pos, edge_labels=edge_labels, font_size=7)
#
#     # Add component values as a separate text box
#     textbox_content = "Component Values:\n"
#     for comp_id, value in component_values.items():
#         comp_type = components_data.get(comp_id, {}).get('class_name', 'unknown')
#
#         if 'resistor' in comp_type.lower():
#             textbox_content += f"{comp_id}: {value} Ω\n"
#         elif 'voltage_source' in comp_type.lower():
#             textbox_content += f"{comp_id}: {value} V\n"
#         elif 'current_source' in comp_type.lower():
#             textbox_content += f"{comp_id}: {value} A\n"
#         else:
#             textbox_content += f"{comp_id}: {value}\n"
#
#     # Add controlled sources
#     if controlled_sources:
#         textbox_content += "\nControlled Sources:\n"
#         for cs_id, cs_info in controlled_sources.items():
#             control_type = cs_info['control_type']
#             controlling = cs_info['controlling_component']
#             gain = cs_info['gain']
#
#             textbox_content += f"{cs_id}: Controlled by {controlling} (gain={gain})\n"
#
#     plt.figtext(1.02, 0.5, textbox_content, verticalalignment='center',
#                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
#
#     # Add results summary
#     results_content = "Circuit Analysis Results:\n"
#
#     # Node voltages summary
#     results_content += "\nNode Voltages:\n"
#     for node, voltage in results.get('node_voltages', {}).items():
#         results_content += f"{node}: {voltage:.2f}V\n"
#
#     # Branch currents summary
#     results_content += "\nBranch Currents:\n"
#     for branch, current in results.get('branch_currents', {}).items():
#         results_content += f"{branch}: {current:.2f}A\n"
#
#     plt.figtext(1.02, 0.1, results_content, verticalalignment='bottom',
#                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
#
#     plt.title('Circuit Analysis Results')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.subplots_adjust(right=0.7)  # Make room for the textbox
#
#     # Save the figure if an output path is provided
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Circuit solution visualization saved as '{output_path}'")
#     else:
#         # Default save path
#         plt.savefig("circuit_solution.png", dpi=300, bbox_inches='tight')
#         print("Circuit solution visualization saved as 'circuit_solution.png'")
#
#     plt.close()
