import numpy as np

from analyze_component_connectivity_fast import get_connection_point


def perform_graph_based_circuit_analysis(circuit_graph, component_values, controlled_sources,
                                         ground_nodes, node_based_graph=None, component_nodes=None, components_data=None):
    """
    对电路执行改进节点分析 (MNA) 来计算电压和电流。

    参数：
    circuit_graph：表示电路连通性的 NetworkX 图
    component_values：元件值字典（电阻、电压/电流源）
    controlled_sources：受控源信息字典
    ground_nodes：接地/参考节点列表
    node_based_graph：用于调试的附加基于节点的表示
    component_nodes：元件到其终端节点的映射
    components_data：原始元件数据字典

    返回：
    results：包含每个元件计算出的电压和电流的字典
    """
    print("\n--- 步骤1：电路准备 ---")

    # 从电路图中提取所有节点（连接点）
    circuit_nodes = [n for n, attrs in circuit_graph.nodes(data=True)
                     if attrs.get('type') == 'node']

    # # 如果没有指定接地，则选择一个参考节点
    # if not ground_nodes and circuit_nodes:
    #     ground_nodes = [circuit_nodes[0]]
    #     print(f"No ground specified. Using {ground_nodes[0]} as reference node.")
    reference_node = ground_nodes[0]

    # 获取非参考节点（这些节点将具有电压变量）
    non_ref_nodes = [node for node in circuit_nodes if node != reference_node]

    # 按类型识别组件
    resistors = []
    voltage_sources = []
    current_sources = []

    for node, attrs in circuit_graph.nodes(data=True):
        if 'component_type' not in attrs:
            continue

        comp_type = attrs.get('component_type', '').lower()
        if comp_type == 'resistor':
            resistors.append(node)
        elif 'voltage_source' in comp_type:
            voltage_sources.append(node)
        elif 'current_source' in comp_type or 'controlled_current_source' in comp_type:
            current_sources.append(node)

    # 获取每个组件连接的节点
    comp_connections = {}

    # 如果提供，首先尝试使用component_nodes
    if component_nodes:
        for comp_id in resistors + voltage_sources + current_sources:
            if comp_id in component_nodes:
                # Extract connected nodes from component_nodes
                conn_nodes = list(component_nodes[comp_id].values())
                if len(conn_nodes) >= 2:
                    comp_connections[comp_id] = conn_nodes

    # 如果未提供或不完整，则返回到circuit_graph邻居
    for comp_id in resistors + voltage_sources + current_sources:
        if comp_id not in comp_connections or len(comp_connections[comp_id]) < 2:
            # Find nodes connected to this component in the graph
            connected_nodes = [n for n in circuit_graph.neighbors(comp_id)
                               if n in circuit_nodes]

            if len(connected_nodes) >= 2:
                comp_connections[comp_id] = connected_nodes[:2]  # 使用前两个节点

    print(f"电路具有 {len(non_ref_nodes)} 个非参考节点和 {len(voltage_sources)} 个电压源")
    print(
        f"找到 {len(resistors)} 个电阻器、{len(voltage_sources)} 个电压源、{len(current_sources)} 个电流源")

    print("\n--- 步骤2：矩阵初始化 ---")

    # 创建节点索引映射（用于矩阵索引）
    node_index = {node: i for i, node in enumerate(non_ref_nodes)}

    # 电压源数量（每个电压源增加一个额外的电流变量）
    n_nodes = len(non_ref_nodes)
    n_vs = len(voltage_sources)
    n_equations = n_nodes + n_vs

    # 创建MNA系数矩阵和右侧向量
    A = np.zeros((n_equations, n_equations))
    b = np.zeros(n_equations)

    # 创建电压源索引映射
    vs_index = {}
    for i, vs in enumerate(voltage_sources):
        vs_index[vs] = n_nodes + i  # 放置在节点变量之后

    print(f"矩阵大小：{n_equations}x{n_equations}")

    print("\n--- 步骤 3：矩阵填充 ---")

    # 添加电阻器（构建导纳矩阵G）
    for r_id in resistors:
        if r_id in component_values:
            resistance = component_values[r_id]
            conductance = 1.0 / resistance

            # 获取连接到该电阻的节点
            if r_id in comp_connections:
                nodes = comp_connections[r_id]

                if len(nodes) >= 2:
                    node1, node2 = nodes[0], nodes[1]

                    # 根据节点连接更新系数矩阵
                    if node1 != reference_node and node1 in node_index:
                        i = node_index[node1]
                        A[i, i] += conductance

                    if node2 != reference_node and node2 in node_index:
                        j = node_index[node2]
                        A[j, j] += conductance

                    if node1 != reference_node and node2 != reference_node and node1 in node_index and node2 in node_index:
                        i, j = node_index[node1], node_index[node2]
                        A[i, j] -= conductance
                        A[j, i] -= conductance
        else:
            print(f"警告：未提供电阻器 {r_id} 的值")

    # 添加独立电压源
    for v_id in voltage_sources:
        # 暂时跳过受控电压源（我们将单独处理它们）
        if v_id in controlled_sources:
            continue

        if v_id in component_values:
            voltage = component_values[v_id]

            # 获取连接到该电压源的节点
            if v_id in comp_connections:
                nodes = comp_connections[v_id]

                if len(nodes) >= 2:
                    # 但根据惯例：第一个节点（起点）为负，第二个节点（终点）为正
                    neg_node, pos_node = nodes[0], nodes[1]

                    # 使用电压源的方向信息
                    if 'direction' in components_data.get(v_id, {}):
                        print(f"使用电压源 {v_id} 的方向信息（方向 = 电流流动）")

                    # VS equation: v(pos) - v(neg) = voltage
                    vs_row = vs_index[v_id]

                    if pos_node != reference_node and pos_node in node_index:
                        i = node_index[pos_node]
                        A[vs_row, i] = 1.0  # 正极系数
                        A[i, vs_row] = 1.0  # 对应的KCL术语

                    if neg_node != reference_node and neg_node in node_index:
                        j = node_index[neg_node]
                        A[vs_row, j] = -1.0  # 负极系数
                        A[j, vs_row] = -1.0  # 对应的KCL术语

                    # 设置RHS值
                    b[vs_row] = voltage

                    # debug
                    print(f"电压源 {v_id}：从 {pos_node}(+) 到 {neg_node}(-) 的 {voltage}V")
        else:
            print(f"警告：未提供电压源{v_id}的值")

    # 添加独立电流源
    for i_id in current_sources:
        # 跳过受控电流源
        if i_id in controlled_sources:
            continue

        if i_id in component_values:
            current = component_values[i_id]

            # 获取连接到该电流源的节点
            if i_id in comp_connections:
                nodes = comp_connections[i_id]

                if len(nodes) >= 2:
                    # 第一个点是负极
                    node1, node2 = nodes[0], nodes[1]

                    # 更新非参考节点的 RHS
                    if node1 != reference_node and node1 in node_index:
                        i = node_index[node1]
                        b[i] -= current  # 流出的电流减少RHS

                    if node2 != reference_node and node2 in node_index:
                        j = node_index[node2]
                        b[j] += current  # 流入的电流增加RHS

                    # debug
                    print(f"电流源 {i_id}：从 {node1} 流向 {node2} 的 {current}A")
        else:
            print(f"警告：未提供当前源 {i_id} 的值")

    # 添加电压受控电压源 (VCVS)
    for cs_id, cs_info in controlled_sources.items():
        if cs_id in voltage_sources:  # 处理电压控制电压源
            control_type = cs_info.get('control_type', '')
            controlling_comp = cs_info.get('controlling_component', '')
            gain = cs_info.get('gain', 1.0)

            if control_type == 'voltage' and controlling_comp in comp_connections:
                # 获取控制节点（控制组件连接的地方）
                control_nodes = comp_connections[controlling_comp]

                if len(control_nodes) >= 2:
                    control_neg, control_pos = control_nodes[0], control_nodes[1]

                    # 得到VCVS节点
                    if cs_id in comp_connections:
                        cs_nodes = comp_connections[cs_id]

                        if len(cs_nodes) >= 2:
                            cs_neg, cs_pos = cs_nodes[0], cs_nodes[1]

                            # VCVS equation: v(cs_pos) - v(cs_neg) = gain * (v(control_pos) - v(control_neg))
                            vs_row = vs_index[cs_id]

                            # 输出电压部分（等式左边）
                            if cs_pos != reference_node and cs_pos in node_index:
                                i = node_index[cs_pos]
                                A[vs_row, i] = 1.0
                                A[i, vs_row] = 1.0

                            if cs_neg != reference_node and cs_neg in node_index:
                                j = node_index[cs_neg]
                                A[vs_row, j] = -1.0
                                A[j, vs_row] = -1.0

                            # 控制电压部分（等式右边，带负号）
                            if control_pos != reference_node and control_pos in node_index:
                                i = node_index[control_pos]
                                A[vs_row, i] -= gain

                            if control_neg != reference_node and control_neg in node_index:
                                j = node_index[control_neg]
                                A[vs_row, j] += gain

                            # Print debug info
                            print(f"VCVS {cs_id}：增益={gain}，由 {controlling_comp} 控制")
                            print(f"  输出从{cs_pos}(+)到{cs_neg}(-)")
                            print(f"  控制从{control_pos}(+)到{control_neg}(-)")

    # 增添电压受控电流源(VCCS)
    for cs_id, cs_info in controlled_sources.items():
        if cs_id in current_sources and cs_info.get('control_type', '') == 'voltage':
            controlling_comp = cs_info.get('controlling_component', '')
            gain = cs_info.get('gain', 1.0)  # 跨导(A/V)

            if controlling_comp in comp_connections:
                control_nodes = comp_connections[controlling_comp]

                if len(control_nodes) >= 2:
                    # 决定控制方向
                    control_comp_data = components_data.get(controlling_comp, {})
                    comp_orientation = "horizontal"

                    if 'box' in control_comp_data:
                        x1, y1, x2, y2 = control_comp_data['box']
                        width = x2 - x1
                        height = y2 - y1
                        comp_orientation = "horizontal" if width > height else "vertical"

                    if comp_orientation == "horizontal":
                        # 水平分量：左端为负极，右端为正极
                        if 'box' in control_comp_data and controlling_comp in component_nodes:
                            left_side = None
                            right_side = None

                            # 根据位置找到左侧和右侧
                            for side, node_id in component_nodes[controlling_comp].items():
                                if side == 'left':
                                    left_side = node_id
                                elif side == 'right':
                                    right_side = node_id

                            if left_side is not None and right_side is not None:
                                control_neg, control_pos = left_side, right_side
                            else:
                                # 默认节点方向
                                control_neg, control_pos = control_nodes[0], control_nodes[1]
                        else:
                            # 默认节点方向
                            control_neg, control_pos = control_nodes[0], control_nodes[1]
                    else:
                        # 垂直元件：顶部为负极，底部为正极
                        if 'box' in control_comp_data and controlling_comp in component_nodes:
                            top_side = None
                            bottom_side = None

                            for side, node_id in component_nodes[controlling_comp].items():
                                if side == 'top':
                                    top_side = node_id
                                elif side == 'bottom':
                                    bottom_side = node_id

                            if top_side is not None and bottom_side is not None:
                                control_neg, control_pos = top_side, bottom_side
                            else:
                                control_neg, control_pos = control_nodes[0], control_nodes[1]
                        else:
                            control_neg, control_pos = control_nodes[0], control_nodes[1]

                    # 得到VCCS输出节点
                    if cs_id in comp_connections:
                        cs_nodes = comp_connections[cs_id]

                        if len(cs_nodes) >= 2:
                            # 电流从第一个节点流向第二个节点
                            out1, out2 = cs_nodes[0], cs_nodes[1]

                            # 对于每个非参考输出节点，将 gain*(v_control_pos - v_control_neg) 添加到其 KCL 方程中

                            # 将控制电压贡献添加到out1的方程中（如果不是参考）
                            if out1 != reference_node and out1 in node_index:
                                i = node_index[out1]

                                if control_pos != reference_node and control_pos in node_index:
                                    j = node_index[control_pos]
                                    A[i, j] -= gain  # 电流从 out1 流出

                                if control_neg != reference_node and control_neg in node_index:
                                    j = node_index[control_neg]
                                    A[i, j] += gain  # 相反的效果

                            # 将控制电压贡献添加到 out2 的方程中（如果不是参考）
                            if out2 != reference_node and out2 in node_index:
                                i = node_index[out2]

                                if control_pos != reference_node and control_pos in node_index:
                                    j = node_index[control_pos]
                                    A[i, j] += gain  # 流入out2的电流

                                if control_neg != reference_node and control_neg in node_index:
                                    j = node_index[control_neg]
                                    A[i, j] -= gain  # 相反的效果

                            # debug
                            print(f"VCCS {cs_id}：跨导={gain}，由 {controlling_comp} 控制")
                            print(f"  控制从{control_pos}(+)到{control_neg}(-)")
                            print(f"  输出从{out1}到{out2}")

    # 添加电流控制电压源 (CCVS)
    for cs_id, cs_info in controlled_sources.items():
        if cs_id in voltage_sources and cs_info.get('control_type', '') == 'current':
            controlling_comp = cs_info.get('controlling_component', '')
            gain = cs_info.get('gain', 1.0)  # 转阻(V/A)

            # 确保我们有控制组件的信息
            if controlling_comp in comp_connections:
                # 获取 CCVS 的节点
                if cs_id in comp_connections and cs_id in vs_index:
                    ccvs_row = vs_index[cs_id]
                    cs_nodes = comp_connections[cs_id]

                    if len(cs_nodes) >= 2:
                        cs_neg, cs_pos = cs_nodes[0], cs_nodes[1]

                        # 输出电压部分（等式左边）
                        if cs_pos != reference_node and cs_pos in node_index:
                            i = node_index[cs_pos]
                            A[ccvs_row, i] = 1.0
                            A[i, ccvs_row] = 1.0

                        if cs_neg != reference_node and cs_neg in node_index:
                            j = node_index[cs_neg]
                            A[ccvs_row, j] = -1.0
                            A[j, ccvs_row] = -1.0

                        # 判断控制电流来源：电压源还是电阻
                        if controlling_comp in vs_index:  # 电压源电流
                            controlling_vs_row = vs_index[controlling_comp]
                            A[ccvs_row, controlling_vs_row] = -gain

                            print(f"CCVS {cs_id}：转阻={gain}，由电压源 {controlling_comp} 的电流控制")

                        elif controlling_comp in resistors and controlling_comp in component_values:  # 电阻电流
                            R_value = component_values.get(controlling_comp, 1.0)
                            control_nodes = comp_connections[controlling_comp]

                            if len(control_nodes) >= 2:
                                ctrl_n1, ctrl_n2 = control_nodes[0], control_nodes[1]

                                # 使用KCL关系：I_R = (V_n1 - V_n2) / R
                                # 控制电流部分：v(cs_pos) - v(cs_neg) = gain * (v(ctrl_n1) - v(ctrl_n2)) / R_value

                                if ctrl_n1 != reference_node and ctrl_n1 in node_index:
                                    k = node_index[ctrl_n1]
                                    A[ccvs_row, k] -= gain / R_value  # 负号是因为控制电流与v(ctrl_n1)成正比

                                if ctrl_n2 != reference_node and ctrl_n2 in node_index:
                                    k = node_index[ctrl_n2]
                                    A[ccvs_row, k] += gain / R_value  # 正号是因为控制电流与v(ctrl_n2)成反比

                                print(f"CCVS {cs_id}：转阻={gain}，由电阻 {controlling_comp}(值={R_value}Ω) 的电流控制")

                        print(f"  输出从{cs_pos}(+)到{cs_neg}(-)")
                        if controlling_comp in resistors and controlling_comp in component_values:
                            print(f"  控制电流方向由 {ctrl_n1} 到 {ctrl_n2}")
                        else:
                            print(f"  控制电流通过 {controlling_comp}")

    # 添加电流控制电流源 (CCCS)
    for cs_id, cs_info in controlled_sources.items():
        if cs_id in current_sources and cs_info.get('control_type', '') == 'current':
            controlling_comp = cs_info.get('controlling_component', '')
            gain = cs_info.get('gain', 1.0)  # 电流增益(无单位)

            # 确保我们有控制组件的信息
            if controlling_comp in comp_connections:
                # 获取 CCCS 的输出节点
                if cs_id in comp_connections:
                    cs_nodes = comp_connections[cs_id]

                    if len(cs_nodes) >= 2:
                        # 电流从第一个节点流向第二个节点
                        out1, out2 = cs_nodes[0], cs_nodes[1]

                        # 判断控制电流来源：电压源还是电阻
                        if controlling_comp in vs_index:  # 电压源电流
                            controlling_vs_row = vs_index[controlling_comp]

                            # 对输出节点添加KCL贡献
                            if out1 != reference_node and out1 in node_index:
                                i = node_index[out1]
                                A[i, controlling_vs_row] = -gain  # 电流从out1流出，为负

                            if out2 != reference_node and out2 in node_index:
                                i = node_index[out2]
                                A[i, controlling_vs_row] = gain  # 电流流入out2，为正

                            print(f"CCCS {cs_id}：电流增益={gain}，由电压源 {controlling_comp} 的电流控制")

                        elif controlling_comp in resistors and controlling_comp in component_values:  # 电阻电流
                            R_value = component_values.get(controlling_comp, 1.0)
                            control_nodes = comp_connections[controlling_comp]

                            if len(control_nodes) >= 2:
                                ctrl_n1, ctrl_n2 = control_nodes[0], control_nodes[1]

                                # 使用KCL关系：I_R = (V_n1 - V_n2) / R
                                # CCCS的输出电流：I_out = gain * (v(ctrl_n1) - v(ctrl_n2)) / R_value

                                # 对out1节点的KCL贡献
                                if out1 != reference_node and out1 in node_index:
                                    i = node_index[out1]

                                    if ctrl_n1 != reference_node and ctrl_n1 in node_index:
                                        j = node_index[ctrl_n1]
                                        A[i, j] -= gain / R_value  # 负号表示电流从out1流出

                                    if ctrl_n2 != reference_node and ctrl_n2 in node_index:
                                        j = node_index[ctrl_n2]
                                        A[i, j] += gain / R_value  # 正号表示反向效应

                                # 对out2节点的KCL贡献
                                if out2 != reference_node and out2 in node_index:
                                    i = node_index[out2]

                                    if ctrl_n1 != reference_node and ctrl_n1 in node_index:
                                        j = node_index[ctrl_n1]
                                        A[i, j] += gain / R_value  # 正号表示电流流入out2

                                    if ctrl_n2 != reference_node and ctrl_n2 in node_index:
                                        j = node_index[ctrl_n2]
                                        A[i, j] -= gain / R_value  # 负号表示反向效应

                                print(f"CCCS {cs_id}：电流增益={gain}，由电阻 {controlling_comp}(值={R_value}Ω) 的电流控制")

                        print(f"  输出从{out1}到{out2}")
                        if controlling_comp in resistors and controlling_comp in component_values:
                            print(f"  控制电流方向由 {ctrl_n1} 到 {ctrl_n2}")
                        else:
                            print(f"  控制电流通过 {controlling_comp}")

    # 向对角线元素添加小电导以防止奇异矩阵
    for i in range(n_nodes):
        A[i, i] += 1e-12

    print("\n--- 步骤4：求解线性系统 ---")
    print(f"矩阵A =\n{A}")
    print(f"向量b =\n{b}")

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Warning: Singular matrix detected. Using least squares solver.")
        # 对奇异矩阵使用伪逆
        x = np.linalg.lstsq(A, b, rcond=None)[0]

    print(f"解出x =\n{x}")

    print("\n--- 步骤5：提取结果 ---")

    results = {
        'node_voltages': {},
        'branch_currents': {},
        'component_voltages': {}
    }

    # 将参考节点电压设置为0
    results['node_voltages'][reference_node] = 0.0

    # 提取节点电压
    for node, idx in node_index.items():
        results['node_voltages'][node] = x[idx]

    # 提取电压源电流
    for vs, idx in vs_index.items():
        current = x[idx]
        results['branch_currents'][vs] = current
        print(f"电压源{vs}: I = {current:.4f}A")

    # 计算电阻电流和元件电压
    for comp_id in comp_connections:
        nodes = comp_connections[comp_id]

        if len(nodes) >= 2:
            node1, node2 = nodes[0], nodes[1]

            # 获取节点电压（如果是参考节点则为 0）
            v1 = results['node_voltages'].get(node1, 0.0)
            v2 = results['node_voltages'].get(node2, 0.0)

            # 计算元件两端的电压（显示时始终为正值）
            voltage_drop = v1 - v2
            results['component_voltages'][comp_id] = abs(voltage_drop)

            # 计算电阻电流
            if comp_id in resistors and comp_id in component_values:
                resistance = component_values[comp_id]
                # 电流从电压高的地方流向电压低的地方
                # 所以此处电流都取正值，画图时会由箭头表示方向
                current = voltage_drop / resistance
                results['branch_currents'][comp_id] = abs(current)

                # 记录调试的方向
                if abs(current) > 1e-8:
                    flow_dir = "node1→node2" if current > 0 else "node2→node1"
                    print(f"电阻{comp_id}: I={current:.6f}A, 方向: {flow_dir}")

            # 对于电压源，从MNA结果中提取电流
            elif comp_id in voltage_sources:
                # 电流已根据MNA解决方案正确显示
                if comp_id in results.get('branch_currents', {}):
                    vs_current = results['branch_currents'][comp_id]
                    print(f"电压源{comp_id}: I={vs_current:.6f}A")
                    # print(f"  正极端子连接至：{node1 if node1 == pos_node else node2}")
                    # print(f"  负极端子连接至：{node1 if node1 == neg_node else node2}")

            # 电流源具有已知电流或依赖于控制电压
            elif comp_id in current_sources:
                if comp_id in component_values:
                    results['branch_currents'][comp_id] = component_values[comp_id]
                    print(f"Current source {comp_id}: I={component_values[comp_id]:.6f}A, direction: node1→node2")

                elif comp_id in controlled_sources:
                    # 受控电流源——根据控制电压计算电流
                    cs_info = controlled_sources[comp_id]

                    if cs_info.get('control_type') == 'voltage':
                        controlling_comp = cs_info.get('controlling_component')
                        gain = cs_info.get('gain', 0.0)

                        if controlling_comp in comp_connections:
                            ctrl_nodes = comp_connections[controlling_comp]

                            if len(ctrl_nodes) >= 2:
                                ctrl_node1, ctrl_node2 = ctrl_nodes[0], ctrl_nodes[1]

                                # 得到控制电压
                                ctrl_v1 = results['node_voltages'].get(ctrl_node1, 0.0)
                                ctrl_v2 = results['node_voltages'].get(ctrl_node2, 0.0)

                                # 计算控制电压 - 如果电压从 ctrl_node1 降至 ctrl_node2，则为正
                                ctrl_voltage = ctrl_v1 - ctrl_v2

                                # 电流 = 增益 * 控制电压
                                # 正电流从节点1流向节点2
                                current = gain * ctrl_voltage
                                results['branch_currents'][comp_id] = current

                                print(f"VCCS {comp_id}: I={current:.6f}A, 控制电压={ctrl_voltage:.6f}V")
    print("完成分析.")
    return results


def visualize_circuit_with_values(components_data, node_based_graph, component_nodes, results, output_path=None):
    """
    创建包含节点电压和元件电流/电压的电路可视化图。使用图像处理中检测到的实际元件位置。

    参数：
    components_data：包含框坐标的元件数据字典
    node_based_graph：基于节点的电路表示
    component_nodes：元件到其终端节点的映射
    results：电路分析结果
    output_path：可视化图的保存路径

    返回：
    fig：matplotlib 图形对象
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 10))
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for comp_id, comp_data in components_data.items():
        if 'box' in comp_data:
            x1, y1, x2, y2 = comp_data['box']
            x_min = min(x_min, x1)
            y_min = min(y_min, y1)
            x_max = max(x_max, x2)
            y_max = max(y_max, y2)

    # 添加一些填充
    padding = 50
    ax.set_xlim(x_min - padding, x_max + padding)

    # 反转y轴以匹配图像坐标
    ax.set_ylim(y_max + padding, y_min - padding)  # Inverted

    # 为不同的组件类型定义颜色
    component_colors = {
        'resistor': 'royalblue',
        'voltage_source': 'crimson',
        'controlled_voltage_source': 'darkorange',
        'current_source': 'green',
        'controlled_current_source': 'purple',
        'capacitor': 'teal',
        'inductor': 'brown',
        'ground': 'black'
    }

    # 首先绘制所有组件和节点之间的连接
    for comp_id, comp_data in components_data.items():
        if 'box' in comp_data and comp_id in component_nodes:
            x1, y1, x2, y2 = comp_data['box']

            # 画出节点之间的连接
            for side, node_id in component_nodes[comp_id].items():
                if node_id in node_based_graph['nodes']:
                    # 获取组件上的终端位置
                    terminal_pos = get_connection_point((x1, y1, x2, y2), side)

                    # 得到节点位置
                    node_pos = node_based_graph['nodes'][node_id]['position']

                    # 画出从终端到节点之间的线
                    ax.plot([terminal_pos[0], node_pos[0]],
                            [terminal_pos[1], node_pos[1]],
                            'k-', linewidth=1, alpha=0.6, zorder=1)

    # 画出电路节点
    for node_id, node_data in node_based_graph['nodes'].items():
        # 得到节点的位置
        pos = node_data['position']

        # 从结果中取出节点电压
        voltage = results.get('node_voltages', {}).get(node_id, 'N/A')

        # 基于连接数的节点大小
        node_size = 5 + 2 * len(node_data['connected_components'])

        # 画节点
        ax.plot(pos[0], pos[1], 'o',
                markersize=node_size,
                markerfacecolor='gold',
                markeredgecolor='black',
                zorder=5)

        # 添加带电压的节点标签（调整位置以提高可读性）
        if voltage != 'N/A':
            label_text = f"{node_id}: {voltage:.2f}V"
        else:
            label_text = f"{node_id}"

        # 将标签放置在节点略右上方
        ax.text(pos[0] + 10, pos[1] - 10,  # 将标签放置在节点略右上方
                label_text,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round'),
                zorder=6)

    # 画元件
    for comp_id, comp_data in components_data.items():
        if 'box' in comp_data:
            x1, y1, x2, y2 = comp_data['box']
            width = x2 - x1
            height = y2 - y1

            # 获取组件类型（删除下划线和数字）
            raw_comp_type = comp_data['class_name'].lower()
            comp_type = ''.join([c for c in raw_comp_type if not (c.isdigit() or c == '_')])

            # 获取原件颜色
            color = component_colors.get(comp_type, 'gray')

            # 对元件创建矩形框
            rect = patches.Rectangle((x1, y1), width, height,
                                     linewidth=2,
                                     edgecolor=color,
                                     facecolor='none',
                                     zorder=3)
            ax.add_patch(rect)

            # 从结果中获取元件值
            current = results.get('branch_currents', {}).get(comp_id, 'N/A')
            voltage = results.get('component_voltages', {}).get(comp_id, 'N/A')

            # 创建具有 ID 和值的组件标签
            label_text = f"{comp_id}\n"
            if voltage != 'N/A':
                label_text += f"{voltage:.2f}V"
            if current != 'N/A':
                label_text += f", {current:.2f}A"

            # 将标签放置在组件中心
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.text(center_x, center_y,
                    label_text,
                    fontsize=9,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round'),
                    zorder=4)

            # 如果有电流，绘制电流方向箭头
            if current != 'N/A' and comp_id in component_nodes:
                terminals = list(component_nodes[comp_id].items())
                if len(terminals) >= 2:
                    # 根据元件侧面获取实际端子
                    side1, node1 = terminals[0]
                    side2, node2 = terminals[1]

                    # 获取组件上端子的实际位置
                    pos1 = get_connection_point((x1, y1, x2, y2), side1)
                    pos2 = get_connection_point((x1, y1, x2, y2), side2)

                    # 计算箭头的中点
                    mid_x = (pos1[0] + pos2[0]) / 2
                    mid_y = (pos1[1] + pos2[1]) / 2

                    # 根据组件类型和电流确定箭头方向
                    if 'source' in comp_type and 'voltage' in comp_type:
                        # 对于电压源，方向相反 - I_V 定义为从负流向正
                        # 源内部的电流，与外部电流方向相反。
                        # 为了表现这一点，I_V的方向都从第二个端子流向第一个端子（假设第一个端子为负）
                        dx = (pos1[0] - pos2[0]) / 5
                        dy = (pos1[1] - pos2[1]) / 5
                    elif 'source' in comp_type and 'current' in comp_type: ##
                        # 对于电流源，直接使用电流符号
                        if not isinstance(current, str) and current > 0:
                            # 电流从第一端子流向第二端子
                            print(pos1,pos2)
                            dx = (pos2[0] - pos1[0]) / 5
                            dy = (pos2[1] - pos1[1]) / 5
                        else:
                            # 电流从第二个端子流向第一个端子
                            dx = (pos1[0] - pos2[0]) / 5
                            dy = (pos1[1] - pos2[1]) / 5
                    else:
                        # 对于电阻器，使用节点电压来确定方向
                        v1 = results.get('node_voltages', {}).get(node1, 0)
                        v2 = results.get('node_voltages', {}).get(node2, 0)

                        if v1 > v2:
                            # 电流从第一端子流向第二端子
                            dx = (pos2[0] - pos1[0]) / 5
                            dy = (pos2[1] - pos1[1]) / 5
                        else:
                            # 电流从第二个端子流向第一个端子
                            dx = (pos1[0] - pos2[0]) / 5
                            dy = (pos1[1] - pos2[1]) / 5

                    # 在中点处绘制具有正确方向的箭头
                    ax.arrow(mid_x - dx / 2, mid_y - dy / 2, dx, dy,
                             head_width=5, head_length=7, fc='red', ec='red',
                             zorder=4)

                    # 在终端处添加小标记
                    ax.plot(pos1[0], pos1[1], 'ko', markersize=4, zorder=3)
                    ax.plot(pos2[0], pos2[1], 'ko', markersize=4, zorder=3)

    # 为组件类型创建图例
    legend_elements = []
    for comp_type, color in component_colors.items():
        legend_elements.append(Line2D([0], [0], color=color, linewidth=2, label=comp_type))

    # 将节点添加到图例
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='gold', markeredgecolor='black',
                                  markersize=8, label='Circuit Node'))

    ax.legend(handles=legend_elements, loc='upper right')


    ax.set_title('Circuit Analysis Results with Node Voltages and Component Values', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    summary_text = "Circuit Analysis Summary:\n\n"

    summary_text += "Node Voltages:\n"
    for node, voltage in sorted(results.get('node_voltages', {}).items()):
        summary_text += f"  {node}: {voltage:.2f}V\n"

    summary_text += "\nComponent Values:\n"
    for comp_id in sorted(components_data.keys()):
        current = results.get('branch_currents', {}).get(comp_id, 'N/A')
        voltage = results.get('component_voltages', {}).get(comp_id, 'N/A')

        if current != 'N/A' or voltage != 'N/A':
            summary_text += f"  {comp_id}: "
            if voltage != 'N/A':
                summary_text += f"{voltage:.2f}V"
            if current != 'N/A':
                summary_text += f", {current:.2f}A"
            summary_text += "\n"

    plt.figtext(1.02, 0.5, summary_text,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
                fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(right=0.7)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"电路可视化保存在'{output_path}'")

    return fig