import os

import cv2
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from analyze_component_connectivity_fast import analyze_component_connectivity_fast, get_connection_point
from process_circuit_image import process_circuit_image


def create_node_based_circuit_graph(components_data, detailed_connections):
    """
    使用并查集创建基于节点的电路表示，以识别组件之间的连接节点。

    参数：
    components_data：组件数据字典
    detailed_connections：详细连接信息列表

    返回：
    node_based_graph：基于节点的电路的字典表示
    nodes_mapping：将每个节点 ID 映射到其连接组件的字典
    component_nodes：将组件映射到其终端节点的字典
    """

    # 初始化并查集数据结构
    class UnionFind:
        def __init__(self):
            self.parent = {}
            self.rank = {}

        def find(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
                return x

            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)

            if root_x == root_y:
                return

            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_x] += 1

    # 创建并查集结构来跟踪连接点
    uf = UnionFind()

    # 为每个组件和侧面创建初始连接点
    connection_points = {}
    for conn in detailed_connections:
        comp1 = conn['comp1']
        comp2 = conn['comp2']
        side1 = conn['side1']
        side2 = conn['side2']

        # 为每个连接点创建唯一的ID
        point1_id = f"{comp1}_{side1}"
        point2_id = f"{comp2}_{side2}"

        # 添加到连接点
        if point1_id not in connection_points:
            connection_points[point1_id] = {
                'component': comp1,
                'side': side1,
                'position': get_connection_point(components_data[comp1]['box'], side1),
                'connections': []
            }

        if point2_id not in connection_points:
            connection_points[point2_id] = {
                'component': comp2,
                'side': side2,
                'position': get_connection_point(components_data[comp2]['box'], side2),
                'connections': []
            }

        # 合并这些点
        uf.union(point1_id, point2_id)

        # 追踪连接
        connection_points[point1_id]['connections'].append(point2_id)
        connection_points[point2_id]['connections'].append(point1_id)

    # 查找所有唯一节点（union-find 中的每个根代表一个节点）
    node_sets = {}
    for point_id in connection_points:
        root = uf.find(point_id)
        if root not in node_sets:
            node_sets[root] = []
        node_sets[root].append(point_id)

    # 使用连续节点ID创建更清晰的节点映射
    nodes_mapping = {}
    component_nodes = {}
    node_based_graph = {'nodes': {}, 'components': {}}

    # 分配连续节点 ID
    for i, (root, points) in enumerate(node_sets.items()):
        node_id = f"N{i + 1}"

        # 创建节点条目
        node_based_graph['nodes'][node_id] = {
            'connected_components': [],
            'position': [0, 0]  # 将会计算平均位置
        }

        # 计算所有点的平均位置
        total_x = 0
        total_y = 0
        components_set = set()

        for point_id in points:
            # 将每个点映射到此节点ID
            nodes_mapping[point_id] = node_id

            # 添加位置数据
            pos = connection_points[point_id]['position']
            total_x += pos[0]
            total_y += pos[1]

            # 跟踪哪些组件连接到该节点
            comp_id = connection_points[point_id]['component']
            components_set.add(comp_id)

            # 跟踪每个组件连接到哪些节点
            if comp_id not in component_nodes:
                component_nodes[comp_id] = {}

            component_nodes[comp_id][connection_points[point_id]['side']] = node_id

        # 计算平均位置
        avg_x = total_x // len(points)
        avg_y = total_y // len(points)
        node_based_graph['nodes'][node_id]['position'] = [avg_x, avg_y]

        # 存储连通分量
        node_based_graph['nodes'][node_id]['connected_components'] = list(components_set)

    # 将组件及其终端节点添加到图中
    for comp_id, comp_data in components_data.items():
        # 初始化图中的组件
        node_based_graph['components'][comp_id] = {
            'type': comp_data['class_name'],
            'position': comp_data['center'],
            'box': comp_data['box'],
            'terminals': component_nodes.get(comp_id, {}),
            'direction': None
        }

        # 添加方向信息
        if 'direction' in comp_data and comp_data['direction']:
            start_pt = comp_data['direction']['start']  # Negative terminal (current emission point)
            end_pt = comp_data['direction']['end']  # Positive terminal

            # Determine component orientation
            x1, y1, x2, y2 = comp_data['box']
            is_vertical = (y2 - y1) > (x2 - x1)

            # Create direction vector
            direction_vector = [end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]]

            # Normalize for better comparison
            vector_length = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
            if vector_length > 0:
                direction_vector = [direction_vector[0] / vector_length, direction_vector[1] / vector_length]

            terminals = component_nodes.get(comp_id, {})
            if terminals and len(terminals) >= 2:
                terminal_points = {}
                # Get all terminal coordinates
                for side, node_id in terminals.items():
                    terminal_points[node_id] = get_connection_point(comp_data['box'], side)

                # For vertical components
                if is_vertical:
                    # Find top and bottom terminals
                    top_node = None
                    bottom_node = None
                    min_y = float('inf')
                    max_y = float('-inf')
                    top_side = None
                    bottom_side = None

                    for side, node_id in terminals.items():
                        point = get_connection_point(comp_data['box'], side)
                        if point[1] < min_y:
                            min_y = point[1]
                            top_node = node_id
                            top_side = side
                        if point[1] > max_y:
                            max_y = point[1]
                            bottom_node = node_id
                            bottom_side = side

                    # If direction vector points downward (y increases), bottom is positive
                    if direction_vector[1] > 0:
                        node_based_graph['components'][comp_id]['positive'] = bottom_node
                        node_based_graph['components'][comp_id]['negative'] = top_node

                        # Store the terminal sides for reordering
                        positive_side = bottom_side
                        negative_side = top_side
                    else:
                        node_based_graph['components'][comp_id]['positive'] = top_node
                        node_based_graph['components'][comp_id]['negative'] = bottom_node

                        # Store the terminal sides for reordering
                        positive_side = top_side
                        negative_side = bottom_side

                # For horizontal components
                else:
                    # Find left and right terminals
                    left_node = None
                    right_node = None
                    min_x = float('inf')
                    max_x = float('-inf')
                    left_side = None
                    right_side = None

                    for side, node_id in terminals.items():
                        point = get_connection_point(comp_data['box'], side)
                        if point[0] < min_x:
                            min_x = point[0]
                            left_node = node_id
                            left_side = side
                        if point[0] > max_x:
                            max_x = point[0]
                            right_node = node_id
                            right_side = side

                    # If direction vector points rightward (x increases), right is positive
                    if direction_vector[0] > 0:
                        node_based_graph['components'][comp_id]['positive'] = right_node
                        node_based_graph['components'][comp_id]['negative'] = left_node

                        # Store the terminal sides for reordering
                        positive_side = right_side
                        negative_side = left_side
                    else:
                        node_based_graph['components'][comp_id]['positive'] = left_node
                        node_based_graph['components'][comp_id]['negative'] = right_node

                        # Store the terminal sides for reordering
                        positive_side = left_side
                        negative_side = right_side

                # Check if this is a directional component that needs reordering
                comp_type = comp_data['class_name'].lower()
                directional_components = [
                    'voltage_source', 'current_source',
                    'controlled_voltage_source', 'controlled_current_source',
                    'diode', 'led'
                ]

                if any(dc in comp_type for dc in directional_components):
                    # Reorder the terminal nodes in component_nodes to match the direction
                    # First, create a new ordered dictionary
                    ordered_terminals = {}

                    # Add negative terminal first (as it should be the "start" point)
                    ordered_terminals[negative_side] = node_based_graph['components'][comp_id]['negative']

                    # Add positive terminal second (as it should be the "end" point)
                    ordered_terminals[positive_side] = node_based_graph['components'][comp_id]['positive']

                    # Add any other terminals that might exist (unlikely for directional components)
                    for side, node_id in terminals.items():
                        if side != negative_side and side != positive_side:
                            ordered_terminals[side] = node_id

                    # Replace the terminals in component_nodes
                    component_nodes[comp_id] = ordered_terminals

                    # Update the terminals in the node_based_graph
                    node_based_graph['components'][comp_id]['terminals'] = ordered_terminals

            node_based_graph['components'][comp_id]['direction'] = {
                'start': start_pt,  # Negative terminal
                'end': end_pt  # Positive terminal
            }

    return node_based_graph, nodes_mapping, component_nodes


def determine_directed_edges(node_based_graph):
    """
    Determine directed edges between components based on the node-based circuit graph.

    Args:
        node_based_graph: Dictionary representation of the node-based circuit

    Returns:
        directed_edges: List of directed edges between components
    """
    directed_edges = []

    # Components that typically drive current in a certain direction
    directional_sources = [
        'voltage_source', 'controlled_voltage_source',
        'current_source', 'controlled_current_source',
        'diode', 'led'
    ]

    # Process each component
    for comp_id, comp_data in node_based_graph['components'].items():
        comp_type = comp_data['type'].lower()
        terminals = comp_data.get('terminals', {})

        # Check if this is a directional component
        if comp_type in directional_sources and 'positive' in comp_data and 'negative' in comp_data:
            # For sources, current flows out of positive terminal and into negative
            pos_node = comp_data['positive']
            neg_node = comp_data['negative']

            # Find other components connected to these nodes
            for node_id in [pos_node, neg_node]:
                connected_comps = node_based_graph['nodes'][node_id]['connected_components']

                for other_comp in connected_comps:
                    if other_comp != comp_id:
                        other_comp_data = node_based_graph['components'][other_comp]
                        other_terminals = other_comp_data.get('terminals', {})

                        # Find which terminal of the other component connects to this node
                        for side, other_node in other_terminals.items():
                            if other_node == node_id:
                                if node_id == pos_node:
                                    # Current flows from this component to the other
                                    directed_edges.append({
                                        'from': comp_id,
                                        'to': other_comp,
                                        'from_node': pos_node,
                                        'to_node': pos_node,
                                        'type': 'source_driven'
                                    })
                                elif node_id == neg_node:
                                    # Current flows from other component to this component
                                    directed_edges.append({
                                        'from': other_comp,
                                        'to': comp_id,
                                        'from_node': neg_node,
                                        'to_node': neg_node,
                                        'type': 'source_driven'
                                    })

    # For non-directional components, use component type hierarchy
    component_hierarchy = {
        'voltage_source': 1,
        'controlled_voltage_source': 1,
        'current_source': 1,
        'controlled_current_source': 1,
        'resistor': 2,
        'capacitor': 2,
        'inductor': 2,
        'ground': 3
    }

    # Process each node
    for node_id, node_data in node_based_graph['nodes'].items():
        connected_comps = node_data['connected_components']

        if len(connected_comps) > 1:
            # Sort components by hierarchy
            sorted_comps = sorted(connected_comps,
                                  key=lambda c: component_hierarchy.get(
                                      node_based_graph['components'][c]['type'].lower(), 2))

            # Assume current flows from lower hierarchy to higher
            for i in range(len(sorted_comps) - 1):
                from_comp = sorted_comps[i]
                to_comp = sorted_comps[i + 1]

                # Skip if we already have a directed edge between these components
                if not any(e['from'] == from_comp and e['to'] == to_comp for e in directed_edges):
                    directed_edges.append({
                        'from': from_comp,
                        'to': to_comp,
                        'from_node': node_id,
                        'to_node': node_id,
                        'type': 'hierarchy'
                    })

    return directed_edges


# def visualize_node_based_graph(node_based_graph, binary_image, directed_edges=None, output_path=None):
#     """
#     Create a visualization of the node-based circuit graph.
#
#     Args:
#         node_based_graph: Dictionary representation of the node-based circuit
#         binary_image: Binary image of the circuit for sizing
#         directed_edges: Optional list of directed edges
#         output_path: Path to save the visualization
#
#     Returns:
#         node_viz: Visualization image of the node-based circuit
#     """
#     # Create a colored image for visualization
#     h, w = binary_image.shape
#     node_viz = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
#
#     # Define colors
#     node_color = (0, 255, 255)  # Yellow
#     component_color = (255, 0, 0)  # Blue
#     edge_color = (0, 255, 0)  # Green
#     directed_color = (0, 0, 255)  # Red
#
#     # Draw nodes first
#     for node_id, node_data in node_based_graph['nodes'].items():
#         position = node_data['position']
#
#         # Draw node as a circle
#         cv2.circle(node_viz, tuple(position), 6, node_color, -1)
#
#         # Add node label
#         cv2.putText(node_viz, node_id, (position[0] - 10, position[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, node_color, 1)
#
#     # Draw connections between components and nodes
#     for comp_id, comp_data in node_based_graph['components'].items():
#         # Draw component box
#         x1, y1, x2, y2 = comp_data['box']
#         cv2.rectangle(node_viz, (x1, y1), (x2, y2), component_color, 2)
#
#         # Add component label
#         center = comp_data['position']
#         cv2.putText(node_viz, comp_id, (center[0] - 15, center[1] - 15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, component_color, 1)
#
#         # Draw connections to terminals
#         terminals = comp_data.get('terminals', {})
#         for side, node_id in terminals.items():
#             # Get position of the terminal on the component
#             terminal_pos = get_connection_point(comp_data['box'], side)
#
#             # Get position of the node
#             node_pos = node_based_graph['nodes'][node_id]['position']
#
#             # Draw line from terminal to node
#             cv2.line(node_viz, terminal_pos, tuple(node_pos), edge_color, 2)
#
#             # Add side label at midpoint
#             mid_x = (terminal_pos[0] + node_pos[0]) // 2
#             mid_y = (terminal_pos[1] + node_pos[1]) // 2
#             cv2.putText(node_viz, side[0], (mid_x, mid_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
#
#     # Draw direction indicators for directional components
#     for comp_id, comp_data in node_based_graph['components'].items():
#         if 'direction' in comp_data and comp_data['direction']:
#             start_pt = comp_data['direction']['start']
#             end_pt = comp_data['direction']['end']
#
#             # Draw direction arrow
#             cv2.arrowedLine(node_viz, start_pt, end_pt, (255, 0, 255), 2)
#
#     # Draw directed edges if provided
#     if directed_edges:
#         for edge in directed_edges:
#             from_comp = edge['from']
#             to_comp = edge['to']
#
#             # Get component centers
#             from_center = node_based_graph['components'][from_comp]['position']
#             to_center = node_based_graph['components'][to_comp]['position']
#
#             # Draw a curved arrow between components
#             control_pt = (
#                 (from_center[0] + to_center[0]) // 2,
#                 (from_center[1] + to_center[1]) // 2 - 30  # Offset for curve
#             )
#
#             # Draw a simple arrow for now
#             cv2.arrowedLine(node_viz, from_center, to_center, directed_color, 2)
#
#             # Add edge type label
#             mid_x = (from_center[0] + to_center[0]) // 2
#             mid_y = (from_center[1] + to_center[1]) // 2
#             edge_label = edge['type'][:3]  # Abbreviate type
#             cv2.putText(node_viz, edge_label, (mid_x, mid_y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, directed_color, 1)
#
#     # Save the visualization if requested
#     if output_path:
#         cv2.imwrite(output_path, node_viz)
#
#     return node_viz
#
#
# def create_networkx_node_based_graph(node_based_graph, directed_edges=None):
#     """
#     Create a NetworkX representation of the node-based circuit graph for more advanced analysis.
#
#     Args:
#         node_based_graph: Dictionary representation of the node-based circuit
#         directed_edges: Optional list of directed edges
#
#     Returns:
#         G: NetworkX graph (DiGraph if directed_edges provided, otherwise Graph)
#     """
#     if directed_edges:
#         G = nx.DiGraph()
#     else:
#         G = nx.Graph()
#
#     # Add nodes (both circuit nodes and components)
#     for node_id, node_data in node_based_graph['nodes'].items():
#         G.add_node(node_id, type='node', position=node_data['position'])
#
#     for comp_id, comp_data in node_based_graph['components'].items():
#         G.add_node(comp_id, type='component',
#                    component_type=comp_data['type'],
#                    position=comp_data['position'])
#
#     # Add edges between components and nodes
#     for comp_id, comp_data in node_based_graph['components'].items():
#         terminals = comp_data.get('terminals', {})
#         for side, node_id in terminals.items():
#             G.add_edge(comp_id, node_id, type='terminal', side=side)
#
#     # Add directed edges if provided
#     if directed_edges:
#         for edge in directed_edges:
#             G.add_edge(edge['from'], edge['to'],
#                        type='directed',
#                        from_node=edge['from_node'],
#                        to_node=edge['to_node'],
#                        edge_type=edge['type'])
#
#     return G
#
#
# def visualize_networkx_node_based_graph(G, output_path=None):
#     """
#     Create a schematic visualization of the NetworkX node-based graph.
#
#     Args:
#         G: NetworkX graph of the node-based circuit
#         output_path: Path to save the visualization
#
#     Returns:
#         fig: Matplotlib figure of the visualization
#     """
#     plt.figure(figsize=(12, 10))
#
#     # Extract node positions
#     pos = {}
#     for node, attrs in G.nodes(data=True):
#         if 'position' in attrs:
#             # Scale down positions to fit nicely on the plot
#             x, y = attrs['position']
#             pos[node] = (x / 100, -y / 100)  # Invert y for proper orientation
#
#     # If some nodes don't have positions, use spring layout for them
#     missing_nodes = [n for n in G.nodes() if n not in pos]
#     if missing_nodes:
#         missing_pos = nx.spring_layout(G.subgraph(missing_nodes))
#         pos.update(missing_pos)
#
#     # Draw nodes with different styles based on type
#     circuit_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'node']
#     component_nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('type') == 'component']
#
#     # Draw circuit nodes (connection points)
#     nx.draw_networkx_nodes(G, pos,
#                            nodelist=circuit_nodes,
#                            node_color='gold',
#                            node_size=300,
#                            node_shape='o')
#
#     # Group components by type for coloring
#     component_types = {}
#     for node in component_nodes:
#         comp_type = G.nodes[node].get('component_type', 'unknown')
#         if comp_type not in component_types:
#             component_types[comp_type] = []
#         component_types[comp_type].append(node)
#
#     # Color palette for different component types
#     colors = plt.cm.tab10.colors
#
#     # Draw each component type with a different color
#     for i, (comp_type, nodes) in enumerate(component_types.items()):
#         color = colors[i % len(colors)]
#         nx.draw_networkx_nodes(G, pos,
#                                nodelist=nodes,
#                                node_color=[color] * len(nodes),
#                                node_size=500,
#                                node_shape='s')
#
#     # Draw edges with different styles based on type
#     terminal_edges = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('type') == 'terminal']
#     directed_edges = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('type') == 'directed']
#
#     # Draw terminal connections
#     nx.draw_networkx_edges(G, pos,
#                            edgelist=terminal_edges,
#                            width=1.5,
#                            edge_color='gray')
#
#     # Draw directed connections
#     if directed_edges:
#         nx.draw_networkx_edges(G, pos,
#                                edgelist=directed_edges,
#                                width=2,
#                                edge_color='red',
#                                arrows=True,
#                                arrowstyle='-|>')
#
#     # Add labels
#     nx.draw_networkx_labels(G, pos, font_size=9)
#
#     # Add edge labels for terminal connections
#     terminal_labels = {(u, v): G.edges[u, v]['side'][:1]
#                        for u, v in terminal_edges if 'side' in G.edges[u, v]}
#     nx.draw_networkx_edge_labels(G, pos,
#                                  edge_labels=terminal_labels,
#                                  font_size=8)
#
#     # Create legend for component types
#     legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
#                                   markerfacecolor=colors[i % len(colors)],
#                                   markersize=10,
#                                   label=comp_type)
#                        for i, comp_type in enumerate(component_types)]
#
#     # Add circuit node to legend
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
#                                       markerfacecolor='gold',
#                                       markersize=10,
#                                       label='Circuit Node'))
#
#     plt.legend(handles=legend_elements, loc='upper right')
#
#     plt.title('Node-Based Circuit Graph')
#     plt.axis('off')
#     plt.tight_layout()
#
#     if output_path:
#         plt.savefig(output_path, dpi=300)
#
#     return plt.gcf()
#
#
# def update_components_with_node_information(components_data, component_nodes):
#     """
#     Update the original components_data with node connection information.
#
#     Args:
#         components_data: Original dictionary of component data
#         component_nodes: Dictionary mapping components to their terminal nodes
#
#     Returns:
#         updated_components: Updated component data with node connections
#     """
#     updated_components = {}
#
#     for comp_id, comp_data in components_data.items():
#         # Copy original component data
#         updated_components[comp_id] = comp_data.copy()
#
#         # Add node connection information
#         if comp_id in component_nodes:
#             updated_components[comp_id]['connected_nodes'] = component_nodes[comp_id]
#
#     return updated_components
#
#
# def integrate_node_based_analysis(binary_image, components_data, connections, connectivity_graph, detailed_connections):
#     """
#     Perform node-based circuit analysis and integrate with existing analysis.
#
#     Args:
#         binary_image: Binary image of the circuit
#         components_data: Dictionary of component data
#         connections: List of connections between components
#         connectivity_graph: Original undirected MultiGraph
#         detailed_connections: List of detailed connection information
#
#     Returns:
#         node_based_graph: Dictionary representation of the node-based circuit
#         networkx_graph: NetworkX representation of the node-based circuit
#         updated_components: Components data updated with node information
#     """
#     print("\nPerforming node-based circuit analysis...")
#
#     # Create node-based circuit graph
#     node_based_graph, nodes_mapping, component_nodes = create_node_based_circuit_graph(
#         components_data,
#         detailed_connections
#     )
#
#     # Identify directed edges
#     directed_edges = determine_directed_edges(node_based_graph)
#
#     # Visualize the node-based graph
#     node_viz = visualize_node_based_graph(
#         node_based_graph,
#         binary_image,
#         directed_edges,
#         "node_based_circuit.png"
#     )
#
#     # Create NetworkX representation
#     networkx_graph = create_networkx_node_based_graph(node_based_graph, directed_edges)
#
#     # Visualize the NetworkX graph
#     visualize_networkx_node_based_graph(networkx_graph, "node_based_schematic.png")
#
#     # Update components with node information
#     updated_components = update_components_with_node_information(components_data, component_nodes)
#
#     # Print node-based analysis results
#     print(f"\nNode-Based Circuit Analysis:")
#     print(f"Identified {len(node_based_graph['nodes'])} circuit nodes")
#     print(f"Identified {len(directed_edges)} directed connections")
#
#     # Print component-node connections
#     print("\nComponent-Node Connections:")
#     for comp_id, terminals in component_nodes.items():
#         terminals_str = ", ".join([f"{side}: {node}" for side, node in terminals.items()])
#         print(f"  {comp_id} connects to: {terminals_str}")
#
#     # Print node connections
#     print("\nCircuit Node Information:")
#     for node_id, node_data in node_based_graph['nodes'].items():
#         components_str = ", ".join(node_data['connected_components'])
#         print(f"  {node_id} connects components: {components_str}")
#
#     # Create a comprehensive visualization
#     create_comprehensive_visualization(
#         binary_image,
#         node_viz,
#         node_based_graph,
#         directed_edges,
#         "node_based_analysis.png"
#     )
#
#     return node_based_graph, networkx_graph, updated_components
#
#
# def create_comprehensive_visualization(binary_image, node_viz, node_based_graph, directed_edges, output_path):
#     """
#     Create a comprehensive visualization showing the binary image, node-based visualization,
#     and a circuit schematic representation.
#
#     Args:
#         binary_image: Binary image of the circuit
#         node_viz: Node-based visualization image
#         node_based_graph: Dictionary representation of the node-based circuit
#         directed_edges: List of directed edges
#         output_path: Path to save the visualization
#     """
#     plt.figure(figsize=(18, 12))
#
#     # Binary image in top-left
#     plt.subplot(221)
#     plt.imshow(binary_image, cmap='gray')
#     plt.title('Binary Circuit Image')
#     plt.axis('off')
#
#     # Node-based visualization in top-right
#     plt.subplot(222)
#     plt.imshow(cv2.cvtColor(node_viz, cv2.COLOR_BGR2RGB))
#     plt.title('Node-Based Circuit Representation')
#     plt.axis('off')
#
#     # Create a schematic in bottom half using NetworkX
#     plt.subplot(212)
#
#     # Create a directed graph for the schematic
#     G = nx.DiGraph()
#
#     # Add component nodes with positions scaled to fit nicely
#     for comp_id, comp_data in node_based_graph['components'].items():
#         x, y = comp_data['position']
#         G.add_node(comp_id,
#                    pos=(x / 100, -y / 100),  # Scale and invert y for better visualization
#                    type=comp_data['type'])
#
#     # Add circuit nodes with positions
#     for node_id, node_data in node_based_graph['nodes'].items():
#         x, y = node_data['position']
#         G.add_node(node_id,
#                    pos=(x / 100, -y / 100),
#                    type='node')
#
#     # Add edges between components based on directed_edges
#     for edge in directed_edges:
#         G.add_edge(edge['from'], edge['to'], type=edge['type'])
#
#     # Get positions from node attributes
#     pos = nx.get_node_attributes(G, 'pos')
#
#     # Draw nodes with different styles based on type
#     component_nodes = [n for n, attrs in G.nodes(data=True) if 'type' in attrs and attrs['type'] != 'node']
#     circuit_nodes = [n for n, attrs in G.nodes(data=True) if 'type' in attrs and attrs['type'] == 'node']
#
#     # Group components by type for coloring
#     component_types = {}
#     for node in component_nodes:
#         comp_type = G.nodes[node]['type']
#         if comp_type not in component_types:
#             component_types[comp_type] = []
#         component_types[comp_type].append(node)
#
#     # Draw circuit nodes
#     nx.draw_networkx_nodes(G, pos,
#                            nodelist=circuit_nodes,
#                            node_color='gold',
#                            node_size=200,
#                            alpha=0.7,
#                            node_shape='o')
#
#     # Draw components with different colors by type
#     colors = plt.cm.tab10.colors
#     for i, (comp_type, nodes) in enumerate(component_types.items()):
#         nx.draw_networkx_nodes(G, pos,
#                                nodelist=nodes,
#                                node_color=[colors[i % len(colors)]] * len(nodes),
#                                node_size=500,
#                                node_shape='s')
#
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>')
#
#     # Add labels
#     nx.draw_networkx_labels(G, pos, font_size=9)
#
#     # Create legend
#     legend_elements = [plt.Line2D([0], [0], marker='s', color='w',
#                                   markerfacecolor=colors[i % len(colors)],
#                                   markersize=10,
#                                   label=comp_type)
#                        for i, comp_type in enumerate(component_types)]
#
#     # Add circuit node to legend
#     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
#                                       markerfacecolor='gold',
#                                       markersize=10,
#                                       label='Circuit Node'))
#
#     plt.legend(handles=legend_elements, loc='upper right')
#     plt.title('Node-Based Circuit Schematic')
#     plt.axis('off')
#
#     # Add overall title
#     plt.suptitle('Node-Based Circuit Analysis', fontsize=16)
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#
#
# def modified_analyze_circuit_connectivity_nodes(image_path, final_image, detected_components):
#     """
#     Modified analysis function that includes node-based circuit representation.
#
#     Args:
#         image_path: Path to the original image
#         final_image: Binary image of the circuit
#         detected_components: Dictionary of detected component data
#
#     Returns:
#         node_based_graph: Dictionary representation of the node-based circuit
#         networkx_graph: NetworkX representation of the node-based circuit
#         updated_components: Components data updated with node information
#     """
#     try:
#         binary_image = final_image
#
#         # Analyze component connectivity using fast method
#         print("Analyzing component connectivity...")
#         connections, connectivity_graph, connectivity_image, sorted_connections, detailed_connections = \
#             analyze_component_connectivity_fast(
#                 binary_image,
#                 detected_components,
#                 "component_connectivity_fast.png"
#             )
#
#         # Add the node-based circuit analysis
#         print("\nPerforming node-based circuit analysis...")
#         node_based_graph, networkx_graph, updated_components = integrate_node_based_analysis(
#             binary_image,
#             detected_components,
#             connections,
#             connectivity_graph,
#             detailed_connections
#         )
#
#         # Create a schematic representation using the node-based graph
#         print("Creating node-based circuit schematic...")
#         visualize_networkx_node_based_graph(networkx_graph, "node_based_circuit_schematic.png")
#
#         # Create a summary visualization
#         plt.figure(figsize=(16, 12))
#
#         # Original image
#         plt.subplot(221)
#         img = cv2.imread(image_path) if os.path.exists(image_path) else np.zeros((400, 400, 3), dtype=np.uint8)
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.title('Original Circuit Image')
#         plt.axis('off')
#
#         # Binary circuit
#         plt.subplot(222)
#         plt.imshow(binary_image, cmap='gray')
#         plt.title('Binary Circuit')
#         plt.axis('off')
#
#         # Node-based visualization
#         node_viz = cv2.imread("node_based_circuit.png")
#         if node_viz is not None:
#             plt.subplot(223)
#             plt.imshow(cv2.cvtColor(node_viz, cv2.COLOR_BGR2RGB))
#             plt.title('Node-Based Circuit')
#             plt.axis('off')
#
#         # Node-based schematic
#         schematic = cv2.imread("node_based_circuit_schematic.png")
#         if schematic is not None:
#             plt.subplot(224)
#             plt.imshow(cv2.cvtColor(schematic, cv2.COLOR_BGR2RGB))
#             plt.title('Node-Based Schematic')
#             plt.axis('off')
#
#         plt.tight_layout()
#         plt.savefig("node_based_analysis_summary.png", dpi=300)
#         plt.close()
#
#         # Print node-based circuit summary
#         print("\nNode-Based Circuit Analysis Summary:")
#         print(f"Components: {len(detected_components)}")
#         print(f"Circuit Nodes: {len(node_based_graph['nodes'])}")
#         print(
#             f"Directed Connections: {len([e for e in networkx_graph.edges() if networkx_graph.get_edge_data(*e).get('type') == 'directed'])}")
#
#         return node_based_graph, networkx_graph, updated_components
#
#     except Exception as e:
#         print(f"An error occurred during node-based analysis: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None, None, None
#
#
# def main_with_node_based_analysis(image_path):
#     """Main function with node-based graph analysis"""
#     # Process the circuit image to get binary image and components
#     final_image, detected_components = process_circuit_image(image_path)
#
#     # Analyze circuit connectivity with node-based analysis
#     node_based_graph, networkx_graph, updated_components = modified_analyze_circuit_connectivity_nodes(
#         image_path, final_image, detected_components
#     )
#
#     return node_based_graph, networkx_graph, updated_components

# print(main_with_node_based_analysis('circuit4.jpg'))