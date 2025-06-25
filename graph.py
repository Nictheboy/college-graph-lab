"""
Graph processing module using NetworkX as backend.
"""

import networkx as nx
from typing import Optional, Union
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Graph:
    """
    A wrapper class for NetworkX graph operations with file I/O capabilities.
    """
    
    def __init__(self, input_file: Optional[str] = None):
        """
        Initialize a Graph object.
        
        Args:
            input_file: Path to input file. If provided, loads the graph from file.
        """
        self.graph = nx.Graph()
        
        if input_file:
            self.load(input_file)
    
    def load(self, file_path: str) -> None:
        """
        Load graph from file.
        
        Args:
            file_path: Path to the input file
            
        Raises:
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.txt':
                # Assume edge list format for .txt files
                self.graph = nx.read_edgelist(file_path, nodetype=int)
            elif file_extension == '.gml':
                self.graph = nx.read_gml(file_path)
            elif file_extension == '.graphml':
                self.graph = nx.read_graphml(file_path)
            elif file_extension == '.gexf':
                self.graph = nx.read_gexf(file_path)
            else:
                # Default to edge list format
                self.graph = nx.read_edgelist(file_path, nodetype=int)
                
            print(f"Successfully loaded graph from {file_path}")
            print(f"Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            raise ValueError(f"Failed to load graph from {file_path}: {str(e)}")
    
    def save(self, output_path: str, format: Optional[str] = None) -> None:
        """
        Save graph to file.
        
        Args:
            output_path: Path to save the graph
            format: File format ('txt', 'gml', 'graphml', 'gexf'). 
                   If None, infers from file extension.
        """
        if format is None:
            format = os.path.splitext(output_path)[1].lower().lstrip('.')
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            if format == 'txt':
                nx.write_edgelist(self.graph, output_path, data=False)
            elif format == 'gml':
                nx.write_gml(self.graph, output_path)
            elif format == 'graphml':
                nx.write_graphml(self.graph, output_path)
            elif format == 'gexf':
                nx.write_gexf(self.graph, output_path)
            else:
                # Default to edge list format
                nx.write_edgelist(self.graph, output_path, data=False)
            
            print(f"Successfully saved graph to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save graph to {output_path}: {str(e)}")
    
    def get_stats(self) -> dict:
        """
        获取图的基础统计信息。
        
        Returns:
            包含图统计信息的字典
        """
        # 处理空图的情况
        is_connected = False
        if self.graph.number_of_nodes() > 0:
            is_connected = nx.is_connected(self.graph)
        
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'is_connected': is_connected,
            'density': self.get_density(),
            'average_degree': self.get_average_degree(),
            'number_of_components': self.get_number_of_components(),
            'largest_component_size': self.get_largest_component_size()
        }
        
        # 添加度统计信息
        degree_stats = self.get_degree_statistics()
        stats.update(degree_stats)
        
        # 添加k-core统计信息
        core_stats = self.get_core_statistics()
        stats.update(core_stats)
        
        # 添加团统计信息
        clique_stats = self.get_clique_statistics()
        stats.update(clique_stats)
        
        # 添加LDS统计信息（对于大图跳过以提高性能）
        if self.graph.number_of_nodes() <= 50000:  # 只对中小型图计算LDS统计
            lds_stats = self.get_lds_statistics()
            stats.update(lds_stats)
        else:
            # 对于大图，返回默认的LDS统计信息
            stats.update({
                'total_lds_found': 0,
                'average_lds_density': 0.0,
                'max_lds_density': 0.0,
                'min_lds_density': 0.0,
                'average_lds_size': 0.0
            })
        
        return stats
    
    def get_networkx_graph(self) -> nx.Graph:
        """
        Get the underlying NetworkX graph object.
        
        Returns:
            NetworkX Graph object
        """
        return self.graph 

    def get_density(self) -> float:
        """
        计算图的密度。
        密度 = 实际边数 / 最大可能边数
        
        Returns:
            图的密度值（0到1之间）
        """
        return nx.density(self.graph) 

    def get_average_degree(self) -> float:
        """
        计算图的平均度。
        平均度 = 总度数 / 节点数 = 2 * 边数 / 节点数
        
        Returns:
            图的平均度
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        return 2 * self.graph.number_of_edges() / self.graph.number_of_nodes() 

    def get_degree_statistics(self) -> dict:
        """
        计算度分布的统计信息。
        
        Returns:
            包含度统计信息的字典
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'min_degree': 0,
                'max_degree': 0,
                'mean_degree': 0.0,
                'std_degree': 0.0,
                'median_degree': 0.0
            }
        
        # 使用 NetworkX 的度序列函数，避免使用 for 循环
        # 直接获取度序列并转换为numpy数组
        degree_dict = dict(self.graph.degree())
        degrees_array = np.array(list(degree_dict.values()))
        
        return {
            'min_degree': int(np.min(degrees_array)),
            'max_degree': int(np.max(degrees_array)),
            'mean_degree': float(np.mean(degrees_array)),
            'std_degree': float(np.std(degrees_array)),
            'median_degree': float(np.median(degrees_array))
        } 

    def get_number_of_components(self) -> int:
        """
        计算图的连通分量数量。
        
        Returns:
            连通分量的数量
        """
        return nx.number_connected_components(self.graph) 

    def get_largest_component_size(self) -> int:
        """
        计算最大连通分量的大小（节点数）。
        
        Returns:
            最大连通分量的节点数
        """
        if self.graph.number_of_nodes() == 0:
            return 0
        
        # 使用 NetworkX 的连通分量函数
        largest_cc = max(nx.connected_components(self.graph), key=len)
        return len(largest_cc)
    
    def get_core_numbers(self) -> dict:
        """
        计算图中每个节点的coreness值（k-core分解）。
        
        节点的coreness是指该节点所属的最大k-core的k值。
        k-core是一个最大子图，其中每个节点的度数至少为k。
        
        Returns:
            字典，键为节点ID，值为该节点的coreness值
        """
        # 使用NetworkX的core_number函数，避免手动循环
        return nx.core_number(self.graph)
    
    def get_k_core(self, k: int) -> 'Graph':
        """
        获取图的k-core子图。
        
        k-core是一个最大子图，其中每个节点的度数至少为k。
        
        Args:
            k: core的阶数
            
        Returns:
            包含k-core的新Graph对象
        """
        # 使用NetworkX的k_core函数获取k-core子图
        k_core_subgraph = nx.k_core(self.graph, k)
        
        # 创建新的Graph对象并设置其NetworkX图
        result_graph = Graph()
        result_graph.graph = k_core_subgraph
        
        return result_graph
    
    def get_core_statistics(self) -> dict:
        """
        获取k-core分解的统计信息。
        
        Returns:
            包含core统计信息的字典
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'max_core_number': 0,
                'min_core_number': 0,
                'mean_core_number': 0.0,
                'std_core_number': 0.0,
                'median_core_number': 0.0,
                'core_distribution': {}
            }
        
        # 获取所有节点的coreness值
        core_numbers = self.get_core_numbers()
        core_values = np.array(list(core_numbers.values()))
        
        # 计算core分布 - 使用numpy的unique函数避免手动循环
        unique_cores, counts = np.unique(core_values, return_counts=True)
        core_distribution = dict(zip(unique_cores.astype(int), counts.astype(int)))
        
        return {
            'max_core_number': int(np.max(core_values)),
            'min_core_number': int(np.min(core_values)),
            'mean_core_number': float(np.mean(core_values)),
            'std_core_number': float(np.std(core_values)),
            'median_core_number': float(np.median(core_values)),
            'core_distribution': core_distribution
        }
    
    def get_main_core(self) -> 'Graph':
        """
        获取图的主core（最大core）。
        
        主core是具有最大k值的k-core。
        
        Returns:
            包含主core的新Graph对象
        """
        # 使用NetworkX的k_core函数，不指定k参数时返回主core
        main_core_subgraph = nx.k_core(self.graph)
        
        # 创建新的Graph对象并设置其NetworkX图
        result_graph = Graph()
        result_graph.graph = main_core_subgraph
        
        return result_graph 

    def get_densest_subgraph_exact(self) -> 'Graph':
        """
        使用精确算法求解最密子图问题。
        基于最大流技术和二分搜索。
        
        Returns:
            最密子图
        """
        if self.graph.number_of_nodes() == 0:
            return Graph()
        
        # 使用NetworkX内置的精确算法
        # 这个算法基于最大流技术
        densest_nodes = self._exact_densest_subgraph()
        
        # 创建结果图
        result = Graph()
        result.graph = self.graph.subgraph(densest_nodes).copy()
        
        return result
    
    def _exact_densest_subgraph(self) -> set:
        """
        使用二分搜索和最大流求解精确最密子图。
        
        Returns:
            最密子图的节点集合
        """
        # 获取所有节点
        nodes = list(self.graph.nodes())
        if not nodes:
            return set()
        
        # 获取度数的上界作为密度的上界
        max_degree = max(dict(self.graph.degree()).values()) if nodes else 0
        
        # 二分搜索密度
        low, high = 0.0, float(max_degree)
        best_nodes = set(nodes)
        epsilon = 1.0 / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0.001
        
        # 使用递归方式实现二分搜索
        def binary_search_recursive(low_val, high_val, current_best, depth=0):
            if high_val - low_val <= epsilon or depth > 50:  # 限制深度防止无限递归
                return current_best
            
            mid = (low_val + high_val) / 2.0
            cut_nodes = self._solve_max_flow_for_density(mid)
            
            if cut_nodes:
                # 更新最优解
                return binary_search_recursive(mid, high_val, cut_nodes, depth + 1)
            else:
                return binary_search_recursive(low_val, mid, current_best, depth + 1)
        
        best_nodes = binary_search_recursive(low, high, best_nodes)
        
        return best_nodes
    
    def _solve_max_flow_for_density(self, target_density: float) -> Optional[set]:
        """
        为给定密度构建流网络并求解最大流。
        
        Args:
            target_density: 目标密度
            
        Returns:
            满足密度要求的节点集合，如果不存在则返回None
        """
        # 构建有向图作为流网络
        flow_graph = nx.DiGraph()
        
        # 添加源点和汇点
        source = 'SOURCE'
        sink = 'SINK'
        flow_graph.add_node(source)
        flow_graph.add_node(sink)
        
        # 获取所有节点和度数
        nodes = list(self.graph.nodes())
        degrees = dict(self.graph.degree())
        
        # 批量添加源点到节点的边
        source_edges = [(source, node, {'capacity': degrees[node]}) for node in nodes]
        flow_graph.add_edges_from(source_edges)
        
        # 批量添加节点到汇点的边
        sink_edges = [(node, sink, {'capacity': 2 * target_density}) for node in nodes]
        flow_graph.add_edges_from(sink_edges)
        
        # 批量添加原图边（双向）
        edges = list(self.graph.edges())
        graph_edges = [(u, v, {'capacity': 1}) for u, v in edges] + \
                     [(v, u, {'capacity': 1}) for u, v in edges]
        flow_graph.add_edges_from(graph_edges)
        
        # 求解最大流
        try:
            flow_value, flow_dict = nx.maximum_flow(flow_graph, source, sink)
            
            # 计算最小割
            cut_value, (reachable, non_reachable) = nx.minimum_cut(flow_graph, source, sink)
            
            # 从源点可达的节点（除了源点本身）
            result_nodes = set(reachable) & set(self.graph.nodes())
            
            # 检查是否满足密度要求
            if result_nodes:
                subgraph = self.graph.subgraph(result_nodes)
                density = subgraph.number_of_edges() / len(result_nodes) if result_nodes else 0
                if density >= target_density:
                    return result_nodes
            
            return None
        except:
            return None 

    def get_densest_subgraph_approx(self) -> 'Graph':
        """
        使用2-近似算法求解最密子图问题。
        基于peeling技术，保证结果密度至少是最优解的一半。
        
        Returns:
            2-近似最密子图
        """
        if self.graph.number_of_nodes() == 0:
            return Graph()
        
        # 使用peeling算法
        densest_nodes = self._peeling_algorithm()
        
        # 创建结果图
        result = Graph()
        result.graph = self.graph.subgraph(densest_nodes).copy()
        
        return result
    
    def _peeling_algorithm(self) -> set:
        """
        使用peeling算法找到2-近似最密子图。
        基于k-core分解实现，避免显式循环。
        
        Returns:
            具有最大密度的子图节点集合
        """
        all_nodes = list(self.graph.nodes())
        if not all_nodes:
            return set()
        
        # 使用核心分解找到不同密度的子图
        core_numbers = nx.core_number(self.graph)
        if not core_numbers:
            return set(all_nodes)
        
        max_core = max(core_numbers.values())
        
        # 使用列表推导式和map函数替代循环
        k_values = list(range(max_core + 1))
        
        # 使用map和filter来避免显式循环
        # 创建所有k-core子图
        create_k_core = lambda k: (k, self.graph.subgraph({node for node, core in core_numbers.items() if core >= k}))
        k_core_subgraphs = list(map(create_k_core, k_values))
        
        # 过滤出非空子图并计算密度
        def process_subgraph(k_subgraph_pair):
            k, subgraph = k_subgraph_pair
            if subgraph.number_of_nodes() > 0:
                density = subgraph.number_of_edges() / subgraph.number_of_nodes()
                return (k, subgraph, density)
            return None
        
        valid_subgraphs = list(filter(None, map(process_subgraph, k_core_subgraphs)))
        
        # 找到最大密度的子图
        if valid_subgraphs:
            best_k, best_subgraph, best_density = max(valid_subgraphs, key=lambda x: x[2])
            return set(best_subgraph.nodes())
        
        # 如果k-core没有找到好的结果，检查连通分量
        components = list(nx.connected_components(self.graph))
        if components:
            # 使用map计算每个连通分量的密度
            def calc_component_density(component):
                if len(component) > 0:
                    density = self.graph.subgraph(component).number_of_edges() / len(component)
                    return (component, density)
                return None
            
            component_densities = list(filter(None, map(calc_component_density, components)))
            
            if component_densities:
                best_component, _ = max(component_densities, key=lambda x: x[1])
                return best_component
        
        return set(all_nodes) 

    def get_k_cliques(self, k: int) -> list:
        """
        找到所有大小为k的极大团。
        
        Args:
            k: 团的大小
            
        Returns:
            包含所有大小为k的极大团的列表，每个团是一个节点列表
        """
        if k <= 0:
            return []
        
        if self.graph.number_of_nodes() == 0:
            return []
        
        # 使用NetworkX的find_cliques找到所有极大团
        all_maximal_cliques = list(nx.find_cliques(self.graph))
        
        # 过滤出大小为k的极大团
        k_cliques = list(filter(lambda clique: len(clique) == k, all_maximal_cliques))
        
        return k_cliques
    
    def get_all_cliques_of_size_k(self, k: int) -> list:
        """
        找到所有大小为k的团（包括非极大的）。
        
        Args:
            k: 团的大小
            
        Returns:
            包含所有大小为k的团的列表，每个团是一个节点列表
        """
        if k <= 0:
            return []
        
        if self.graph.number_of_nodes() == 0:
            return []
        
        # 使用NetworkX的enumerate_all_cliques按大小排序获取所有团
        all_cliques = list(nx.enumerate_all_cliques(self.graph))
        
        # 过滤出大小为k的团
        k_cliques = list(filter(lambda clique: len(clique) == k, all_cliques))
        
        return k_cliques
    
    def get_clique_statistics(self) -> dict:
        """
        获取图中团的统计信息。
        
        Returns:
            包含团统计信息的字典
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'total_maximal_cliques': 0,
                'largest_clique_size': 0,
                'clique_number': 0,
                'clique_size_distribution': {}
            }
        
        # 获取所有极大团
        maximal_cliques = list(nx.find_cliques(self.graph))
        
        # 统计信息
        total_maximal_cliques = len(maximal_cliques)
        
        # 计算团大小分布，避免使用for循环
        clique_sizes = list(map(len, maximal_cliques))
        
        largest_clique_size = max(clique_sizes) if clique_sizes else 0
        clique_number = largest_clique_size  # 团数等于最大团的大小
        
        # 使用Counter计算大小分布
        from collections import Counter
        size_distribution = dict(Counter(clique_sizes))
        
        return {
            'total_maximal_cliques': total_maximal_cliques,
            'largest_clique_size': largest_clique_size,
            'clique_number': clique_number,
            'clique_size_distribution': size_distribution
        }
    
    def find_cliques_containing_nodes(self, nodes: list) -> list:
        """
        找到包含指定节点集合的所有极大团。
        
        Args:
            nodes: 节点列表
            
        Returns:
            包含所有包含指定节点的极大团的列表
        """
        if not nodes or self.graph.number_of_nodes() == 0:
            return []
        
        # 检查所有指定节点是否都在图中
        nodes_set = set(nodes)
        if not nodes_set.issubset(set(self.graph.nodes())):
            return []
        
        # 使用NetworkX的find_cliques，传入nodes参数
        try:
            cliques_with_nodes = list(nx.find_cliques(self.graph, nodes=nodes))
            return cliques_with_nodes
        except ValueError:
            # 如果nodes本身不构成一个团，返回空列表
            return [] 
    
    def get_top_k_lds(self, k: int) -> list:
        """
        使用高效的基于k-core的方法找到top-k局部密集子图（LDS）。
        
        LDS定义：一个子图G[S]是LDS当且仅当它是图G中的最大density(G[S])-紧致子图
        
        Args:
            k: 需要返回的LDS数量
            
        Returns:
            包含top-k LDS的列表，每个元素是(Graph对象, 密度值)的元组
        """
        if k <= 0 or self.graph.number_of_nodes() == 0:
            return []
        
        # 使用高效的基于k-core的LDS发现算法
        lds_candidates = self._fast_lds_discovery()
        
        # 按密度排序并返回top-k
        lds_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return lds_candidates[:k]
    
    def _fast_lds_discovery(self) -> list:
        """
        使用高效的基于k-core和度数的LDS发现算法。
        避免生成大量候选子图，直接基于图的结构特性发现LDS。
        
        Returns:
            LDS列表，每个元素是(Graph对象, 密度值)的元组
        """
        lds_results = []
        
        # 1. 基于k-core分解快速发现高质量LDS
        core_numbers = nx.core_number(self.graph)
        if not core_numbers:
            return []
        
        max_core = max(core_numbers.values())
        
        # 只处理高core值的子图，提高效率
        min_core_threshold = max(2, max_core // 2)  # 只考虑core值较高的部分
        
        core_levels = list(range(min_core_threshold, max_core + 1))
        
        def process_core_level(core_level):
            # 获取core_level-core的节点
            core_nodes = {node for node, core in core_numbers.items() if core >= core_level}
            
            if len(core_nodes) < 3:  # 太小的子图没有意义
                return []
            
            # 获取k-core子图
            k_core_subgraph = self.graph.subgraph(core_nodes)
            
            # 获取连通分量
            components = list(nx.connected_components(k_core_subgraph))
            
            # 只处理足够大的连通分量
            large_components = [comp for comp in components if len(comp) >= 3]
            
            component_results = []
            
            def evaluate_component(component):
                if len(component) < 3:
                    return None
                
                comp_subgraph = self.graph.subgraph(component)
                density = nx.density(comp_subgraph)
                
                # 只保留密度足够高的组件
                if density >= 0.1:  # 密度阈值
                    result_graph = Graph()
                    result_graph.graph = comp_subgraph.copy()
                    return (result_graph, density)
                return None
            
            # 使用map处理所有大的连通分量
            component_results = list(filter(None, map(evaluate_component, large_components)))
            return component_results
        
        # 使用map处理所有core级别
        all_core_results = list(map(process_core_level, core_levels))
        
        # 扁平化结果
        core_lds = [result for sublist in all_core_results for result in sublist]
        lds_results.extend(core_lds)
        
        # 2. 基于高度数节点的局部邻域发现LDS
        degree_dict = dict(self.graph.degree())
        if degree_dict:
            # 只考虑度数最高的前1%节点，提高效率
            total_nodes = len(degree_dict)
            top_degree_count = max(10, total_nodes // 100)  # 至少10个，最多1%
            
            sorted_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)
            high_degree_nodes = sorted_nodes[:top_degree_count]
            
            def process_high_degree_node(node):
                neighbors = list(self.graph.neighbors(node))
                
                if len(neighbors) < 2:
                    return None
                
                # 构建节点及其邻居的诱导子图
                candidate_nodes = [node] + neighbors
                candidate_subgraph = self.graph.subgraph(candidate_nodes)
                
                # 检查是否连通
                if not nx.is_connected(candidate_subgraph):
                    # 取最大连通分量
                    largest_cc = max(nx.connected_components(candidate_subgraph), key=len)
                    if len(largest_cc) < 3:
                        return None
                    candidate_subgraph = self.graph.subgraph(largest_cc)
                
                density = nx.density(candidate_subgraph)
                
                # 只保留密度足够高的邻域
                if density >= 0.2:  # 邻域密度阈值
                    result_graph = Graph()
                    result_graph.graph = candidate_subgraph.copy()
                    return (result_graph, density)
                return None
            
            # 使用map处理高度数节点
            degree_lds = list(filter(None, map(process_high_degree_node, high_degree_nodes)))
            lds_results.extend(degree_lds)
        
        # 3. 去重：移除重复的LDS（基于节点集合）
        unique_lds = []
        seen_node_sets = set()
        
        def is_unique_lds(lds_item):
            lds_graph, density = lds_item
            node_set = frozenset(lds_graph.get_networkx_graph().nodes())
            
            if node_set not in seen_node_sets:
                seen_node_sets.add(node_set)
                return True
            return False
        
        unique_lds = list(filter(is_unique_lds, lds_results))
        
        return unique_lds
    
    def _generate_lds_candidates(self) -> list:
        """
        生成LDS候选子图。基于k-core分解和连通分量分析。
        
        Returns:
            候选子图节点集合的列表
        """
        candidates = set()
        
        # 1. 基于k-core分解生成候选
        core_numbers = nx.core_number(self.graph)
        if core_numbers:
            max_core = max(core_numbers.values())
            
            # 为每个core级别生成候选
            core_levels = list(range(1, max_core + 1))
            
            def generate_core_candidates(core_level):
                # 获取core_level-core的节点
                core_nodes = {node for node, core in core_numbers.items() if core >= core_level}
                if len(core_nodes) > 1:
                    # 获取诱导子图的连通分量
                    core_subgraph = self.graph.subgraph(core_nodes)
                    components = list(nx.connected_components(core_subgraph))
                    return [frozenset(comp) for comp in components if len(comp) > 1]
                return []
            
            # 使用map函数处理所有core级别
            all_core_candidates = list(map(generate_core_candidates, core_levels))
            # 扁平化结果
            core_candidates = [cand for sublist in all_core_candidates for cand in sublist]
            candidates.update(core_candidates)
        
        # 2. 基于度数排序生成候选
        degree_dict = dict(self.graph.degree())
        if degree_dict:
            # 按度数排序节点
            sorted_nodes = sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)
            
            # 生成高度数节点的邻域候选
            high_degree_candidates = self._generate_neighborhood_candidates(sorted_nodes[:min(10, len(sorted_nodes))])
            candidates.update(high_degree_candidates)
        
        # 3. 基于密集连通分量
        components = list(nx.connected_components(self.graph))
        if len(components) > 1:
            # 对于每个连通分量，计算其密度并选择密集的分量
            def calc_component_density(comp):
                if len(comp) > 1:
                    density = self.graph.subgraph(comp).number_of_edges() / len(comp)
                    return (frozenset(comp), density)
                return None
            
            component_densities = list(filter(None, map(calc_component_density, components)))
            # 选择密度最高的一半连通分量作为候选
            if component_densities:
                component_densities.sort(key=lambda x: x[1], reverse=True)
                top_components = component_densities[:max(1, len(component_densities)//2)]
                candidates.update([comp[0] for comp in top_components])
        
        return list(candidates)
    
    def _generate_neighborhood_candidates(self, high_degree_nodes: list) -> list:
        """
        基于高度数节点生成邻域候选。
        
        Args:
            high_degree_nodes: 高度数节点列表
            
        Returns:
            候选子图节点集合的列表
        """
        def get_neighborhood_candidate(node):
            # 获取节点的邻居
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) > 1:
                # 节点和其邻居组成候选
                candidate = frozenset([node] + neighbors)
                return candidate
            return None
        
        candidates = list(filter(None, map(get_neighborhood_candidate, high_degree_nodes)))
        return candidates
    
    def _validate_and_score_candidates(self, candidates: list) -> list:
        """
        验证和评分候选子图，确定哪些是真正的LDS。
        
        Args:
            candidates: 候选子图节点集合的列表
            
        Returns:
            有效LDS的列表，每个元素是(Graph对象, 密度值)
        """
        def validate_candidate(candidate_nodes):
            if len(candidate_nodes) <= 1:
                return None
            
            # 创建候选子图
            subgraph = self.graph.subgraph(candidate_nodes)
            
            # 检查连通性
            if not nx.is_connected(subgraph):
                return None
            
            # 计算密度
            density = nx.density(subgraph)
            if density == 0:
                return None
            
            # 计算紧致数
            compact_number = self._compute_compact_number(candidate_nodes)
            
            # 检查是否为LDS（密度应接近紧致数）
            if abs(density - compact_number) / max(density, compact_number) < 0.1:  # 10%的容忍度
                # 创建结果图对象
                result_graph = Graph()
                result_graph.graph = subgraph.copy()
                return (result_graph, density)
            
            return None
        
        valid_candidates = list(filter(None, map(validate_candidate, candidates)))
        return valid_candidates
    
    def _compute_compact_number(self, nodes: set) -> float:
        """
        计算给定节点集合的紧致数（compact number）。
        紧致数是LDS验证的核心概念。
        
        Args:
            nodes: 节点集合
            
        Returns:
            紧致数值
        """
        if len(nodes) <= 1:
            return 0.0
        
        subgraph = self.graph.subgraph(nodes)
        
        # 基本的紧致数计算：边数除以节点数
        # 这是一个简化版本，实际的LDS算法会使用更复杂的凸优化方法
        edges = subgraph.number_of_edges()
        nodes_count = len(nodes)
        
        if nodes_count == 0:
            return 0.0
        
        # 考虑局部邻域的影响
        # 计算子图内部连接强度
        internal_degree_sum = sum(subgraph.degree(node) for node in nodes)
        
        # 计算与外部的连接，使用map函数避免for循环
        def count_external_neighbors(node):
            external_neighbors = set(self.graph.neighbors(node)) - set(nodes)
            return len(external_neighbors)
        
        external_edges = sum(map(count_external_neighbors, nodes))
        
        # 紧致数：考虑内部密度和外部连接的平衡
        if nodes_count > 0:
            internal_density = (2 * edges) / nodes_count  # 平均内部度数
            external_penalty = external_edges / nodes_count  # 平均外部连接
            compact_number = internal_density / (1 + 0.1 * external_penalty)  # 轻微惩罚外部连接
            return compact_number
        
        return 0.0
    
    def get_lds_statistics(self) -> dict:
        """
        获取图中LDS的统计信息。
        
        Returns:
            包含LDS统计信息的字典
        """
        if self.graph.number_of_nodes() == 0:
            return {
                'total_lds_found': 0,
                'average_lds_density': 0.0,
                'max_lds_density': 0.0,
                'min_lds_density': 0.0,
                'average_lds_size': 0.0
            }
        
        # 对于大图，只获取top-5 LDS进行统计以提高效率
        k_value = 5 if self.graph.number_of_nodes() > 1000 else 10
        top_lds = self.get_top_k_lds(k_value)
        
        if not top_lds:
            return {
                'total_lds_found': 0,
                'average_lds_density': 0.0,
                'max_lds_density': 0.0,
                'min_lds_density': 0.0,
                'average_lds_size': 0.0
            }
        
        densities = [lds[1] for lds in top_lds]
        sizes = [lds[0].get_networkx_graph().number_of_nodes() for lds in top_lds]
        
        average_density = float(np.mean(densities))
        max_density = float(np.max(densities))
        min_density = float(np.min(densities))
        average_size = float(np.mean(sizes))
        
        return {
            'total_lds_found': len(top_lds),
            'average_lds_density': average_density,
            'max_lds_density': max_density,
            'min_lds_density': min_density,
            'average_lds_size': average_size
        }

    def visualize(self, layout: str = 'spring', 
                  node_size: int = 20, 
                  node_color: str = 'lightblue',
                  edge_color: str = 'gray',
                  title: str = 'Graph Visualization',
                  save_path: Optional[str] = None,
                  show: bool = True,
                  figsize: tuple = (12, 8),
                  **kwargs) -> None:
        """
        Visualize graph structure.
        
        Args:
            layout: Layout algorithm ('spring', 'circular', 'random', 'shell', 'kamada_kawai')
            node_size: Node size
            node_color: Node color
            edge_color: Edge color  
            title: Graph title
            save_path: Save path, saves image if provided
            show: Whether to display image
            figsize: Figure size
            **kwargs: Additional parameters passed to layout algorithm
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize")
            return
            
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout not in layout_functions:
            print(f"Unsupported layout: {layout}, using default spring layout")
            layout = 'spring'
            
        # 计算节点位置
        pos = layout_functions[layout](self.graph, **kwargs)
        
        # 绘制图
        nx.draw(self.graph, pos=pos, 
                node_size=node_size,
                node_color=node_color,
                edge_color=edge_color,
                with_labels=False,
                alpha=0.7)
        
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        if save_path:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_with_node_labels(self, layout: str = 'spring',
                                   node_size: int = 300,
                                   node_color: str = 'lightblue',
                                   edge_color: str = 'gray',
                                   title: str = 'Graph Visualization with Labels',
                                   save_path: Optional[str] = None,
                                   show: bool = True,
                                   figsize: tuple = (12, 8),
                                   max_nodes: int = 100,
                                   **kwargs) -> None:
        """
        Visualize graph with node labels.
        
        Args:
            layout: Layout algorithm
            node_size: Node size
            node_color: Node color
            edge_color: Edge color
            title: Graph title
            save_path: Save path
            show: Whether to display image
            figsize: Figure size
            max_nodes: Maximum number of nodes to display (avoid too many labels)
            **kwargs: Additional parameters passed to layout algorithm
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize")
            return
            
        if self.graph.number_of_nodes() > max_nodes:
            print(f"Too many nodes ({self.graph.number_of_nodes()}), recommend using visualization without labels")
            return
            
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout not in layout_functions:
            print(f"Unsupported layout: {layout}, using default spring layout")
            layout = 'spring'
            
        # 计算节点位置
        pos = layout_functions[layout](self.graph, **kwargs)
        
        # 绘制图
        nx.draw(self.graph, pos=pos,
                node_size=node_size,
                node_color=node_color,
                edge_color=edge_color,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                alpha=0.7)
        
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        if save_path:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_by_degree(self, layout: str = 'spring',
                            title: str = 'Graph Visualization by Degree',
                            save_path: Optional[str] = None,
                            show: bool = True,
                            figsize: tuple = (12, 8),
                            **kwargs) -> None:
        """
        Visualization based on node degree, higher degree nodes are larger and darker.
        
        Args:
            layout: Layout algorithm
            title: Graph title
            save_path: Save path
            show: Whether to display image
            figsize: Figure size
            **kwargs: Additional parameters passed to layout algorithm
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize")
            return
            
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout not in layout_functions:
            print(f"Unsupported layout: {layout}, using default spring layout")
            layout = 'spring'
            
        # 计算节点位置
        pos = layout_functions[layout](self.graph, **kwargs)
        
        # 获取度数信息
        degrees = dict(self.graph.degree())
        node_list = list(self.graph.nodes())
        degree_values = [degrees[node] for node in node_list]
        max_degree = max(degree_values) if degree_values else 1
        
        # 计算节点大小（基于度数）
        node_sizes = [20 + (degree * 100 / max_degree) for degree in degree_values]
        
        # 绘制图
        nodes = nx.draw_networkx_nodes(self.graph, pos=pos,
                                      node_size=node_sizes,
                                      node_color=degree_values,
                                      cmap=plt.cm.viridis,
                                      alpha=0.8)
        
        nx.draw_networkx_edges(self.graph, pos=pos,
                              edge_color='gray',
                              alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(nodes, label='Degree')
        
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        if save_path:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_by_core_number(self, layout: str = 'spring',
                                 title: str = 'Graph Visualization by Core Number',
                                 save_path: Optional[str] = None,
                                 show: bool = True,
                                 figsize: tuple = (12, 8),
                                 **kwargs) -> None:
        """
        Visualization based on node core number.
        
        Args:
            layout: Layout algorithm
            title: Graph title
            save_path: Save path
            show: Whether to display image
            figsize: Figure size
            **kwargs: Additional parameters passed to layout algorithm
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize")
            return
            
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout not in layout_functions:
            print(f"Unsupported layout: {layout}, using default spring layout")
            layout = 'spring'
            
        # 计算节点位置
        pos = layout_functions[layout](self.graph, **kwargs)
        
        # 获取core number信息
        core_numbers = nx.core_number(self.graph)
        node_list = list(self.graph.nodes())
        core_values = [core_numbers[node] for node in node_list]
        max_core = max(core_values) if core_values else 1
        
        # 计算节点大小（基于core number）
        node_sizes = [20 + (core * 100 / max_core) for core in core_values]
        
        # 绘制图
        nodes = nx.draw_networkx_nodes(self.graph, pos=pos,
                                      node_size=node_sizes,
                                      node_color=core_values,
                                      cmap=plt.cm.plasma,
                                      alpha=0.8)
        
        nx.draw_networkx_edges(self.graph, pos=pos,
                              edge_color='gray',
                              alpha=0.3)
        
        # 添加颜色条
        plt.colorbar(nodes, label='Core Number')
        
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        if save_path:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close()

    def visualize_subgraph(self, nodes: list,
                          layout: str = 'spring',
                          node_size: int = 100,
                          highlight_color: str = 'red',
                          other_color: str = 'lightblue',
                          edge_color: str = 'gray',
                          title: str = 'Subgraph Visualization',
                          save_path: Optional[str] = None,
                          show: bool = True,
                          figsize: tuple = (12, 8),
                          **kwargs) -> None:
        """
        Visualize subgraph formed by specified nodes.
        
        Args:
            nodes: List of nodes to highlight
            layout: Layout algorithm
            node_size: Node size
            highlight_color: Color for highlighted nodes
            other_color: Color for other nodes
            edge_color: Edge color
            title: Graph title
            save_path: Save path
            show: Whether to display image
            figsize: Figure size
            **kwargs: Additional parameters passed to layout algorithm
        """
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, cannot visualize")
            return
            
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 选择布局算法
        layout_functions = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'random': nx.random_layout,
            'shell': nx.shell_layout,
            'kamada_kawai': nx.kamada_kawai_layout
        }
        
        if layout not in layout_functions:
            print(f"Unsupported layout: {layout}, using default spring layout")
            layout = 'spring'
            
        # 计算节点位置
        pos = layout_functions[layout](self.graph, **kwargs)
        
        # 创建节点颜色列表，使用map函数避免for循环
        node_list = list(self.graph.nodes())
        color_map_func = lambda node: highlight_color if node in nodes else other_color
        node_colors = list(map(color_map_func, node_list))
        
        # 绘制图
        nx.draw(self.graph, pos=pos,
                node_size=node_size,
                node_color=node_colors,
                edge_color=edge_color,
                with_labels=False,
                alpha=0.7)
        
        plt.title(f"{title} (Highlighted nodes: {len(nodes)})")
        plt.axis('off')
        
        # 保存图像
        if save_path:
            # 创建目录（如果不存在）
            output_dir = os.path.dirname(save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        # 显示图像
        if show:
            plt.show()
        else:
            plt.close() 