"""
Graph processing module using NetworkX as backend.
"""

import networkx as nx
from typing import Optional, Union
import os
import numpy as np


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