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