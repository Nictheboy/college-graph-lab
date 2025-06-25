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
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'is_connected': nx.is_connected(self.graph),
            'density': self.get_density(),
            'average_degree': self.get_average_degree(),
            'number_of_components': self.get_number_of_components(),
            'largest_component_size': self.get_largest_component_size()
        }
        
        # 添加度统计信息
        degree_stats = self.get_degree_statistics()
        stats.update(degree_stats)
        
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
        degrees = [d for n, d in self.graph.degree()]
        degrees_array = np.array(degrees)
        
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