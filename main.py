"""
Main script to test the Graph class functionality.
"""

from graph import Graph
import os


def main():
    """
    Test the Graph class with different data files.
    """
    print("=== Graph Lab 测试程序 ===\n")
    
    # 测试数据文件路径
    data_files = [
        "data/Amazon.txt",
        "data/CondMat.txt", 
        "data/Gowalla.txt"
    ]
    
    # 输出目录
    output_dir = "output"
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"跳过文件 {data_file}（文件不存在）")
            continue
            
        print(f"正在处理文件: {data_file}")
        
        try:
            # 读入文件
            g = Graph(data_file)
            
            # 获取并显示图的统计信息
            stats = g.get_stats()
            print(f"  节点数: {stats['nodes']}")
            print(f"  边数: {stats['edges']}")
            print(f"  是否连通: {stats['is_connected']}")
            print(f"  密度: {stats['density']:.6f}")
            print(f"  平均度: {stats['average_degree']:.2f}")
            print(f"  连通分量数: {stats['number_of_components']}")
            print(f"  最大连通分量大小: {stats['largest_component_size']}")
            print(f"  最小度: {stats['min_degree']}")
            print(f"  最大度: {stats['max_degree']}")
            print(f"  度的均值: {stats['mean_degree']:.2f}")
            print(f"  度的标准差: {stats['std_degree']:.2f}")
            print(f"  度的中位数: {stats['median_degree']:.2f}")
            
            # 保存图到输出目录
            base_name = os.path.splitext(os.path.basename(data_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_output.txt")
            g.save(output_path)
            
            print(f"  已保存到: {output_path}\n")
            
        except Exception as e:
            print(f"  处理 {data_file} 时出错: {str(e)}\n")
    
    # 额外测试：创建一个简单的图并保存
    print("=== 创建测试图 ===")
    test_graph = Graph()
    
    # 手动添加一些边来创建测试图
    nx_graph = test_graph.get_networkx_graph()
    nx_graph.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    
    print("创建了一个简单的测试图")
    stats = test_graph.get_stats()
    print(f"节点数: {stats['nodes']}")
    print(f"边数: {stats['edges']}")
    
    # 保存测试图
    test_output = os.path.join(output_dir, "test_graph.txt")
    test_graph.save(test_output)
    print(f"测试图已保存到: {test_output}")
    
    # 重新读取测试图验证
    print("\n=== 验证保存的测试图 ===")
    reloaded_graph = Graph(test_output)
    reloaded_stats = reloaded_graph.get_stats()
    print(f"重新加载的图 - 节点数: {reloaded_stats['nodes']}, 边数: {reloaded_stats['edges']}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 