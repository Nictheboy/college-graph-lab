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
    
    # 使用map函数处理数据文件，避免for循环
    def process_data_file(data_file):
        if not os.path.exists(data_file):
            print(f"跳过文件 {data_file}（文件不存在）")
            return None
            
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
            
            # 添加k-core统计信息
            print(f"  最大core数: {stats['max_core_number']}")
            print(f"  最小core数: {stats['min_core_number']}")
            print(f"  core数均值: {stats['mean_core_number']:.2f}")
            print(f"  core数标准差: {stats['std_core_number']:.2f}")
            print(f"  core数中位数: {stats['median_core_number']:.2f}")
            print(f"  core分布: {stats['core_distribution']}")
            
            # 添加团统计信息
            print(f"  极大团总数: {stats['total_maximal_cliques']}")
            print(f"  最大团大小: {stats['largest_clique_size']}")
            print(f"  团数: {stats['clique_number']}")
            print(f"  团大小分布: {stats['clique_size_distribution']}")
            
            # 添加LDS统计信息
            print(f"  LDS发现数量: {stats['total_lds_found']}")
            print(f"  LDS平均密度: {stats['average_lds_density']:.6f}")
            print(f"  LDS最大密度: {stats['max_lds_density']:.6f}")
            print(f"  LDS最小密度: {stats['min_lds_density']:.6f}")
            print(f"  LDS平均大小: {stats['average_lds_size']:.2f}")
            
            # 测试k-core分解功能
            print("  === K-Core 分解测试 ===")
            
            # 获取主core
            main_core = g.get_main_core()
            main_core_stats = main_core.get_stats()
            print(f"  主core节点数: {main_core_stats['nodes']}")
            print(f"  主core边数: {main_core_stats['edges']}")
            
            # 测试不同k值的k-core
            max_k = stats['max_core_number']
            if max_k > 0:
                test_k = min(3, max_k)  # 测试k=3或最大k值（如果小于3）
                k_core = g.get_k_core(test_k)
                k_core_stats = k_core.get_stats()
                print(f"  {test_k}-core节点数: {k_core_stats['nodes']}")
                print(f"  {test_k}-core边数: {k_core_stats['edges']}")
            
            # 测试k-clique分解功能
            print("  === K-Clique 分解测试 ===")
            
            # 测试不同k值的k-clique
            max_clique_size = stats['largest_clique_size']
            if max_clique_size > 0:
                # 测试k=3的团（如果存在）
                if max_clique_size >= 3:
                    k3_cliques = g.get_k_cliques(3)
                    print(f"  大小为3的极大团数量: {len(k3_cliques)}")
                    if len(k3_cliques) > 0:
                        print(f"  前5个3-团: {k3_cliques[:5]}")
                
                # 测试最大团大小的团
                max_k_cliques = g.get_k_cliques(max_clique_size)
                print(f"  最大团(大小为{max_clique_size})数量: {len(max_k_cliques)}")
                if len(max_k_cliques) > 0:
                    print(f"  最大团示例: {max_k_cliques[0]}")
                
                # 测试包含所有团的情况（对小图）
                if stats['nodes'] <= 20:  # 只对小图运行
                    k2_all_cliques = g.get_all_cliques_of_size_k(2)
                    print(f"  大小为2的所有团数量: {len(k2_all_cliques)}")
                    
            else:
                print("  图中没有团")
            
            # 测试最密子图算法
            print("  === 最密子图算法测试 ===")
            
            # 测试2-近似算法
            try:
                approx_densest = g.get_densest_subgraph_approx()
                approx_stats = approx_densest.get_stats()
                approx_density = approx_stats['density']
                print(f"  2-近似算法结果:")
                print(f"    节点数: {approx_stats['nodes']}")
                print(f"    边数: {approx_stats['edges']}")
                print(f"    密度: {approx_density:.6f}")
            except Exception as e:
                print(f"  2-近似算法出错: {str(e)}")
            
            # 对于小图，测试精确算法
            if stats['nodes'] <= 1000:  # 只对小图运行精确算法
                try:
                    exact_densest = g.get_densest_subgraph_exact()
                    exact_stats = exact_densest.get_stats()
                    exact_density = exact_stats['density']
                    print(f"  精确算法结果:")
                    print(f"    节点数: {exact_stats['nodes']}")
                    print(f"    边数: {exact_stats['edges']}")
                    print(f"    密度: {exact_density:.6f}")
                    
                    # 计算近似比
                    if exact_density > 0:
                        approx_ratio = approx_density / exact_density
                        print(f"    近似比: {approx_ratio:.4f}")
                except Exception as e:
                    print(f"  精确算法出错: {str(e)}")
            else:
                print("  图太大，跳过精确算法测试")
            
            # 测试LDS算法
            print("  === LDS（局部密集子图）算法测试 ===")
            
            try:
                # 根据图的大小调整k值
                if stats['nodes'] > 50000:
                    k_value = 2  # 大图使用较小的k值
                elif stats['nodes'] > 10000:
                    k_value = 3  # 中等图使用中等k值
                else:
                    k_value = 5  # 小图使用较大的k值
                
                top_k_lds = g.get_top_k_lds(k_value)
                print(f"  Top-{k_value} LDS结果:")
                
                if top_k_lds:
                    # 使用map函数显示每个LDS的信息，避免for循环
                    def display_lds_info(lds_info):
                        idx, (lds_graph, density) = lds_info
                        lds_stats = lds_graph.get_stats()
                        return f"    LDS {idx+1}: 节点数={lds_stats['nodes']}, 边数={lds_stats['edges']}, 密度={density:.6f}"
                    
                    lds_info_with_index = list(enumerate(top_k_lds))
                    lds_descriptions = list(map(display_lds_info, lds_info_with_index))
                    
                    # 显示所有LDS信息
                    list(map(print, lds_descriptions))
                    
                    # 显示最佳LDS的详细信息
                    best_lds_graph, best_density = top_k_lds[0]
                    best_lds_stats = best_lds_graph.get_stats()
                    print(f"  最佳LDS详细信息:")
                    print(f"    连通性: {best_lds_stats['is_connected']}")
                    print(f"    平均度: {best_lds_stats['average_degree']:.2f}")
                    
                else:
                    print(f"    未找到有效的LDS")
                    
            except Exception as e:
                print(f"  LDS算法出错: {str(e)}")
            
            # 保存图到输出目录
            base_name = os.path.splitext(os.path.basename(data_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_output.txt")
            g.save(output_path)
            
            print(f"  已保存到: {output_path}\n")
            
        except Exception as e:
            print(f"  处理 {data_file} 时出错: {str(e)}\n")
    
    # 使用map函数处理所有数据文件，避免for循环
    list(map(process_data_file, data_files))
    
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
    
    # 测试k-clique分解功能
    print("=== 测试图K-Clique分解测试 ===")
    
    # 获取团统计信息
    clique_stats = test_graph.get_clique_statistics()
    print(f"极大团总数: {clique_stats['total_maximal_cliques']}")
    print(f"最大团大小: {clique_stats['largest_clique_size']}")
    print(f"团数: {clique_stats['clique_number']}")
    print(f"团大小分布: {clique_stats['clique_size_distribution']}")
    
    # 测试具体的k-clique
    if clique_stats['largest_clique_size'] >= 3:
        k3_cliques = test_graph.get_k_cliques(3)
        print(f"大小为3的极大团: {k3_cliques}")
        
        # 测试包含所有团
        k3_all_cliques = test_graph.get_all_cliques_of_size_k(3)
        print(f"大小为3的所有团: {k3_all_cliques}")
    
    # 测试包含特定节点的团
    containing_cliques = test_graph.find_cliques_containing_nodes([1, 2])
    print(f"包含节点[1,2]的极大团: {containing_cliques}")
    
    # 测试最密子图算法
    print("=== 测试图最密子图算法测试 ===")
    
    # 测试2-近似算法
    try:
        approx_densest = test_graph.get_densest_subgraph_approx()
        approx_stats = approx_densest.get_stats()
        approx_density = approx_stats['density']
        print(f"2-近似算法结果:")
        print(f"  节点数: {approx_stats['nodes']}")
        print(f"  边数: {approx_stats['edges']}")
        print(f"  密度: {approx_density:.6f}")
    except Exception as e:
        print(f"2-近似算法出错: {str(e)}")
    
    # 测试精确算法
    try:
        exact_densest = test_graph.get_densest_subgraph_exact()
        exact_stats = exact_densest.get_stats()
        exact_density = exact_stats['density']
        print(f"精确算法结果:")
        print(f"  节点数: {exact_stats['nodes']}")
        print(f"  边数: {exact_stats['edges']}")
        print(f"  密度: {exact_density:.6f}")
        
        # 计算近似比
        if exact_density > 0:
            approx_ratio = approx_density / exact_density
            print(f"  近似比: {approx_ratio:.4f}")
    except Exception as e:
        print(f"精确算法出错: {str(e)}")
    
    # 测试LDS算法
    print("=== 测试图LDS算法测试 ===")
    
    try:
        # 测试top-5 LDS
        k_value = 5
        top_k_lds = test_graph.get_top_k_lds(k_value)
        print(f"Top-{k_value} LDS结果:")
        
        if top_k_lds:
            # 使用map函数显示每个LDS的信息，避免for循环
            def display_test_lds_info(lds_info):
                idx, (lds_graph, density) = lds_info
                lds_stats = lds_graph.get_stats()
                return f"  LDS {idx+1}: 节点数={lds_stats['nodes']}, 边数={lds_stats['edges']}, 密度={density:.6f}"
            
            lds_info_with_index = list(enumerate(top_k_lds))
            lds_descriptions = list(map(display_test_lds_info, lds_info_with_index))
            
            # 显示所有LDS信息
            list(map(print, lds_descriptions))
            
            # 显示最佳LDS的详细信息
            best_lds_graph, best_density = top_k_lds[0]
            best_lds_stats = best_lds_graph.get_stats()
            print(f"最佳LDS详细信息:")
            print(f"  连通性: {best_lds_stats['is_connected']}")
            print(f"  平均度: {best_lds_stats['average_degree']:.2f}")
            print(f"  组成节点: {list(best_lds_graph.get_networkx_graph().nodes())}")
            
        else:
            print("  未找到有效的LDS")
            
    except Exception as e:
        print(f"LDS算法出错: {str(e)}")
    
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