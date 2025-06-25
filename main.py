"""
Main script to test the Graph class functionality.
"""

from graph import Graph
import os
import json


def save_results_to_file(file_path, data, format='txt'):
    """
    将结果保存到文件中
    
    Args:
        file_path: 输出文件路径
        data: 要保存的数据
        format: 输出格式 ('txt', 'json')
    """
    # 创建目录（如果不存在）
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def convert_numpy_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {str(k) if hasattr(k, 'item') else k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy类型
            return obj.item()
        else:
            return obj
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            if format == 'json':
                # 转换numpy类型后再序列化
                converted_data = convert_numpy_types(data)
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            else:
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"{key}: {value}\n")
                elif isinstance(data, list):
                    for item in data:
                        f.write(f"{item}\n")
                else:
                    f.write(str(data))
        print(f"结果已保存到: {file_path}")
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {str(e)}")


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
            
            # 为每个数据文件创建专门的输出文件夹
            base_name = os.path.splitext(os.path.basename(data_file))[0]
            graph_output_dir = os.path.join(output_dir, base_name)
            
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
            
            # 保存基础统计信息到文件
            save_results_to_file(
                os.path.join(graph_output_dir, "basic_stats.json"),
                stats,
                format='json'
            )
            
            # 测试k-core分解功能
            print("  === K-Core 分解测试 ===")
            
            # 获取主core
            main_core = g.get_main_core()
            main_core_stats = main_core.get_stats()
            print(f"  主core节点数: {main_core_stats['nodes']}")
            print(f"  主core边数: {main_core_stats['edges']}")
            
            # 保存主core统计信息
            save_results_to_file(
                os.path.join(graph_output_dir, "main_core_stats.json"),
                main_core_stats,
                format='json'
            )
            
            # 保存主core图
            main_core.save(os.path.join(graph_output_dir, "main_core.txt"))
            
            # 测试不同k值的k-core
            max_k = stats['max_core_number']
            k_core_results = {}
            if max_k > 0:
                test_k = min(3, max_k)  # 测试k=3或最大k值（如果小于3）
                k_core = g.get_k_core(test_k)
                k_core_stats = k_core.get_stats()
                print(f"  {test_k}-core节点数: {k_core_stats['nodes']}")
                print(f"  {test_k}-core边数: {k_core_stats['edges']}")
                
                # 保存k-core统计信息
                k_core_results[f"{test_k}_core"] = k_core_stats
                save_results_to_file(
                    os.path.join(graph_output_dir, f"{test_k}_core_stats.json"),
                    k_core_stats,
                    format='json'
                )
                
                # 保存k-core图
                k_core.save(os.path.join(graph_output_dir, f"{test_k}_core.txt"))
            
            # 测试k-clique分解功能
            print("  === K-Clique 分解测试 ===")
            
            # 测试不同k值的k-clique
            max_clique_size = stats['largest_clique_size']
            clique_results = {}
            
            if max_clique_size > 0:
                # 测试k=3的团（如果存在）
                if max_clique_size >= 3:
                    k3_cliques = g.get_k_cliques(3)
                    print(f"  大小为3的极大团数量: {len(k3_cliques)}")
                    if len(k3_cliques) > 0:
                        print(f"  前5个3-团: {k3_cliques[:5]}")
                    
                    # 保存3-团结果
                    clique_results["3_cliques"] = {
                        "count": len(k3_cliques),
                        "cliques": k3_cliques[:100]  # 只保存前100个避免文件过大
                    }
                
                # 测试最大团大小的团
                max_k_cliques = g.get_k_cliques(max_clique_size)
                print(f"  最大团(大小为{max_clique_size})数量: {len(max_k_cliques)}")
                if len(max_k_cliques) > 0:
                    print(f"  最大团示例: {max_k_cliques[0]}")
                
                # 保存最大团结果
                clique_results["max_cliques"] = {
                    "size": max_clique_size,
                    "count": len(max_k_cliques),
                    "cliques": max_k_cliques[:50]  # 只保存前50个避免文件过大
                }
                
                # 测试包含所有团的情况（对小图）
                if stats['nodes'] <= 20:  # 只对小图运行
                    k2_all_cliques = g.get_all_cliques_of_size_k(2)
                    print(f"  大小为2的所有团数量: {len(k2_all_cliques)}")
                    
                    # 保存所有2-团结果
                    clique_results["2_all_cliques"] = {
                        "count": len(k2_all_cliques),
                        "cliques": k2_all_cliques
                    }
                
            else:
                print("  图中没有团")
                clique_results["no_cliques"] = True
            
            # 保存团分解结果
            save_results_to_file(
                os.path.join(graph_output_dir, "clique_results.json"),
                clique_results,
                format='json'
            )
            
            # 测试最密子图算法
            print("  === 最密子图算法测试 ===")
            
            densest_results = {}
            
            # 测试2-近似算法
            try:
                approx_densest = g.get_densest_subgraph_approx()
                approx_stats = approx_densest.get_stats()
                approx_density = approx_stats['density']
                print(f"  2-近似算法结果:")
                print(f"    节点数: {approx_stats['nodes']}")
                print(f"    边数: {approx_stats['edges']}")
                print(f"    密度: {approx_density:.6f}")
                
                # 保存近似算法结果
                densest_results["approx_algorithm"] = approx_stats
                
                # 保存近似最密子图
                approx_densest.save(os.path.join(graph_output_dir, "densest_subgraph_approx.txt"))
                
            except Exception as e:
                print(f"  2-近似算法出错: {str(e)}")
                densest_results["approx_algorithm_error"] = str(e)
            
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
                    
                    # 保存精确算法结果
                    densest_results["exact_algorithm"] = exact_stats
                    
                    # 保存精确最密子图
                    exact_densest.save(os.path.join(graph_output_dir, "densest_subgraph_exact.txt"))
                    
                    # 计算近似比
                    if exact_density > 0 and 'approx_algorithm' in densest_results:
                        approx_ratio = approx_density / exact_density
                        print(f"    近似比: {approx_ratio:.4f}")
                        densest_results["approximation_ratio"] = approx_ratio
                        
                except Exception as e:
                    print(f"  精确算法出错: {str(e)}")
                    densest_results["exact_algorithm_error"] = str(e)
            else:
                print("  图太大，跳过精确算法测试")
                densest_results["exact_algorithm_skipped"] = "Graph too large"
            
            # 保存最密子图算法结果
            save_results_to_file(
                os.path.join(graph_output_dir, "densest_subgraph_results.json"),
                densest_results,
                format='json'
            )
            
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
                
                lds_results = {"k_value": k_value, "lds_list": []}
                
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
                    
                    # 保存每个LDS的详细信息和图
                    for idx, (lds_graph, density) in enumerate(top_k_lds):
                        lds_stats = lds_graph.get_stats()
                        lds_info = {
                            "rank": idx + 1,
                            "density": density,
                            "stats": lds_stats,
                            "nodes": list(lds_graph.get_networkx_graph().nodes())
                        }
                        lds_results["lds_list"].append(lds_info)
                        
                        # 保存LDS图
                        lds_graph.save(os.path.join(graph_output_dir, f"lds_{idx+1}.txt"))
                    
                    # 显示最佳LDS的详细信息
                    best_lds_graph, best_density = top_k_lds[0]
                    best_lds_stats = best_lds_graph.get_stats()
                    print(f"  最佳LDS详细信息:")
                    print(f"    连通性: {best_lds_stats['is_connected']}")
                    print(f"    平均度: {best_lds_stats['average_degree']:.2f}")
                    
                    lds_results["best_lds"] = {
                        "density": best_density,
                        "stats": best_lds_stats,
                        "nodes": list(best_lds_graph.get_networkx_graph().nodes())
                    }
                    
                else:
                    print(f"    未找到有效的LDS")
                    lds_results["no_lds_found"] = True
                
                # 保存LDS算法结果
                save_results_to_file(
                    os.path.join(graph_output_dir, "lds_results.json"),
                    lds_results,
                    format='json'
                )
                    
            except Exception as e:
                print(f"  LDS算法出错: {str(e)}")
                # 保存错误信息
                save_results_to_file(
                    os.path.join(graph_output_dir, "lds_error.txt"),
                    f"LDS算法错误: {str(e)}",
                    format='txt'
                )
            
            # 保存原始图到专门的输出目录
            g.save(os.path.join(graph_output_dir, "original_graph.txt"))
            
            print(f"  所有结果已保存到: {graph_output_dir}\n")
            
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
    
    # 为测试图创建专门的输出文件夹
    test_graph_output_dir = os.path.join(output_dir, "test_graph")
    
    # 保存测试图的基础统计信息
    save_results_to_file(
        os.path.join(test_graph_output_dir, "basic_stats.json"),
        stats,
        format='json'
    )
    
    # 测试k-clique分解功能
    print("=== 测试图K-Clique分解测试 ===")
    
    # 获取团统计信息
    clique_stats = test_graph.get_clique_statistics()
    print(f"极大团总数: {clique_stats['total_maximal_cliques']}")
    print(f"最大团大小: {clique_stats['largest_clique_size']}")
    print(f"团数: {clique_stats['clique_number']}")
    print(f"团大小分布: {clique_stats['clique_size_distribution']}")
    
    test_clique_results = {"clique_stats": clique_stats}
    
    # 测试具体的k-clique
    if clique_stats['largest_clique_size'] >= 3:
        k3_cliques = test_graph.get_k_cliques(3)
        print(f"大小为3的极大团: {k3_cliques}")
        
        # 测试包含所有团
        k3_all_cliques = test_graph.get_all_cliques_of_size_k(3)
        print(f"大小为3的所有团: {k3_all_cliques}")
        
        test_clique_results["3_cliques"] = {
            "maximal_cliques": k3_cliques,
            "all_cliques": k3_all_cliques
        }
    
    # 测试包含特定节点的团
    containing_cliques = test_graph.find_cliques_containing_nodes([1, 2])
    print(f"包含节点[1,2]的极大团: {containing_cliques}")
    
    test_clique_results["containing_nodes_1_2"] = containing_cliques
    
    # 保存测试图团分解结果
    save_results_to_file(
        os.path.join(test_graph_output_dir, "clique_results.json"),
        test_clique_results,
        format='json'
    )
    
    # 测试最密子图算法
    print("=== 测试图最密子图算法测试 ===")
    
    test_densest_results = {}
    
    # 测试2-近似算法
    try:
        approx_densest = test_graph.get_densest_subgraph_approx()
        approx_stats = approx_densest.get_stats()
        approx_density = approx_stats['density']
        print(f"2-近似算法结果:")
        print(f"  节点数: {approx_stats['nodes']}")
        print(f"  边数: {approx_stats['edges']}")
        print(f"  密度: {approx_density:.6f}")
        
        test_densest_results["approx_algorithm"] = approx_stats
        approx_densest.save(os.path.join(test_graph_output_dir, "densest_subgraph_approx.txt"))
        
    except Exception as e:
        print(f"2-近似算法出错: {str(e)}")
        test_densest_results["approx_algorithm_error"] = str(e)
    
    # 测试精确算法
    try:
        exact_densest = test_graph.get_densest_subgraph_exact()
        exact_stats = exact_densest.get_stats()
        exact_density = exact_stats['density']
        print(f"精确算法结果:")
        print(f"  节点数: {exact_stats['nodes']}")
        print(f"  边数: {exact_stats['edges']}")
        print(f"  密度: {exact_density:.6f}")
        
        test_densest_results["exact_algorithm"] = exact_stats
        exact_densest.save(os.path.join(test_graph_output_dir, "densest_subgraph_exact.txt"))
        
        # 计算近似比
        if exact_density > 0 and 'approx_algorithm' in test_densest_results:
            approx_ratio = approx_density / exact_density
            print(f"  近似比: {approx_ratio:.4f}")
            test_densest_results["approximation_ratio"] = approx_ratio
            
    except Exception as e:
        print(f"精确算法出错: {str(e)}")
        test_densest_results["exact_algorithm_error"] = str(e)
    
    # 保存测试图最密子图算法结果
    save_results_to_file(
        os.path.join(test_graph_output_dir, "densest_subgraph_results.json"),
        test_densest_results,
        format='json'
    )
    
    # 测试LDS算法
    print("=== 测试图LDS算法测试 ===")
    
    try:
        # 测试top-5 LDS
        k_value = 5
        top_k_lds = test_graph.get_top_k_lds(k_value)
        print(f"Top-{k_value} LDS结果:")
        
        test_lds_results = {"k_value": k_value, "lds_list": []}
        
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
            
            # 保存每个LDS的详细信息和图
            for idx, (lds_graph, density) in enumerate(top_k_lds):
                lds_stats = lds_graph.get_stats()
                lds_info = {
                    "rank": idx + 1,
                    "density": density,
                    "stats": lds_stats,
                    "nodes": list(lds_graph.get_networkx_graph().nodes())
                }
                test_lds_results["lds_list"].append(lds_info)
                
                # 保存LDS图
                lds_graph.save(os.path.join(test_graph_output_dir, f"lds_{idx+1}.txt"))
            
            # 显示最佳LDS的详细信息
            best_lds_graph, best_density = top_k_lds[0]
            best_lds_stats = best_lds_graph.get_stats()
            print(f"最佳LDS详细信息:")
            print(f"  连通性: {best_lds_stats['is_connected']}")
            print(f"  平均度: {best_lds_stats['average_degree']:.2f}")
            print(f"  组成节点: {list(best_lds_graph.get_networkx_graph().nodes())}")
            
            test_lds_results["best_lds"] = {
                "density": best_density,
                "stats": best_lds_stats,
                "nodes": list(best_lds_graph.get_networkx_graph().nodes())
            }
            
        else:
            print("  未找到有效的LDS")
            test_lds_results["no_lds_found"] = True
        
        # 保存测试图LDS算法结果
        save_results_to_file(
            os.path.join(test_graph_output_dir, "lds_results.json"),
            test_lds_results,
            format='json'
        )
            
    except Exception as e:
        print(f"LDS算法出错: {str(e)}")
        # 保存错误信息
        save_results_to_file(
            os.path.join(test_graph_output_dir, "lds_error.txt"),
            f"LDS算法错误: {str(e)}",
            format='txt'
        )
    
    # 保存测试图
    test_graph.save(os.path.join(test_graph_output_dir, "original_graph.txt"))
    print(f"测试图已保存到: {test_graph_output_dir}")
    
    # 重新读取测试图验证
    print("\n=== 验证保存的测试图 ===")
    reloaded_graph = Graph(os.path.join(test_graph_output_dir, "original_graph.txt"))
    reloaded_stats = reloaded_graph.get_stats()
    print(f"重新加载的图 - 节点数: {reloaded_stats['nodes']}, 边数: {reloaded_stats['edges']}")
    
    # 保存重新加载图的验证结果
    verification_result = {
        "original_stats": stats,
        "reloaded_stats": reloaded_stats,
        "verification_success": stats['nodes'] == reloaded_stats['nodes'] and stats['edges'] == reloaded_stats['edges']
    }
    save_results_to_file(
        os.path.join(test_graph_output_dir, "verification_result.json"),
        verification_result,
        format='json'
    )
    
    print("\n=== 图的可视化测试 ===")
    
    # 创建可视化子文件夹
    visualization_dir = os.path.join(test_graph_output_dir, "visualizations")
    
    # 测试基础可视化
    print("Testing basic visualization...")
    test_graph.visualize(
        title="Test Graph - Basic Visualization",
        save_path=os.path.join(visualization_dir, "basic.png"),
        show=False  # 设置为False避免在无显示环境中出错
    )
    
    # 测试带标签的可视化
    print("Testing visualization with labels...")
    test_graph.visualize_with_node_labels(
        title="Test Graph - Visualization with Labels",
        save_path=os.path.join(visualization_dir, "with_labels.png"),
        show=False
    )
    
    # 测试基于度数的可视化
    print("Testing degree-based visualization...")
    test_graph.visualize_by_degree(
        title="Test Graph - Degree-based Visualization",
        save_path=os.path.join(visualization_dir, "by_degree.png"),
        show=False
    )
    
    # 测试基于core number的可视化
    print("Testing core number-based visualization...")
    test_graph.visualize_by_core_number(
        title="Test Graph - Core Number-based Visualization",
        save_path=os.path.join(visualization_dir, "by_core_number.png"),
        show=False
    )
    
    # 测试子图可视化（高亮特定节点）
    print("Testing subgraph visualization...")
    highlight_nodes = [1, 2, 3]  # 高亮一些节点
    test_graph.visualize_subgraph(
        nodes=highlight_nodes,
        title="Test Graph - Subgraph Visualization",
        save_path=os.path.join(visualization_dir, "subgraph_highlight.png"),
        show=False
    )
    
    # 对于小的数据集，也进行可视化测试
    print("\n=== 数据集可视化测试 ===")
    
    # 使用Amazon数据集进行可视化测试（如果文件存在且不太大）
    amazon_file = "data/Amazon.txt"
    if os.path.exists(amazon_file):
        print(f"测试 {amazon_file} 的可视化...")
        try:
            amazon_graph = Graph(amazon_file)
            amazon_stats = amazon_graph.get_stats()
            
            # 为Amazon图创建可视化文件夹
            amazon_visualization_dir = os.path.join(output_dir, "Amazon", "visualizations")
            
            # 只对较小的图进行可视化
            if amazon_stats['nodes'] <= 500:
                print(f"Amazon graph has {amazon_stats['nodes']} nodes, performing visualization...")
                
                # 基础可视化
                amazon_graph.visualize(
                    title=f"Amazon Graph - Basic Visualization ({amazon_stats['nodes']} nodes)",
                    save_path=os.path.join(amazon_visualization_dir, "basic.png"),
                    show=False,
                    node_size=10  # 较小的节点避免重叠
                )
                
                # 基于度数的可视化
                amazon_graph.visualize_by_degree(
                    title=f"Amazon Graph - Degree-based Visualization ({amazon_stats['nodes']} nodes)",
                    save_path=os.path.join(amazon_visualization_dir, "by_degree.png"),
                    show=False
                )
                
                # 基于core number的可视化
                amazon_graph.visualize_by_core_number(
                    title=f"Amazon Graph - Core Number-based Visualization ({amazon_stats['nodes']} nodes)",
                    save_path=os.path.join(amazon_visualization_dir, "by_core_number.png"),
                    show=False
                )
                
                # 可视化main core
                main_core = amazon_graph.get_main_core()
                main_core_nodes = list(main_core.get_networkx_graph().nodes())
                if main_core_nodes:
                    amazon_graph.visualize_subgraph(
                        nodes=main_core_nodes,
                        title=f"Amazon Graph - Main Core Visualization ({len(main_core_nodes)} nodes)",
                        save_path=os.path.join(amazon_visualization_dir, "main_core.png"),
                        show=False,
                        highlight_color='red',
                        node_size=15
                    )
                
                # 保存可视化结果摘要
                visualization_summary = {
                    "graph_name": "Amazon",
                    "total_nodes": amazon_stats['nodes'],
                    "visualizations_created": [
                        "basic.png",
                        "by_degree.png",
                        "by_core_number.png",
                        "main_core.png" if main_core_nodes else None
                    ],
                    "main_core_size": len(main_core_nodes) if main_core_nodes else 0
                }
                save_results_to_file(
                    os.path.join(amazon_visualization_dir, "visualization_summary.json"),
                    visualization_summary,
                    format='json'
                )
                
            else:
                print(f"Amazon graph has too many nodes ({amazon_stats['nodes']}), skipping visualization")
                # 保存跳过可视化的记录
                skip_record = {
                    "graph_name": "Amazon",
                    "total_nodes": amazon_stats['nodes'],
                    "visualization_skipped": True,
                    "reason": "Too many nodes"
                }
                save_results_to_file(
                    os.path.join(amazon_visualization_dir, "visualization_skipped.json"),
                    skip_record,
                    format='json'
                )
                
        except Exception as e:
            print(f"Amazon graph visualization test error: {str(e)}")
            # 保存错误记录
            error_record = {
                "graph_name": "Amazon",
                "visualization_error": str(e)
            }
            save_results_to_file(
                os.path.join(output_dir, "Amazon", "visualization_error.json"),
                error_record,
                format='json'
            )
    
    print("\nVisualization tests completed!")
    print("All visualization images have been saved to the output directory")
    
    print("\n测试完成！")


if __name__ == "__main__":
    main() 