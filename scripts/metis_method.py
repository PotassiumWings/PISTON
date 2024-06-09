import pymetis as metis
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm


def partition_graph_fennel(adj_matrix, num_partitions):
    print("Using fennel:")
    G = nx.from_numpy_array(adj_matrix)
    partition = nx_comm.greedy_modularity_communities(G, num_partitions)

    # 根据分区结果构建子图
    subgraphs = []
    for part in partition:
        subgraph = np.zeros_like(adj_matrix)
        for i in part:
            for j in part:
                subgraph[i][j] = adj_matrix[i][j]
        subgraphs.append(subgraph)

    return subgraphs


def partition_graph_metis(adj_matrix, num_partitions):
    print("Using metis:")
    # 转换成METIS所需的图格式
    adj_list = [[] for _ in range(len(adj_matrix))]
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                adj_list[i].append(j)

    # 使用METIS进行图分区
    (edgecuts, parts) = metis.part_graph(adjacency=adj_list, nparts=num_partitions)

    # 根据分区结果构建子图
    subgraphs = [[] for _ in range(num_partitions)]
    for i, part in enumerate(parts):
        subgraphs[part].append(i)

    # 构建子图的邻接矩阵
    subgraph_matrices = []
    for subgraph in subgraphs:
        subgraph_matrix = np.zeros_like(adj_matrix)
        for i in subgraph:
            for j in subgraph:
                subgraph_matrix[i][j] = adj_matrix[i][j]
        subgraph_matrices.append(subgraph_matrix)

    return subgraph_matrices


# 生成一个简单的邻接矩阵
# 这里只是一个示例，你可以替换成你的实际数据
adj_matrix = np.array([
    [0, 2, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])

# 指定划分成2个子图
num_partitions = 2

# 进行图分区
partition_graph = partition_graph_fennel
subgraph_matrices = partition_graph(adj_matrix, num_partitions)

# 输出结果
for i, subgraph_matrix in enumerate(subgraph_matrices):
    print(f'Subgraph {i + 1}:')
    print(subgraph_matrix)
    print()
