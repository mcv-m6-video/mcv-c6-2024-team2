import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A

def get_spatial_graph_original(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def get_graph(num_node, edges):

    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A # 3, 25, 25

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A

def get_groups(dataset='CoCo', CoM=0):
    groups  =[]

    if CoM == 0:  # Default is the nose
        groups.append([0])
        groups.append([1, 2, 3, 4])  # Eyes and ears
        groups.append([5, 6])  # Shoulders
        groups.append([7, 8, 9, 10])  # Arms
        groups.append([11, 12])  # Hips
        groups.append([13, 14, 15, 16])  # Legs

    elif CoM == 5:  # Center of mass at the left shoulder
        groups.append([5])
        groups.append([0, 1, 2, 3, 4])  # Head
        groups.append([6, 7, 9])  # Right arm
        groups.append([11, 13, 15])  # Left side lower body
        groups.append([12, 14, 16])  # Right side lower body

    elif CoM == 6:  # Center of mass at the right shoulder
        groups.append([6])
        groups.append([0, 1, 2, 3, 4])  # Head
        groups.append([5, 8, 10])  # Left arm
        groups.append([11, 13, 15])  # Left side lower body
        groups.append([12, 14, 16])  # Right side lower body

    else:
        raise ValueError("Invalid Center of Mass value")

    return groups
def get_edgeset(dataset='CoCo', CoM=0):
    groups = get_groups(dataset=dataset, CoM=CoM)
    
    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        
        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])

    return edges