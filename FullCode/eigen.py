import numpy as np

def getEigvec(HT):
    H, Dv, De, W = getHypergraph(HT)
    eigval, eigvec = eig(H, Dv, De, W)

    return eigval, eigvec

def getHypergraph(HT):
    """
    This function aims to convert a graph to a hypergraph.
    
    graph: 
        HT: a hypergraph incidence matrix, shape is [num_hyperedges, num_hypernodes], contains 0 and 1.
    
    return Hypergraph:
        H: a hypergraph incidence matrix, shape is [num_hypernodes, num_hyperedges], contains 0 and 1.
        Dv: node degree matrix --> [num_nodes, num_nodes]
        De: hyperedge degree matrix --> [hyperedge_num, hyperedge_num]
        W: hyperedge weight matrix --> [hyperedge_num, hyperedge_num]
        
    """
    H = HT.T

    De = np.diag(np.sum(H, axis = 0)) # hyperedge degree matrix --> [num_hyperedges, num_hyperedge]
    Dv = np.diag(np.sum(H, axis = 1)) # nodes degree matrix  --> [num_nodes, num_nodes]
    W = De    # hyperedge weight matrix --> [hyperedge_num, hyperedge_num]
    
    return H, Dv, De, W

def eig(H, Dv, De, W):
    """
    This function aims to get hypergraph eigendecomposition.
    Reference: 
        https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
        https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/data/algos.py
    
    H: a hypergraph incidence matrix, shape is [num_hypernodes, num_hyperedges], contains 0 and 1.
    Dv: node degree matrix --> [num_nodes, num_nodes]
    De: hyperedge degree matrix --> [hyperedge_num, hyperedge_num]
    W: hyperedge weight matrix --> [hyperedge_num, hyperedge_num]
    
    return:
        EigVal: eigenvalue --> [num_nodes, 1]
        EigVec: eigenvector --> [num_nodes, num_nodes]
    """
    
    Dv_half = np.diag(Dv.sum(axis = 0).clip(1) ** -0.5)
    De_1 = np.diag(De.sum(axis = 1).clip(1) ** -1)
    num_nodes = len(Dv)
    
    delta = np.eye(num_nodes) - Dv_half @ H @  De_1 @ H.T @ Dv_half
    
    EigVal, EigVec = np.linalg.eigh(delta)
    # EigVal = np.sort(np.abs(np.real(EigVal)))
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])


    return EigVal, EigVec