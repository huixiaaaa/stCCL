import numpy as np
import pandas as pd
import sklearn.neighbors
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def read_data(data_root,sample_name):
    ## loading data
    adata_tmp = sc.read_visium(data_root / sample_name)
    adata_tmp.var_names_make_unique()
    adata_tmp.obs['slice_id'] = sample_name
    ## add ground truth
    # df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    # adata_tmp.obs['Ground Truth'] = df_meta['layer_guess']
    # adata_tmp.obs['Ground Truth'] = df_meta.loc[adata_tmp.obs_names, 'ground_truth']
    return adata_tmp


def spatial_construct_graph(adata, rad_cutoff=None, k_cutoff=None):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    n_spot = coor.shape[0]

    #Find the nearest neighbor based on the radius
    nbrs1 = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
    #Return two array distances An array stores the distances of each point to other points
    #indices The second array contains its index
    distances1, indices1 = nbrs1.radius_neighbors(coor, return_distance=True)
    interaction1 = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        interaction1[i,indices1[i]] = 1
    adj1 = interaction1

    # KNN_list1 = []
    # for it in range(indices1.shape[0]):
    #     KNN_list1.append(pd.DataFrame(zip([it]*indices1[it].shape[0], indices1[it], distances1[it])))

    #KNN
    nbrs2 = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1,metric='chebyshev').fit(coor)#,metric='cosine'
    distances2, indices2 = nbrs2.kneighbors(coor)

    x = indices2[:, 0].repeat(k_cutoff)
    y = indices2[:, 1:].flatten()
    interaction2 = np.zeros([n_spot, n_spot])
    interaction2[x, y] = 1
    interaction2[y, x] = 1

    adj2 = interaction2
    adj2 = adj2 + adj2.T
    adj2 = np.where(adj2>1, 1, adj2)

    # KNN_list2 = []
    # for it in range(indices2.shape[0]):
    #     KNN_list2.append(pd.DataFrame(zip([it]*indices2.shape[1],indices2[it,:], distances2[it,:])))

    adata.obsm['graph_neigh1'] = interaction1
    adata.obsm['graph_neigh2'] = interaction2
    adata.obsm['adj1'] = adj1
    adata.obsm['adj2'] = adj2

    # KNN_df1 = pd.concat(KNN_list1)
    # KNN_df2 = pd.concat(KNN_list2)
    # df = pd.concat([KNN_df1,KNN_df2],ignore_index=True)
    # df.columns = ['Cell1', 'Cell2', 'Distance']
    # KNN_df = df.drop_duplicates(subset=["Cell1", "Cell2"],ignore_index=True)


    # Spatial_Net = KNN_df.copy()
    # Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,] #It removes its own distance
    # id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index) ))#Establish correspondence between indexes and cells
    # #Map the index to the cell
    # Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    # Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    # adata.uns['Spatial_Net'] = Spatial_Net
    print("spatial graph(k) completed!")
    print("spatial graph(radius) completed!")

#基于基因表达相似性构建特征图，先降维
def features_construct_graph(adata, k=3,  mode="connectivity", metric="cosine"):
    print("start construct feature graph")
    features = adata.obsm['feat']

    from sklearn.decomposition import PCA
    # pca = PCA(n_components=64, random_state=2023)
    pca = PCA(n_components=32, random_state=2023)
    features = pca.fit_transform(features.copy())

    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    adata.obsm['f_graph_neigh'] = A
    adj2 = A
    adj2 = adj2 + adj2.T
    adj2 = np.where(adj2 > 1, 1, adj2)
    adata.obsm['fadj'] = adj2
    print("feature graph completed!")


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    #Preprocessing of adjacency matrix of simple GCN model and conversion of tuple representation
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def permutation(feature):
    #Feature random arrangement
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated

# def add_contrastive_label(adata):
#     # contrastive label
#     n_spot = adata.n_obs
#     one_matrix = np.ones([n_spot, 1])
#     zero_matrix = np.zeros([n_spot, 1])
#     label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
#     adata.obsm['label_CSL'] = label_CSL

def get_feature(adata):
    adata_Vars =  adata[:, adata.var['highly_variable']] #Only 3000 highly variable genes are selected

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ]
    # data augmentation
    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a
