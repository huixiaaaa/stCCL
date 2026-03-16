import ot
import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn



def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='stMVCL', random_seed=2023, refine=False,n_refine=25):

   # #KPCA
   #  from sklearn.decomposition import KernelPCA
   #  kpca = KernelPCA(n_components=20)
   #  embedding = kpca.fit_transform(adata.obsm[used_obsm].copy())
    #PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=20, random_state=42)
    embedding = pca.fit_transform(adata.obsm[used_obsm].copy())
    adata.obsm['emb_pca'] = embedding

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    if refine:
        new_type = refine_label(adata, n_refine, key='mclust')
        adata.obs['mclust'] = new_type

    return adata

def refine_label(adata, radius=0, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    return new_type

