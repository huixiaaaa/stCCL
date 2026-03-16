import os
import copy
import torch
torch.cuda.empty_cache()  # 释放未使用的缓存
from pathlib import Path
from MVCL.MVInfoNCE.config import set_arg
from MVCL.MVInfoNCE.data import *
from MVCL.MVInfoNCE.utils import *
from MVCL.MVInfoNCE.MVInfoNCE import *
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score


os.environ['R_HOME'] = '/home/ZHX/anaconda3/envs/GraphST/lib/R'
current_path = os.getcwd()
print("当前路径：", current_path)
file_path=current_path
# gpu
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# path
data_root = Path('/home/ZHX/Dataset/stereo_seq_MouseOlfactoryBulb')
result_path = "/home/ZHX/code111/GraphST-main/results/MVInfoNCE2_result/"
#设置参数
opt = set_arg()
arg = opt.parse_args(['--n_domain', '7','--radius', '50','--n_refine','25'])
print(arg)

sample_name = 'stereo-seq_MOB'
counts_file = os.path.join(data_root, 'RNA_counts.tsv')
coor_file = os.path.join(data_root, 'position.tsv')

counts = pd.read_csv(counts_file, sep='\t', index_col=0)
coor_df = pd.read_csv(coor_file, sep='\t')
print(counts.shape, coor_df.shape)

counts.columns = ['Spot_'+str(x) for x in counts.columns]
coor_df.index = coor_df['label'].map(lambda x: 'Spot_'+str(x))
coor_df = coor_df.loc[:, ['x','y']]
coor_df.head()

adata = sc.AnnData(counts.T)
adata.X = csr_matrix(adata.X)
adata.var_names_make_unique()
print(adata)

coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
adata.obsm["spatial"] = coor_df.to_numpy()
sc.pp.calculate_qc_metrics(adata, inplace=True)
print(adata)

out_path = os.path.join(result_path , sample_name)
# 检查文件夹是否存在，如果不存在则创建文件夹
if not os.path.exists(out_path):
    os.makedirs(out_path)
plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')
plt.savefig(os.path.join(out_path,"stereo_MOB01.png"))


used_barcode = pd.read_csv(os.path.join(data_root,'used_barcodes.txt'), sep='\t', header=None)
used_barcode = used_barcode[0]
adata = adata[used_barcode,]
print(adata)

plt.rcParams["figure.figsize"] = (5,4)
sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
plt.title("")
plt.axis('off')
plt.savefig(os.path.join(out_path,"stereo_MOB02.png"))


sc.pp.filter_genes(adata, min_cells=50)
print('After flitering: ', adata.shape)

#preprocessing
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print("=============预处理后{}切片信息===============".format(sample_name))
print(adata)




# train model
adata = train(adata,arg)
print("Completion of training")

# clustering
adata = mclust_R(adata, arg.n_domain,used_obsm='MVInfoNCE' )
# adata = mclust_R(adata, arg.n_domain,used_obsm='MVInfoNCE' ,refine=True, n_refine=arg.n_refine)

# # filter out NA nodes
# adata = adata[~pd.isnull(adata.obs['Ground Truth'])]
# ARI = adjusted_rand_score(adata.obs['mclust'], adata.obs['Ground Truth'])
# print('ARI = %.4f' %ARI)
# adata.uns['ARI']=ARI

