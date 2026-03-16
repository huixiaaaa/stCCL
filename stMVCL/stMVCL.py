import torch
import torch.nn as nn
from tqdm import tqdm
import scipy.sparse as sp
import torch.nn.functional as F
from .utils import fix_seed
from .data import *
from .model import Model

#################stMVCL############################
# #################消融实验7：特征视图的负样本构建时，没有剪枝############################

def train(adata,arg):
    fix_seed(seed=arg.seed)
    # 根据CUDA的可用性选择运行设备（GPU或CPU）
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    #构建基于距离k=6的空间图1 sadj1和基于radius=150的空间图2 sadj2
    if 'adj2' not in adata.uns.keys():
        spatial_construct_graph(adata, k_cutoff=arg.knn,rad_cutoff=arg.radius)
        # raise ValueError("Construct two kinds of spatial neighborhood network first!")
    #构建基于基因表达的余弦相似度k=3构建的特征图 fadj
    if 'feat' not in adata.obsm.keys():
        get_feature(adata)
    if 'fadj' not in adata.obsm.keys():
        features_construct_graph(adata)

    # if 'label_CSL' not in adata.obsm.keys():
    #     add_contrastive_label(adata)
    print('Size of Input: ', adata_Vars.shape)
    print(adata)

    features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
    features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
    # label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
    adj1 = adata.obsm['adj1']
    adj2 = adata.obsm['adj2']
    fadj = adata.obsm['fadj']
    graph_neigh1 = torch.FloatTensor(adata.obsm['graph_neigh1'].copy() + np.eye(adj1.shape[0])).to(device)
    graph_neigh2 = torch.FloatTensor(adata.obsm['graph_neigh2'].copy() + np.eye(adj2.shape[0])).to(device)
    f_graph_neigh = torch.FloatTensor(adata.obsm['f_graph_neigh'].copy() + np.eye(fadj.shape[0])).to(device)

    dim_input = features.shape[1]
    # dim_output = dim_output

    adj1 = preprocess_adj(adj1)
    adj1 = torch.FloatTensor(adj1).to(device)
    adj2 = preprocess_adj(adj2)
    adj2 = torch.FloatTensor(adj2).to(device)
    fadj = preprocess_adj(fadj)
    fadj = torch.FloatTensor(fadj).to(device)

    model = Model(dim_input, arg.latent_dim).to(device)
    # model = Model(adata.X.shape[1], arg.latent_dim).to(device)
    # model = Model(dim_input, hidden_dims[0], dim_output).to(device)

    # loss_CSL = nn.BCEWithLogitsLoss()
    #data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), arg.lr,weight_decay=arg.weight_decay)
    print('Begin to train ST data...')
    model.train()
    losses = []
    for epoch in tqdm(range(arg.epoch)):
        model.train()

        features_a = permutation(features)
        # hiden_feat, emb = model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
        # hiden_feat, emb,loss_ctr = model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
        hiden_feat, emb,loss_ctr1,loss_ctr3= model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
        # hiden_feat, emb,ret1,ret2,ret3= model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
        # loss_sl_1 = loss_CSL(ret1,label_CSL)
        # loss_sl_2 = loss_CSL(ret2,label_CSL)
        # loss_sl_3 = loss_CSL(ret3,label_CSL)
        loss_ctr = loss_ctr1 + loss_ctr3
        loss_feat = F.mse_loss(features, emb)
        loss = loss_feat+0.2*loss_ctr
        # loss = loss_feat + 0.2 * loss_ctr1+5*loss_ctr3
        # loss = loss_feat
        # loss = loss_ctr
        # loss = 10 * loss_feat + 1 * (loss_sl_1 + loss_sl_2)
        # loss = 10 * loss_feat + 1 * (loss_sl_1 + loss_sl_2 + loss_sl_3)
        losses.append(loss.item())

        if epoch % (arg.epoch / 10) == 0:
            # print(f'EP[%4d]: rec_loss=%.4f.' % (epoch, loss.data))
            print(' epoch: ', epoch, ' feat_loss = {:.2f}'.format(loss_feat),
                  ' loss_sl_1 = {:.2f}'.format(loss_ctr1), ' loss_sl_2 = {:.2f}'.format(loss_ctr3),
                  # ' loss_cl = {:.2f}'.format(loss_ctr),
                  ' total_loss = {:.2f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    import matplotlib.pyplot as plt
    x = range(1, len(losses) + 1)
    plt.plot(x, losses)
    plt.show()

    with torch.no_grad():
        model.eval()
        # latent_rep,emb_rec,_ = model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)[1].detach().cpu().numpy()
        # adata.obsm["stMVCL"] = emb_rec
        # latent_rep, reconst_rep, _ = model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2,
        #                                       f_graph_neigh)
        latent_rep, reconst_rep, _ ,_= model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
        latent = latent_rep.to('cpu').detach().numpy()
        reconst_rep = reconst_rep.to('cpu').detach().numpy()
        adata.obsm['latent'] = latent
        adata.obsm['stMVCL'] = reconst_rep
        # if save_loss:
        #     adata.uns['MVCL_loss'] = loss
        # if save_reconstrction:
        #     # ReX = emb_rec.to('cpu').detach().numpy()
        #     ReX = reconst_rep
        #     ReX[ReX<0] = 0
        #     adata.layers['stMVCL_ReX'] = ReX
        return adata

# #################消融实验1：仅使用空间视图，无对比学习框架############################
# ################消融实验2：仅使用特征视图，无对比学习框架############################
# ################消融实验3：空间视图和特征视图进行简单加权平均融合，无对比学习框架############################
# ################消融实验4：对空间视图和特征视图进行多头注意力机制动态融合，无对比学习框架############################
#
# def train(adata,arg):
#     fix_seed(seed=arg.seed)
#     # 根据CUDA的可用性选择运行设备（GPU或CPU）
#     device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
#
#     adata.X = sp.csr_matrix(adata.X)
#
#     if 'highly_variable' in adata.var.columns:
#         adata_Vars =  adata[:, adata.var['highly_variable']]
#     else:
#         adata_Vars = adata
#     #构建基于距离k=6的空间图1 sadj1和基于radius=150的空间图2 sadj2
#     if 'adj2' not in adata.uns.keys():
#         spatial_construct_graph(adata, k_cutoff=arg.knn,rad_cutoff=arg.radius)
#         # raise ValueError("Construct two kinds of spatial neighborhood network first!")
#     #构建基于基因表达的余弦相似度k=3构建的特征图 fadj
#     if 'feat' not in adata.obsm.keys():
#         get_feature(adata)
#     if 'fadj' not in adata.obsm.keys():
#         features_construct_graph(adata)
#
#     # if 'label_CSL' not in adata.obsm.keys():
#     #     add_contrastive_label(adata)
#     print('Size of Input: ', adata_Vars.shape)
#     print(adata)
#
#     features = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)
#     features_a = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(device)
#     # label_CSL = torch.FloatTensor(adata.obsm['label_CSL']).to(device)
#     adj1 = adata.obsm['adj1']
#     adj2 = adata.obsm['adj2']
#     fadj = adata.obsm['fadj']
#     graph_neigh1 = torch.FloatTensor(adata.obsm['graph_neigh1'].copy() + np.eye(adj1.shape[0])).to(device)
#     graph_neigh2 = torch.FloatTensor(adata.obsm['graph_neigh2'].copy() + np.eye(adj2.shape[0])).to(device)
#     f_graph_neigh = torch.FloatTensor(adata.obsm['f_graph_neigh'].copy() + np.eye(fadj.shape[0])).to(device)
#
#     dim_input = features.shape[1]
#     # dim_output = dim_output
#
#     adj1 = preprocess_adj(adj1)
#     adj1 = torch.FloatTensor(adj1).to(device)
#     adj2 = preprocess_adj(adj2)
#     adj2 = torch.FloatTensor(adj2).to(device)
#     fadj = preprocess_adj(fadj)
#     fadj = torch.FloatTensor(fadj).to(device)
#
#     model = Model(dim_input, arg.latent_dim).to(device)
#
#
#     optimizer = torch.optim.Adam(model.parameters(), arg.lr,weight_decay=arg.weight_decay)
#     print('Begin to train ST data...')
#     model.train()
#     losses = []
#     for epoch in tqdm(range(arg.epoch)):
#         model.train()
#
#         features_a = permutation(features)
#         hiden_feat, emb = model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
#         loss_feat = F.mse_loss(features, emb)
#
#         loss = loss_feat
#         losses.append(loss.item())
#
#         if epoch % (arg.epoch / 10) == 0:
#             # print(f'EP[%4d]: rec_loss=%.4f.' % (epoch, loss.data))
#             print(' epoch: ', epoch, ' feat_loss = {:.2f}'.format(loss_feat),
#                   ' total_loss = {:.2f}'.format(loss))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     import matplotlib.pyplot as plt
#     x = range(1, len(losses) + 1)
#     plt.plot(x, losses)
#     plt.show()
#
#     with torch.no_grad():
#         model.eval()
#         latent_rep, reconst_rep= model(features, features_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh)
#         latent = latent_rep.to('cpu').detach().numpy()
#         reconst_rep = reconst_rep.to('cpu').detach().numpy()
#         adata.obsm['latent'] = latent
#         adata.obsm['stMVCL'] = reconst_rep
#
#         return adata
