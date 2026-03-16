import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model // self.num_heads) ** 0.5
        attention_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.output_linear(output)

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.
    """

    def __init__(self, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys, temperature):
        return info_nce(query, positive_key, negative_keys,
                        temperature=temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=1., reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        # print(logits)
        # mu_p = torch.mean(positive_logit)
        # var_p = torch.var(positive_logit)
        # # print(mu_p, var_p)
        # mu_all = torch.mean(logits)
        # var_all = torch.var(logits)
        # # print(mu_all, var_all)
        # print((mu_p-mu_all)/mu_p)
        # temperature = (var_p.pow(2)-var_all.pow(2))/(-(mu_p-mu_all)+torch.sqrt((mu_p-mu_all).pow(2)+2*(var_p.pow(2)-var_all.pow(2))*np.log(len(positive_logit)*(len(positive_logit)+1)/(2*len(positive_logit)))))
        # print(temperature)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)
    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

# #######################MM方式，共享参数，统一编码和解码#######################
# ###ARI均值：0.559353506	NMI均值：0.670038814
# class Model(nn.Module):
#     def __init__(self, in_features, out_features,  dropout=0.0, act=nn.ELU()):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.act = act
#
#         self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
#         self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
#         self.reset_parameters()
#         self.multiheadattr = MultiHeadAttention(out_features, num_heads=2)
#         self.info_nce = InfoNCE()
#
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight1)
#         torch.nn.init.xavier_uniform_(self.weight2)
#
#     def forward(self, feat, feat_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh):
#         z1 = F.dropout(feat, self.dropout, self.training)
#         z1 = torch.mm(z1, self.weight1)
#         z1 = torch.mm(adj1, z1)
#         hiden_emb1 = z1 #hiden_emb1为feat和adj1的嵌入表示
#         emb1 = self.act(z1) #锚点
#
#         # z2 = F.dropout(feat, self.dropout, self.training)
#         # z2 = torch.mm(z2, self.weight1)
#         # z2 = torch.mm(adj2, z2)
#         # hiden_emb2 = z2 #hiden_emb2为feat和adj2的嵌入表示
#         # emb2 = self.act(z2) #锚点
#
#         z3 = F.dropout(feat, self.dropout, self.training)
#         z3 = torch.mm(z3, self.weight1)
#         z3 = torch.mm(fadj, z3)
#         hiden_emb3 = z3 #hiden_emb2为feat和fadj的嵌入表示
#         emb3 = self.act(z3) #锚点
#
#         #空间图1的负例
#         z_a = F.dropout(feat_a, self.dropout, self.training)
#         z_a = torch.mm(z_a, self.weight1)
#         z_a = torch.mm(adj1, z_a)
#         emb_a = self.act(z_a)
#
#         # # #空间图2的负例
#         # z_b = F.dropout(feat_a, self.dropout, self.training) #空间图2的负例
#         # z_b = torch.mm(z_b, self.weight1)
#         # z_b = torch.mm(adj2, z_b)
#         # emb_b = self.act(z_b)
#
#         # # #特征图的负例
#         z_c = F.dropout(feat_a, self.dropout, self.training) #特征图的负例
#         z_c = torch.mm(z_c, self.weight1)
#         z_c = torch.mm(fadj, z_c)
#         emb_c = self.act(z_c)
#
#         # # #特征图的负例，扰动
#         # # 设置剪枝概率
#         # perturb_prob = 0.4
#         # # 生成与特征图形状相同的随机掩码
#         # mask = torch.rand_like(fadj) < perturb_prob
#         # # 将随机掩码应用到特征图上，实现随机剪枝
#         # mask_fadj = torch.where(mask, torch.tensor([0.], device=fadj.device), fadj)
#         # # # #特征图的负例
#         # z_c = F.dropout(feat_a, self.dropout, self.training)  # 特征图的负例
#         # z_c = torch.mm(z_c, self.weight1)
#         # # z_c = torch.mm(fadj, z_c)
#         # z_c = torch.mm(mask_fadj, z_c)
#         # emb_c = self.act(z_c)
#
#         # # 多头注意力机制 adj1+fadj
#         # q = z1.view(1, z1.shape[0], z1.shape[1])
#         # k = z3.view(1, z3.shape[0], z3.shape[1])
#         # v = z3.view(1, z3.shape[0], z3.shape[1])
#
#         # 多头注意力机制 adj1+fadj
#         q = z3.view(1, z3.shape[0], z3.shape[1])
#         k = z1.view(1, z1.shape[0], z1.shape[1])
#         v = z1.view(1, z1.shape[0], z1.shape[1])
#
#         z = self.multiheadattr(q, k, v)
#         co_hiden_emb = z.view(z.shape[1], z.shape[2])
#         # co_hiden_emb = co_hiden_emb
#         # co_hiden_emb = (z1+z2)/2
#         co_hiden_emb = co_hiden_emb + (z1 + z3) / 2
#         # # # co_hiden_emb = self.batch_norm(co_hiden_emb+z1)
#         # co_hiden_emb = (z2+z2)/2
#         co_emb = self.act(co_hiden_emb)
#
#         # co_adj = (adj1+adj2)/2
#         co_adj = adj1 + fadj
#         # co_adj = torch.clamp(co_adj, max=1)
#         h = torch.mm(co_hiden_emb, self.weight2)
#         h = torch.mm(co_adj, h)
#
#         loss_ctr1 = self.info_nce(emb1, co_emb, emb_a, temperature=0.05)
#         # loss_ctr2 = self.info_nce(emb2,co_emb, emb_b, temperature=0.05)
#         loss_ctr3 = self.info_nce(emb3, co_emb, emb_c, temperature=0.05)
#         # loss_ctr=loss_ctr1
#         # loss_ctr=loss_ctr2
#         # loss_ctr = loss_ctr1 + loss_ctr3
#         # # 计算权重，使损失越小权重越小
#         total_loss = loss_ctr1 + loss_ctr3
#         # weight1 = loss_ctr1 / total_loss
#         # weight2 = loss_ctr3 / total_loss
#         # # # # 计算权重，使损失越小权重越大
#         # # weight1 = 1- loss_ctr1 / total_loss
#         # # weight2 = 1- loss_ctr2 / total_loss
#         # # 计算加权总损失
#         # loss_ctr = weight1 * loss_ctr1 + weight2 * loss_ctr3
#         loss_ctr = loss_ctr1 + loss_ctr3
#         return co_hiden_emb, h, loss_ctr
#         # return co_hiden_emb, h


#################MM方式，空间和特征分开编码(stMVCL)############################
# #################消融实验5：对空间视图和特征视图进行多头注意力机制动态融合，在空间视图进行对比学习############################
# #################消融实验6：对空间视图和特征视图进行多头注意力机制动态融合，在特征视图进行对比学习############################
# #################消融实验7：特征视图的负样本构建时，没有剪枝############################

class Model(nn.Module):
    def __init__(self, in_features, out_features,  dropout=0.0, act=nn.ELU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        # self.weight4 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        self.multiheadattr = MultiHeadAttention(out_features, num_heads=2)
        self.info_nce = InfoNCE()


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)
        torch.nn.init.xavier_uniform_(self.weight3)
        # torch.nn.init.xavier_uniform_(self.weight4)

    def forward(self, feat, feat_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh):
        ###空间视图 radius方式
        z1 = F.dropout(feat, self.dropout, self.training)
        z1 = torch.mm(z1, self.weight1)
        z1 = torch.mm(adj1, z1)
        hiden_emb1 = z1 #hiden_emb1为feat和adj1的嵌入表示
        emb1 = self.act(z1) #锚点

        # ###空间视图 KNN方式
        # z2 = F.dropout(feat, self.dropout, self.training)
        # z2 = torch.mm(z2, self.weight1)
        # z2 = torch.mm(adj2, z2)
        # hiden_emb2 = z2  # hiden_emb2为feat和adj2的嵌入表示
        # emb2 = self.act(z2)  # 锚点

        ###特征视图
        z3 = F.dropout(feat, self.dropout, self.training)
        z3 = torch.mm(z3, self.weight3)
        z3 = torch.mm(fadj, z3)
        hiden_emb3 = z3 #hiden_emb2为feat和fadj的嵌入表示
        emb3 = self.act(z3) #锚点

        #空间图1的负例
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj1, z_a)
        emb_a = self.act(z_a)

        # # #空间图2的负例
        # z_b = F.dropout(feat_a, self.dropout, self.training)  # 空间图2的负例
        # z_b = torch.mm(z_b, self.weight1)
        # z_b = torch.mm(adj2, z_b)
        # emb_b = self.act(z_b)

        # # #空间图1的负例，扰动
        # # 设置剪枝概率
        # perturb_prob = 0.01
        # # 生成与空间图1形状相同的随机掩码
        # mask = torch.rand_like(adj1) < perturb_prob
        # # 将随机掩码应用到空间图1上，实现随机剪枝
        # mask_adj1 = torch.where(mask, torch.tensor([0.], device=adj1.device), adj1)
        # # # #空间图1的负例
        # z_a = F.dropout(feat_a, self.dropout, self.training)  # 空间图1的负例
        # z_a = torch.mm(z_a, self.weight3)
        # z_a = torch.mm(mask_adj1, z_a)
        # emb_a = self.act(z_a)

        # # # #特征图的负例
        # z_c = F.dropout(feat_a, self.dropout, self.training)  # 特征图的负例
        # z_c = torch.mm(z_c, self.weight1)
        # z_c = torch.mm(fadj, z_c)
        # emb_c = self.act(z_c)


        # #特征图的负例，扰动
        # 设置剪枝概率
        perturb_prob = 0.4
        # 生成与特征图形状相同的随机掩码
        mask = torch.rand_like(fadj) < perturb_prob
        # 将随机掩码应用到特征图上，实现随机剪枝
        mask_fadj = torch.where(mask, torch.tensor([0.], device=fadj.device), fadj)
        # # #特征图的负例
        z_c = F.dropout(feat_a, self.dropout, self.training)  # 特征图的负例
        z_c = torch.mm(z_c, self.weight1)
        z_c = torch.mm(mask_fadj, z_c)
        emb_c = self.act(z_c)

        ###空间视图1和特征视图进行融合
        # 多头注意力机制 adj1+fadj
        q = z3.view(1, z3.shape[0], z3.shape[1])
        k = z1.view(1, z1.shape[0], z1.shape[1])
        v = z1.view(1, z1.shape[0], z1.shape[1])

        # ###空间视图2和特征视图进行融合
        # # 多头注意力机制 adj1+fadj
        # q = z3.view(1, z3.shape[0], z3.shape[1])
        # k = z2.view(1, z2.shape[0], z2.shape[1])
        # v = z2.view(1, z2.shape[0], z2.shape[1])

        z = self.multiheadattr(q, k, v)
        co_hiden_emb = z.view(z.shape[1], z.shape[2])
        co_hiden_emb = co_hiden_emb + (z1 + z3) / 2
        # co_hiden_emb = co_hiden_emb + (z2 + z3) / 2
        co_emb = self.act(co_hiden_emb)



        ####双层对比学习
        loss_ctr1 = self.info_nce(emb1, co_emb, emb_a, temperature=0.05)
        # loss_ctr2 = self.info_nce(emb2, co_emb, emb_b, temperature=0.05)
        loss_ctr3 = self.info_nce(emb3, co_emb, emb_c, temperature=0.05)
        # loss_ctr = loss_ctr2 + loss_ctr3
        # loss_ctr = loss_ctr1 + loss_ctr3

        # co_adj = 0.7*adj1+ 0.3*fadj
        co_adj = adj1 + fadj

        # co_adj = adj2 + fadj
        h = torch.mm(co_hiden_emb, self.weight2)
        h = torch.mm(co_adj, h)

        # return co_hiden_emb, h, loss_ctr3
        return co_hiden_emb, h, loss_ctr1,loss_ctr3
# ################MM方式，空间和特征分开编码############################
#
# #################消融实验1：仅使用空间视图，无对比学习框架############################
# #################消融实验2：仅使用特征视图，无对比学习框架############################
# #################消融实验3：空间视图和特征视图进行简单加权平均融合，无对比学习框架############################
# #################消融实验4：对空间视图和特征视图进行多头注意力机制动态融合，无对比学习框架############################
#
# class Model(nn.Module):
#     def __init__(self, in_features, out_features,  dropout=0.0, act=nn.ELU()):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.act = act
#
#         self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
#         self.weight3 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
#         self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
#         # self.weight4 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
#         self.reset_parameters()
#         self.multiheadattr = MultiHeadAttention(out_features, num_heads=2)
#         self.info_nce = InfoNCE()
#
#
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.weight1)
#         torch.nn.init.xavier_uniform_(self.weight2)
#         torch.nn.init.xavier_uniform_(self.weight3)
#         # torch.nn.init.xavier_uniform_(self.weight4)
#
#     def forward(self, feat, feat_a, adj1, adj2, fadj, graph_neigh1, graph_neigh2, f_graph_neigh):
#         ###空间视图 radius方式
#         z1 = F.dropout(feat, self.dropout, self.training)
#         z1 = torch.mm(z1, self.weight1)
#         z1 = torch.mm(adj1, z1)
#         hiden_emb1 = z1 #hiden_emb1为feat和adj1的嵌入表示
#         emb1 = self.act(z1) #锚点
#
#         ##特征视图
#         z3 = F.dropout(feat, self.dropout, self.training)
#         z3 = torch.mm(z3, self.weight3)
#         z3 = torch.mm(fadj, z3)
#         hiden_emb3 = z3 #hiden_emb2为feat和fadj的嵌入表示
#         emb3 = self.act(z3) #锚点
#
#         ##空间视图1和特征视图进行融合
#         # 多头注意力机制 adj1+fadj
#         q = z3.view(1, z3.shape[0], z3.shape[1])
#         k = z1.view(1, z1.shape[0], z1.shape[1])
#         v = z1.view(1, z1.shape[0], z1.shape[1])
#
#         z = self.multiheadattr(q, k, v)
#         co_hiden_emb = z.view(z.shape[1], z.shape[2])
#         # co_hiden_emb = co_hiden_emb
#         co_hiden_emb = co_hiden_emb + (z1 + z3) / 2
#
#         # co_hiden_emb=z1 #仅空间视图嵌入
#         # co_hiden_emb = z3 ##仅特征视图嵌入
#         # co_hiden_emb = (z1+z3)/2 #空间视图和特征视图进行简单加权平均融合
#         co_emb = self.act(co_hiden_emb)
#
#
#         # co_adj = adj1 #仅空间图
#         # co_adj = fadj #仅特征图
#         # co_adj = 0.3*adj1+0.7*fadj #空间视图和特征视图进行简单加权平均融合
#         co_adj = 0.7*adj1+0.3*fadj #空间视图和特征视图进行简单加权平均融合
#
#         h = torch.mm(co_hiden_emb, self.weight2)
#         h = torch.mm(co_adj, h)
#
#         return co_hiden_emb, h
#
#
