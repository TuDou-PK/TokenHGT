import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class tokenHGT(nn.Module):
    def __init__(self, args, pre_trained_weight):
        super().__init__()
        self.dp = args.dp
        self.type_id = args.type_id
        self.batch_size = args.batch_size
        self.eigen_norm = args.eigen_norm


        self.device = args.device
        
        pre_trained_weight = torch.FloatTensor(pre_trained_weight)
#         self.feature_embedding = nn.Embedding(args.max_words, args.node_dim).to(self.device)
        self.feature_embedding = nn.Embedding.from_pretrained(pre_trained_weight, freeze = False, padding_idx = 0) # [max_words, 300]

        self.eigen_encoder = nn.Linear(self.dp, args.node_dim).to(self.device)
        self.lap_eig_dropout = nn.Dropout2d(p=args.lap_node_id_eig_dropout) if args.lap_node_id_eig_dropout > 0 else None

        if self.type_id:
            self.order_encoder = nn.Embedding(2, args.node_dim).to(self.device)
        self.graph_token = nn.Embedding(1, args.node_dim*2).to(self.device)
        self.null_token = nn.Embedding(1, args.node_dim*2).to(self.device)


        self.layernorm1 = nn.LayerNorm(args.node_dim).to(self.device) if args.layernorm else None
        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else None

        # layers for transformer
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(d_model=args.node_dim*2, nhead=args.nhead, device=self.device)
        self.transformerEncoder = nn.TransformerEncoder(encoder_layer=self.transformerEncoderLayer, num_layers=args.num_layers)

        # after transformer
        self.lm_head_transform_weight = nn.Linear(args.node_dim*2, args.node_dim).to(self.device)
        self.gelu = torch.nn.GELU()
        self.layernorm2 = nn.LayerNorm(args.node_dim).to(self.device)

        # output layer
        # self.softmax = nn.Softmax(dim=1)
        # self.output_layer0 = nn.Linear(args.node_dim, args.node_dim).to(self.device)
        self.output_layer = nn.Linear(args.node_dim, args.num_classes).to(self.device)
        # self.lm_output_learned_bias = nn.Parameter(torch.ones(1)*0.001).to(self.device)

        # self.apply(lambda module: init_params(module, n_layers=2))

    def forward(self, batch_data):

        _, HTs, node_features, targets, _, hyperedges, eigvecs = batch_data


        node_features = torch.tensor(node_features).to(self.device) #: [batch_size, max_num_node]
        targets = torch.tensor(targets).long().to(self.device) #: [batch_size]
        # node_masks = torch.tensor(node_masks).to(self.device) #: [batch_size, max_num_node]
        # eigvecs = torch.tensor(eigvecs).to(self.device) #: [batch_size, random_num_node, random_num_node]


# ---------------------------------- 0. Feature token ------------------------------------- #
        num_nodes = [HTs[i].shape[1] for i in range(len(HTs))]
        num_edges = [HTs[i].shape[0] for i in range(len(HTs))]
        max_num_node = np.max(num_nodes)
        max_num_edge = np.max(num_edges)
        max_len = max_num_edge + max_num_node

        node_masks = self.get_node_mask(num_nodes, self.device)

        node_features, graph_masks, graph_indexs = self.prepareData(node_features, node_masks, max_num_node, max_num_edge, num_edges)
        feature_token = self.feature_embedding(node_features) #: [batch_size, max_len, node_dim]



# ---------------------------------- 1. Eigenvector token ---------------------------------- #
        lap_eigvecs = torch.cat([F.pad(torch.tensor(i), (0, max_num_node - i.shape[1]), value=float('0')) for i in eigvecs]) # [sum(node_num), max_num_node]

        if self.eigen_norm:
            temp_n1 = torch.linalg.norm(node_features, 'fro')
            temp_n2 = torch.linalg.norm(lap_eigvecs, 'fro')
            lap_eigvecs = lap_eigvecs * (temp_n1 / temp_n2)

        # [sum(num_nodes), dp]
        lap_dim = lap_eigvecs.shape[-1]
        if self.dp > lap_dim:
            lap_eigvecs = F.pad(lap_eigvecs, (0, self.dp - lap_dim), value=float('0'))
        else:
            lap_eigvecs = lap_eigvecs[:, :self.dp]

        if self.lap_eig_dropout is not None:
            lap_eigvecs = self.lap_eig_dropout(lap_eigvecs[..., None, None]).view(lap_eigvecs.size())

        lap_node_id = self.handle_eigvec(lap_eigvecs, node_masks, sign_flip = True) # [sum(node_num), dp], only randomly change the sign of the eigvec
        eigen_features = self.fill_tensor_with_feature(self.batch_size, max_len, num_nodes, num_edges, lap_node_id, hyperedges).to(self.device) # [batch_size, max_len, dp]
        eigen_token = self.eigen_encoder(eigen_features) # [batch_size, max_len, node_dim]



# ------------------------------------ 2. Build token --------------------------------------- #
        token = torch.cat((feature_token , eigen_token), dim=2) # [batch_size, max_len, node_dim*2]

        # token = feature_token # + eigen_token

        if self.type_id:
            token = token + self.get_type_embed(graph_indexs, num_nodes, num_edges)

        padded_feature, padding_mask = self.add_special_tokens(token, graph_masks)

        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float('0')) # 有的填充部分会有非0数字影响，去除影响噪声
        # padded_feature = token.masked_fill(graph_masks[..., None], float('0'))

# ------------------------------------ 3. tokenHGT------------------------------------------ #
#         if self.layernorm1 is not None:
#             padded_feature = self.layernorm1(padded_feature)
#
        # x = self.dropout(padded_feature)

        # B x T x C -> T x B x C
        x = padded_feature.transpose(0, 1) # [max_len, batch_size, node_dim]

        x = self.transformerEncoder(src=x, src_key_padding_mask=padding_mask)  # --> [max_len, batch, hidden_dim]
        x = x.transpose(0, 1) # [batch, max_len, hidden_dim]
        x = self.layernorm2(self.gelu(self.lm_head_transform_weight(x))) # [batch, max_len * hidden_dim]

        # Point for learnable bias, embedding project back, 以及可以采用pooling来降维


        # x = x + self.lm_output_learned_bias

        output = self.output_layer(x[:, 0, :]) # [batch, max_len, num_classes]
        return output, targets


    def get_node_mask(self, node_num, device):
        b = len(node_num)  # --> 6
        max_n = max(node_num) # --> 30
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[None, :].expand(b, max_n)  # [B, max_n] --> [6, 30]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[:, None]  # [B, 1] --> [6, 1]
        node_mask = torch.less(node_index, node_num)  # [B, max_n] --> [6, 30]
        return node_mask

    def add_special_tokens(self, padded_feature, padding_mask):
        """
        :param padded_feature: Tensor([B, T, D]) --> [6, 70, 768]
        :param padding_mask: BoolTensor([B, T])  --> [6, 70]
        :return: padded_feature: Tensor([B, 2/3 + T, D]), padding_mask: BoolTensor([B, 2/3 + T])
        """
        b, _, d = padded_feature.size() # b-->6, d-->768

        num_special_tokens = 2
        graph_token_feature = self.graph_token.weight.expand(b, 1, d)  # [1, D] 不应该是 [b, 1, d]-->[6, 1, 768]吗？
        null_token_feature = self.null_token.weight.expand(b, 1, d)  # [1, D], this is optional 这不也应该是 [6, 1, 768]吗？
        special_token_feature = torch.cat((graph_token_feature, null_token_feature), dim=1)  # [B, 2, D] --> [6, 2, 768]
        special_token_mask = torch.zeros(b, num_special_tokens, dtype=torch.bool, device=padded_feature.device) # --> [6, 2]

        padded_feature = torch.cat((special_token_feature, padded_feature), dim=1)  # [B, 2 + T, D] --> [6, 2+70, 768]
        padding_mask = torch.cat((special_token_mask, padding_mask), dim=1)  # [B, 2 + T] --> [6, 2+70]
        return padded_feature, padding_mask

    def get_type_embed(self, padded_index, num_nodes, num_edges):
        """
        :param padded_index: LongTensor([B, T]) --> [6, 70]
        :return: Tensor([B, T, D])
        """
        type_id = torch.zeros(padded_index.shape).long()
        for i in range(len(type_id)):
            type_id[i][num_nodes[i]: num_nodes[i] + num_edges[i]] = 1

        order_embed = self.order_encoder(type_id.to(self.device))
        return order_embed

    def prepareData(self, node_features, node_masks, max_num_node, max_num_edge, num_edges):
        """
        :param node_features: [batch_size, max_num_node], the end part of it is 0.
        :param node_masks: [batch_size, max_num_node], the end part where node_features is 0 is 0.
        :param max_num_node: max number of nodes in a batch
        :param max_num_edge: max number of edges in a batch
        :return:
            node_features: [batch_size, max_num_node+max_num_edge]
            node_masks: [batch_size, max_num_node+max_num_edge], the padding part is True, for transformer.
            node_indexs: [batch_size, max_num_node+max_num_edge]
        """

        graph_indexs = None
        length = max_num_node + max_num_edge
        batch_size = node_features.shape[0]
        graph_masks = torch.cat([node_masks, torch.zeros((batch_size, max_num_edge)).to(self.device)], dim=1)

        for i in range(batch_size):
            n = torch.sum(graph_masks[i]).long()
            node_index = torch.arange(0, n).unsqueeze(0).to(self.device)
            edge_index = torch.arange(0, num_edges[i]).unsqueeze(0).to(self.device)
            cat_index = torch.cat([node_index, edge_index], dim = 1)
            pad_index = F.pad(cat_index, (0, length - cat_index.shape[1]), mode='constant')
            if graph_indexs is None:
                graph_indexs = pad_index
            else:
                graph_indexs = torch.cat((graph_indexs, pad_index), dim=0)

            graph_masks[i][n: n+num_edges[i]] = 1

        node_features = torch.cat([node_features, torch.zeros((batch_size, max_num_edge)).long().to(self.device)], dim=1)

        return node_features, graph_masks<1, graph_indexs

    def get_random_sign_flip(self, eigvec, node_mask):
        """
        eigvec      [sum(node_num), dp]   --> [100, dp]
        node_mask   [batch, max(node_num)] --> [6, 30]
        return:     --> [100, dp]
        """
        b, max_n = node_mask.size() # b=6, max_n=30
        d = eigvec.size(1) # --> dp

        sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype) # sign_flip --> [6, dp]  从[0， 1)均匀分布取值
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0 # 最后调成 -1 和 1.的tensor # sign_flip --> [6, dp]
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d) # --> [6, 30, dp]
        sign_flip = sign_flip[node_mask] # --> [100, dp]
        return sign_flip # 里面只有1和-1

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        """
        eigvec      [sum(node_num), dp]   --> [100, dp]
        node_mask  [batch, max(node_num)] --> [6, 30]
        sign_flip:  Bool
        return: 看不懂 [18, 9, dp] or [18, 18]?
        """
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask) # --> [100, dp], 按照node_mask
            eigvec = eigvec * sign_flip # 对每个对应位置相乘，[100, dp]  应该像是随机改变正负符号
        else:
            pass
        return eigvec

    def fill_tensor_with_feature(self, batch, max_len, node_num, edge_num, lap_feature, edge_index):
        """
        for each row in eigvec_feature, get the batch eigenvec feature matrix from lap_feature.
        fill the lap_feature into the eigvec_feature tensor.
        Fill each rows at first index [0:node_num[i]] with batch_eigenvec one by one.
        and fill the last index [node_num[i]:edge_num[i]] with edge_lap_feature from compute_edge_lap_feature func.

        Args:
            batch: int, how many graphs in this lap_feature.
            max_len: int, max_node + max_edge in this batch.
            node_num: [batch], a list of node number in each graph.
            edge_num: [batch], a list of edge number in each graph.
            lap_feature: [sum(node_num), feature_dim]
            edge_index: [batch], a list of edge index in each graph. In each edge_index, each row index is the edge index, each row is a torch array of node index.
        return:
            eigvec_feature: [batch, max_len, feature_dim]
        """
        eigvec_feature = torch.zeros([batch, max_len, lap_feature.size(1)])
        start = 0
        for i in range(batch):
            batch_eigvec = lap_feature[start:start + node_num[i]]
            eigvec_feature[i, 0:node_num[i]] = batch_eigvec # 给node部分填充对应特征向量
            edge_lap_feature = self.compute_edge_lap_feature(edge_index[i], edge_num[i], batch_eigvec)
            eigvec_feature[i, node_num[i]:node_num[i] + edge_num[i]] = edge_lap_feature
            start += node_num[i]
        return eigvec_feature

    def compute_edge_lap_feature(self, edge_index, edge_num, batch_eigvec):
        """
        get the laplacian feature of each node, sum the laplacian feature of the nodes of each edge as the laplacian feature of the edge.

        Args:
            edge_index: [sum(edge_num)], each row index is the edge index, each row is a torch array of node index.
            edge_num: a int number, the number of edges in this graph.
            batch_eigvec: [node_num, feature_dim], the laplacian feature of each node.
        Returns:
            edge_lap_feature: [sum(edge_num), feature_dim]
        """
        edge_lap_feature = torch.zeros([edge_num, batch_eigvec.size(1)])
        for i in range(edge_num):
            edge_lap_feature[i] = batch_eigvec[edge_index[i]].sum(dim=0)
        return edge_lap_feature

def pad_batch(array, batch_size):
    return np.concatenate((array, np.arange(batch_size - len(array))))

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)# / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
