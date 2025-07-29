from collections import Sequence

import torch
from torch import nn

from torchdrug import core, data, utils
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):
    """
    神经Bellman-Ford网络
    实现基于关系图的消息传递神经网络，专为API关系投影设计
    """

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, dependent=True):
        """
        初始化NBFNet模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度，可以是单个值或列表
            num_relation: 关系类型数量
            message_func: 消息函数类型，默认为"distmult"，支持"transe"和"rotate"
            short_cut: 是否使用残差连接
            layer_norm: 是否使用层归一化
            activation: 激活函数，默认为"relu"
            concat_hidden: 是否连接所有隐藏层输出
            dependent: 关系嵌入是否依赖于查询
        """
        super(NeuralBellmanFordNetwork, self).__init__()

        # 处理隐藏层维度，确保为列表形式
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        
        # 模型参数初始化
        self.input_dim = input_dim
        # 输出维度根据是否连接所有隐藏层决定
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)  # 各层维度列表
        self.num_relation = num_relation
        self.short_cut = short_cut  # 是否使用残差连接
        self.concat_hidden = concat_hidden  # 是否连接所有隐藏层

        # 创建多层关系卷积网络
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func,  layer_norm,
                                                               activation, dependent))

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        执行基于神经Bellman-Ford算法的多层消息传递
        
        投影算法关键步骤解析:
        1. 设置边界条件: 将RelationProjection中的初始化表示作为边界条件
           - 这一步使图中的节点接收到查询信号，边界条件表示起始状态
        2. 多层消息传递: 逐层迭代更新节点表示
           - 每一层都使用GeneralizedRelationalConv执行消息聚合
           - T轮迭代相当于考虑路径长度≤T的所有路径
        3. 连接最终表示与查询: 将GNN的输出与查询嵌入连接以增强表示能力
        
        数学原理:
        - 类似于Bellman-Ford算法寻找最短路径，但使用神经网络参数化
        - 多层迭代允许信息在图中传播T步，捕获更复杂的关系模式
        - 投影实质是基于图结构的消息传递，从查询节点传递到目标节点
        """
        # 设置边界条件，即从RelationProjection获得的初始表示
        with graph.node():
            graph.boundary = input  # [num_nodes, batch_size, hidden_dim]
        hiddens = []  # 存储各层的隐藏状态
        layer_input = input  # 第一层的输入

        # 多层消息传递，每层迭代更新节点表示
        for layer in self.layers:
            hidden = layer(graph, layer_input)  # 通过当前层关系卷积传递消息
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input  # 残差连接，帮助深层网络训练
            hiddens.append(hidden)  # 保存当前层输出
            layer_input = hidden  # 更新下一层的输入
        
        # 将查询嵌入扩展到所有节点，增强特征表示
        node_query = graph.query.expand(graph.num_node, -1, -1)  # [num_nodes, batch_size, hidden_dim]

        # 决定最终节点表示，连接隐藏层输出与查询
        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)  # 使用所有层的信息
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)  # 只使用最后一层
            
        return {
            "node_feature": node_feature,  # 返回最终节点特征
        }


@R.register("model.CompGCN")
class CompositionalGraphConvolutionalNetwork(nn.Module, core.Configurable):
    """
    组合图卷积网络
    通过组合操作处理关系和节点特征的GNN模型
    """

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="mult", short_cut=False, layer_norm=False,
                 activation="relu", concat_hidden=False):
        """
        初始化CompGCN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度，可以是单个值或列表
            num_relation: 关系类型数量
            message_func: 消息函数类型，默认为"mult"，支持"sub"和"corr"
            short_cut: 是否使用残差连接
            layer_norm: 是否使用层归一化
            activation: 激活函数，默认为"relu"
            concat_hidden: 是否连接所有隐藏层输出
        """
        super(CompositionalGraphConvolutionalNetwork, self).__init__()

        # 处理隐藏层维度
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        
        # 模型参数初始化
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        # 创建组合图卷积层
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.CompositionalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                            message_func, layer_norm, activation))
        # 关系嵌入初始化
        self.relation = nn.Embedding(num_relation, input_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        """
        前向传播函数
        
        参数:
            graph: 知识图谱
            input: 初始节点特征 [num_nodes, batch_size, hidden_dim]
            all_loss: 损失累积字典（可选）
            metric: 指标累积字典（可选）
            
        返回:
            包含最终节点特征的字典
        """
        # 设置图的关系嵌入
        graph.relation_input = self.relation.weight
        hiddens = []  # 存储各层的隐藏状态
        layer_input = input  # 第一层的输入

        # 逐层消息传递
        for layer in self.layers:
            hidden = layer(graph, layer_input)  # 通过当前层处理
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input  # 残差连接
            hiddens.append(hidden)  # 保存当前层输出
            layer_input = hidden  # 更新下一层的输入

        # 扩展查询嵌入维度以匹配所有节点
        node_query = graph.query.expand(graph.num_node, -1, -1)
        
        # 根据配置决定最终节点特征
        if self.concat_hidden:
            # 连接所有隐藏层特征和查询
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            # 只使用最后一层特征和查询
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": node_feature,  # 返回节点特征
        }
