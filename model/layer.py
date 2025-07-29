import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, utils
from torchdrug.layers import functional


class GeneralizedRelationalConv(layers.MessagePassingBase):
    """
    广义关系卷积层，处理关系图中的消息传递
    支持多种消息函数和聚合方法
    """
    eps = 1e-6  # 数值稳定性常数

    # 消息函数到乘法操作的映射
    message2mul = {
        "transe": "add",  # TransE使用加法
        "distmult": "mul",  # DistMult使用乘法
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 layer_norm=True, activation="relu", dependent=True):
        """
        初始化关系卷积层
        
        参数:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            num_relation: 关系类型数量
            query_input_dim: 查询嵌入维度
            message_func: 消息函数类型，支持"distmult"、"transe"、"rotate"
            layer_norm: 是否使用层归一化
            activation: 激活函数
            dependent: 关系嵌入是否依赖于查询
        """
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func

        self.dependent = dependent  # 关系嵌入是否依赖于查询

        # 层归一化和激活函数
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # 线性变换层，用于合并输入和更新
        self.linear = nn.Linear(input_dim * 13, output_dim)

        # 关系嵌入生成方式
        if dependent:
            # 动态生成关系嵌入
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # 静态关系嵌入
            self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        """
        计算节点间的消息传递
        
        投影中的消息计算细节解析:
        1. 获取关系嵌入: 可以是静态的或依赖于查询
        2. 计算消息: 根据message_func定义的方式计算消息
           - distmult: 源节点特征与关系嵌入的元素级乘法(h⊙r)
           - transe: 源节点特征与关系嵌入的加法(h+r)
           - rotate: 复数域旋转操作，基于复数乘法
        3. 添加边界条件: 将初始节点表示作为自环消息
        
        数学解释:
        - DistMult (默认): message = h_x ⊙ r_{x,a_j}
          节点特征与关系特征的哈达玛积，捕获特征间的相互作用
        - TransE: message = h_x + r_{x,a_j}
          节点特征与关系特征的向量加法，表示关系作为翻译向量
        - RotatE: 将特征视为复数，通过复数乘法实现关系旋转
        
        投影算法中，消息计算是实现关系语义的关键步骤
        """
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()  # 分解边为源点、目标点、关系

        # 获取关系嵌入，可以是静态的或依赖于查询
        if self.dependent:
            # 从查询动态生成关系嵌入，增强表达能力
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            # 使用预定义的关系嵌入，与查询无关
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)  # [num_relation, batch_size, hidden_dim]

        # 获取源节点特征和对应的关系嵌入
        node_input = input[node_in]  # [num_edges, batch_size, hidden_dim]
        edge_input = relation_input[relation]  # [num_edges, batch_size, hidden_dim]

        # 根据消息函数类型计算消息
        if self.message_func == "transe":
            # TransE: 加法消息函数
            message = edge_input + node_input
        elif self.message_func == "distmult":
            # DistMult: 乘法消息函数(默认)
            message = edge_input * node_input
        elif self.message_func == "rotate":
            # RotatE: 复数域旋转操作
            node_re, node_im = node_input.chunk(2, dim=-1)  # 拆分实部和虚部
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im  # 复数乘法实部
            message_im = node_re * edge_im + node_im * edge_re  # 复数乘法虚部
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        
        # 添加边界条件消息(初始表示)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        """
        聚合来自邻居节点的消息
        
        参数:
            graph: 知识图谱
            message: 边上的消息 [num_edges+num_nodes, batch_size, hidden_dim]
            
        返回:
            节点聚合特征 [num_nodes, batch_size, hidden_dim*13]
        """
        node_out = graph.edge_list[:, 1]  # 目标节点索引
        # 添加自环
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        # 边权重
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)  # 扩展维度
        # 出度
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        # 计算四种统计聚合
        mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        
        # 计算标准差
        std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
        
        # 特征拼接与缩放
        features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)  # 压平最后两个维度
        
        # 基于度数计算缩放因子
        scale = degree_out.log()
        scale = scale / scale.mean()
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        
        # 应用缩放
        update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        return update

    def message_and_aggregate(self, graph, input):
        """
        优化版消息计算与聚合，直接使用稀疏矩阵操作提高效率
        
        投影中的消息聚合细节解析:
        1. 高效消息传递: 使用稀疏矩阵乘法代替显式消息计算
        2. 多重聚合: 计算四种统计量(sum/mean, sq_sum/sq_mean, max, min)
        3. 标准差计算: 基于均值和平方均值计算标准差
        4. 特征融合与缩放: 合并多种统计量并使用节点度数进行缩放
        
        数学解释:
        - 统计聚合: 四种统计量提供了不同视角的信息
          * mean: 平均邻居信息，表示一般趋势
          * max/min: 极端邻居信息，捕获重要特征
          * std: 邻居间的变异性，表示不确定性
        - 度数缩放: S_{a_j} = [1, log(deg(a_j)), 1/(log(deg(a_j))+η)]
          调整节点特征基于度数，防止高度节点主导
        
        这是投影算法中最核心的计算步骤，决定了如何聚合关系信息
        """
        # 特殊情况下回退到分步实现
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation
        batch_size = len(graph.query)
        input = input.flatten(1)  # [num_nodes, batch_size*hidden_dim]
        boundary = graph.boundary.flatten(1)  # [num_nodes, batch_size*hidden_dim]
        degree_out = graph.degree_out.unsqueeze(-1) + 1  # 出度加1，避免零除
        
        # 获取关系嵌入
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)  # [num_relation, batch_size*hidden_dim]
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)  # [num_relation, batch_size*hidden_dim]
        
        # 获取邻接矩阵
        adjacency = graph.adjacency.transpose(0, 1)  # [num_node, num_node, num_relation]
        
        # 确定消息函数对应的聚合操作
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # 使用稀疏矩阵计算四种统计量
        # 1. 求和/均值
        sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
        # 2. 平方和/平方均值
        sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
        # # 3. 最大值
        # max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
        # # 4. 最小值
        # min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
        
        # 计算统计特征
        mean = (sum + boundary) / degree_out  # 加入自环消息并除以度数
        sq_mean = (sq_sum + boundary ** 2) / degree_out
        # max = torch.max(max, boundary)  # 与自环消息比较取最大
        # min = torch.min(min, boundary)  # 与自环消息比较取最小
        std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()  # 标准差计算
        
        # 特征融合与缩放
        # 1. 拼接四种统计量
        features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        features = features.flatten(-2)  # 压平特征
        
        # 2. 计算度数相关的缩放因子
        scale = degree_out.log()
        scale = scale / scale.mean()  # 归一化
        scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
        
        # 3. 应用缩放因子
        update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)

        # 重塑张量维度为 [num_nodes, batch_size, hidden_dim*13]
        return update.view(len(update), batch_size, -1)

    def combine(self, input, update):
        """
        合并原始节点特征和聚合后的更新特征
        
        投影算法中的节点更新细节解析:
        1. 特征融合: 连接原始特征和聚合特征
        2. 线性变换: 通过全连接层降维并学习特征组合
        3. 标准化与激活: 应用层归一化和非线性激活函数
        
        数学解释:
        - 节点更新方程: h^(t)_{a_j} = φ(W · [h^(t-1)_{a_j} || u^(t)_{a_j}])
          其中 φ 是激活函数, || 表示特征连接操作
        - 这种更新方式保留了节点的历史信息并融合邻居信息
        
        投影算法中，这是完成单轮消息传递的最后一步，生成新的节点表示
        """
        # 连接原始特征和更新特征
        # input: [num_nodes, batch_size, hidden_dim]
        # update: [num_nodes, batch_size, hidden_dim*13]
        output = self.linear(torch.cat([input, update], dim=-1))
        
        # 应用层归一化
        if self.layer_norm:
            output = self.layer_norm(output)
        
        # 应用激活函数(通常是ReLU)
        if self.activation:
            output = self.activation(output)
        
        return output


class CompositionalGraphConv(layers.MessagePassingBase):
    """
    组合图卷积层，用于CompGCN模型
    使用组合操作处理关系和节点特征
    """
    # 消息函数映射到乘法操作
    message2mul = {
        "sub": "add",  # 减法使用加法(负关系)
        "mult": "mul",  # 乘法使用乘法
    }

    def __init__(self, input_dim, output_dim, num_relation, message_func="mult", layer_norm=False, activation="relu"):
        """
        初始化组合图卷积层
        
        参数:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            num_relation: 关系类型数量
            message_func: 消息函数类型，支持"mult"、"sub"、"corr"
            layer_norm: 是否使用层归一化
            activation: 激活函数
        """
        super(CompositionalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.message_func = message_func

        # 层归一化和激活函数
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        # 自环关系嵌入
        self.loop_relation = nn.Embedding(1, input_dim)
        # 线性变换层
        self.linear = nn.Linear(3 * input_dim, output_dim)
        self.relation_linear = nn.Linear(input_dim, output_dim)

    # 其余方法省略，与GeneralizedRelationalConv类似但处理逻辑不同
