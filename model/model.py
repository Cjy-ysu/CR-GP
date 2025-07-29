import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Stack

@R.register("model.GNN")
class QueryExecutor(nn.Module, core.Configurable):
    """
    逻辑查询执行器，用于执行复杂的API关系查询
    处理逻辑操作（交集、并集、否定）和关系投影操作
    """
    stack_size = 10  # 操作数栈大小

    def __init__(self, model, dropout_ratio=0, num_mlp_layer=2):
        """
        初始化查询执行器
        
        参数:
            model: 底层GNN模型，用于关系投影
            dropout_ratio: 边丢弃率，用于训练时创建不完整图
            num_mlp_layer: MLP层数，用于最终预测
        """
        super(QueryExecutor, self).__init__()
        self.model = RelationProjection(model, num_mlp_layer)  # 关系投影模型
        self.symbolic_model = SymbolicTraversal()  # 符号化遍历模型
        self.dropout_ratio = dropout_ratio  # 边丢弃率

    def traversal_dropout(self, graph, h_prob, r_index):
        """
        丢弃可直接遍历的边以创建不完整图，用于训练时增强模型鲁棒性
        
        参数:
            graph: 知识图谱
            h_prob: 查询节点概率分布
            r_index: 关系类型索引
        
        返回:
            经过边丢弃的图
        """
        # 查找与查询节点和关系匹配的边
        sample, h_index = h_prob.nonzero().t()
        r_index = r_index[sample]
        any = -torch.ones_like(h_index)
        pattern = torch.stack([h_index, any, r_index], dim=-1)
        inverse_pattern = torch.stack([any, h_index, r_index ^ 1], dim=-1)
        pattern = torch.cat([pattern, inverse_pattern])
        edge_index = graph.match(pattern)[0]

        # 避免删除度为1的节点的边
        h_index, t_index = graph.edge_list.t()[:2]
        degree_h = h_index.bincount()
        degree_t = t_index.bincount()
        h_index, t_index = graph.edge_list[edge_index, :2].t()
        must_keep = (degree_h[h_index] <= 1) | (degree_t[t_index] <= 1)
        edge_index = edge_index[~must_keep]

        # 随机丢弃边
        is_sampled = torch.rand(len(edge_index), device=self.device) <= self.dropout_ratio
        edge_index = edge_index[is_sampled]

        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def execute(self, graph, query, all_loss=None, metric=None):
        """
        在图上执行查询操作
        
        参数:
            graph: 知识图谱
            query: 查询操作序列（后缀表示法）
            all_loss: 损失累积字典
            metric: 指标累积字典
        """
        batch_size = len(query)
        # 初始化操作数栈和变量栈
        self.stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        self.var = Stack(batch_size, query.shape[1], graph.num_node, device=self.device)
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=self.device)  # 指令指针
        
        all_sample = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        op = query[all_sample, self.IP]  # 获取当前操作
        
        # 执行查询，直到所有查询完成
        while not op.is_stop().all():
            # 识别操作类型
            is_operand = op.is_operand()
            is_intersection = op.is_intersection()
            is_union = op.is_union()
            is_negation = op.is_negation()
            is_projection = op.is_projection()
            
            # 根据操作类型执行相应操作
            if is_operand.any():
                h_index = op[is_operand].get_operand()
                self.apply_operand(is_operand, h_index, graph.num_node)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            if is_negation.any():
                self.apply_negation(is_negation)
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                r_index = op[is_projection].get_operand()
                self.apply_projection(is_projection, graph, r_index, all_loss=all_loss, metric=metric)

            op = query[all_sample, self.IP]  # 更新当前操作

        # 检查栈状态
        if (self.stack.SP > 1).any():
            raise ValueError("More operands than expected")

    def forward(self, graph, query, all_loss=None, metric=None):
        """
        前向传播函数，执行查询并返回logit
        
        参数:
            graph: 知识图谱
            query: 查询操作序列
            all_loss: 损失累积字典
            metric: 指标累积字典
        
        返回:
            查询结果的logit值
        """
        self.execute(graph, query, all_loss=all_loss, metric=metric)

        # 从栈中弹出结果并转换为logit
        t_prob = self.stack.pop()
        t_logit = ((t_prob + 1e-10) / (1 - t_prob + 1e-10)).log()
        return t_logit

    def visualize(self, graph, full_graph, query):
        """
        可视化查询执行过程
        
        参数:
            graph: 当前知识图谱
            full_graph: 完整知识图谱
            query: 查询操作序列
        
        返回:
            变量概率、部分答案和完整答案
        """
        self.execute(graph, query)
        var_probs = self.var.stack
        answers = self.symbolic_var.stack

        self.execute(full_graph, query)
        all_answers = self.symbolic_var.stack

        return var_probs, answers, all_answers

    def apply_operand(self, mask, h_index, num_node):
        """
        应用操作数操作：将节点转换为one-hot概率分布并压入栈
        
        参数:
            mask: 需要应用操作的批次掩码
            h_index: 节点索引
            num_node: 节点总数
        """
        h_prob = functional.one_hot(h_index, num_node)  # 节点one-hot编码
        self.stack.push(mask, h_prob)  # 压入操作数栈
        self.var.push(mask, h_prob)    # 压入变量栈
        self.IP[mask] += 1  # 更新指令指针

    def apply_intersection(self, mask):
        """
        应用交集操作：计算两个概率分布的逻辑与
        
        参数:
            mask: 需要应用操作的批次掩码
        """
        y_prob = self.stack.pop(mask)  # 弹出第二个操作数
        x_prob = self.stack.pop(mask)  # 弹出第一个操作数
        z_prob = self.conjunction(x_prob, y_prob)  # 计算交集
        self.stack.push(mask, z_prob)  # 压入结果
        self.var.push(mask, z_prob)    # 更新变量
        self.IP[mask] += 1  # 更新指令指针

    def apply_union(self, mask):
        """
        应用并集操作：计算两个概率分布的逻辑或
        
        参数:
            mask: 需要应用操作的批次掩码
        """
        y_prob = self.stack.pop(mask)  # 弹出第二个操作数
        x_prob = self.stack.pop(mask)  # 弹出第一个操作数
        z_prob = self.disjunction(x_prob, y_prob)  # 计算并集
        self.stack.push(mask, z_prob)  # 压入结果
        self.var.push(mask, z_prob)    # 更新变量
        self.IP[mask] += 1  # 更新指令指针

    def apply_negation(self, mask):
        """
        应用否定操作：计算概率分布的逻辑非
        
        参数:
            mask: 需要应用操作的批次掩码
        """
        x_prob = self.stack.pop(mask)  # 弹出操作数
        y_prob = self.negation(x_prob)  # 计算否定
        self.stack.push(mask, y_prob)  # 压入结果
        self.var.push(mask, y_prob)    # 更新变量
        self.IP[mask] += 1  # 更新指令指针

    def apply_projection(self, mask, graph, r_index, all_loss=None, metric=None):
        """
        应用投影操作：通过关系将起始节点投影到目标节点
        
        关键步骤解析:
        1. 从操作数栈中取出查询节点分布h_prob，表示起点的概率分布
        2. 使用RelationProjection模型执行关系投影，将h_prob通过关系r_index映射到t_prob
        3. 这里的t_prob表示与查询节点有关系r_index的目标节点的概率分布
        
        输入输出维度分析:
        - h_prob: [batch_size, num_nodes]，表示每个批次中查询节点的概率分布
        - r_index: [batch_size]，表示每个批次要查询的关系类型
        - t_prob: [batch_size, num_nodes]，表示目标节点的预测概率分布
        """
        h_prob = self.stack.pop(mask)  # 从栈中弹出查询节点概率分布
        h_prob = h_prob.detach()  # 分离计算图，防止梯度传递到前面的操作
        # 执行关系投影，核心投影步骤
        t_prob = self.model(graph, h_prob, r_index, all_loss=all_loss, metric=metric)
        self.stack.push(mask, t_prob)  # 将投影结果压入栈
        self.var.push(mask, t_prob)    # 同时记录在变量栈中
        self.IP[mask] += 1  # 更新指令指针

    def conjunction(self, x, y):
        """
        概率分布的逻辑与操作：元素级乘法
        """
        return x * y

    def disjunction(self, x, y):
        """
        概率分布的逻辑或操作：p(A or B) = p(A) + p(B) - p(A and B)
        """
        return x + y - x * y

    def negation(self, x):
        """
        概率分布的逻辑非操作：1 - p
        """
        return 1 - x


@R.register("model.RelationProjection")
class RelationProjection(nn.Module, core.Configurable):
    """
    关系投影模型，核心投影实现类
    将起始节点通过关系投影到目标节点
    """
    def __init__(self, model, num_mlp_layer=2):
        """
        初始化关系投影模型
        
        参数:
            model: 底层GNN模型，如NBFNet
            num_mlp_layer: MLP层数，用于最终预测
        """
        super(RelationProjection, self).__init__()
        self.model = model  # GNN模型
        self.query = nn.Embedding(model.num_relation, model.input_dim)  # 关系嵌入
        self.mlp = layers.MLP(model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])  # 预测层

    def forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        """
        执行关系投影的核心算法
        
        关键步骤解析:
        1. 获取关系嵌入: 将关系ID转换为向量表示
        2. 初始化节点表示: 使用爱因斯坦求和约定计算初始表示
           - 该操作实质是将查询节点分布与关系嵌入相结合
           - "bn,bd->nbd"意味着: 批次×节点 与 批次×维度 相乘得到 节点×批次×维度
        3. 调用GNN模型进行消息传递: 通过图结构传播信息
        4. 使用MLP和sigmoid转换节点特征为关系概率
        
        数学解释:
        - einsum操作: input_{n,b,d} = h_prob_{b,n} * query_{b,d}
          这是概率分布与关系嵌入的外积，为每个节点初始化特征
        - 节点n的初始特征值与查询节点概率h_prob[b,n]成正比
        - 若节点n在某批次b中是查询节点(h_prob[b,n]=1)，则input[n,b]将完全等于关系嵌入query[b]
        """
        query = self.query(r_index)  # 获取关系嵌入 [batch_size, hidden_dim]
        graph = graph.clone()  # 克隆图以避免修改原图
        with graph.graph():
            graph.query = query  # 将关系嵌入附加到图上，供后续GNN使用
        
        # 关键投影操作: 将查询节点概率分布与关系嵌入结合
        # h_prob: [batch_size, num_nodes], query: [batch_size, hidden_dim]
        # 输出input: [num_nodes, batch_size, hidden_dim]
        # 这是投影初始化的核心步骤，决定了初始节点表示
        input = torch.einsum("bn, bd -> nbd", h_prob, query)
        
        # 通过GNN传播信息，执行消息传递
        output = self.model(graph, input, all_loss=all_loss, metric=metric)
        
        # 将GNN输出转换为关系存在概率
        # output["node_feature"]: [num_nodes, batch_size, output_dim]
        t_prob = F.sigmoid(self.mlp(output["node_feature"]).squeeze(-1))  # [num_nodes, batch_size]
        
        # 转置结果，使批次维度在前
        return t_prob.t()  # [batch_size, num_nodes]


@R.register("model.Symbolic")
class SymbolicTraversal(nn.Module, core.Configurable):
    """
    符号化遍历算法，用于验证和比较
    """
    def forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        """
        执行符号化遍历
        
        参数:
            graph: 知识图谱
            h_prob: 起始节点概率分布
            r_index: 关系类型索引
            all_loss: 损失累积字典
            metric: 指标累积字典
            
        返回:
            目标节点概率分布
        """
        batch_size = len(h_prob)
        # 匹配指定关系类型的所有边
        any = -torch.ones_like(r_index)
        pattern = torch.stack([any, any, r_index], dim=-1)
        edge_index, num_edges = graph.match(pattern)
        
        # 为每个批次创建打包图
        num_nodes = graph.num_node.repeat(batch_size)
        graph = data.PackedGraph(graph.edge_list[edge_index], num_nodes=num_nodes, num_edges=num_edges)

        # 创建邻接矩阵并执行稀疏矩阵乘法
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight,
                                            (graph.num_node, graph.num_node))
        # 使用max聚合执行符号化遍历
        t_prob = functional.generalized_spmm(adjacency.t(), h_prob.view(-1, 1), sum="max").clamp(min=0)

        return t_prob.view_as(h_prob)
