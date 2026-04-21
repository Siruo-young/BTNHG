import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv

from BTNHGV2ParameterClass import BTNHGV2ParameterClass
from ExtendedNNModule import ExtendedNNModule


class RAHeteroGCN(ExtendedNNModule):
	"""
	Relation-Aware + Adaptive-weight HeteroGCN

	设计要点：
	1) 每条 relation 使用独立 GraphConv 进行消息传递；
	2) 对同一目标节点类型的多 relation 消息，引入 node-level attention（每个节点自适应权重）：
	   alpha = softmax(MLP(x_dst))，再对各 relation 消息加权求和；
	3) 每层后保留残差、LayerNorm、激活和 Dropout，提高训练稳定性。
	"""
	def __init__(
		self,
		heteroData: HeteroData,
		hidden_channels=BTNHGV2ParameterClass.hidden_channels,
		num_layers=BTNHGV2ParameterClass.num_layers,
		dropout=BTNHGV2ParameterClass.dropout,
	):
		super().__init__()
		self.heteroData = heteroData
		self.hidden_channels = hidden_channels
		self.num_layers = num_layers
		self._dropout = nn.Dropout(p=dropout)
		self._edge_types = list(self.heteroData.edge_types)
		self._node_types = list(self.heteroData.node_types)

		valid_y = self.heteroData["address"].y
		valid_y = valid_y[valid_y != -1]
		self.num_classes = int(valid_y.unique().numel())

		# 输入投影：将不同节点类型映射到同一 hidden 空间
		self.input_proj = nn.ModuleDict({
			node_type: nn.Linear(
				self.heteroData[node_type].x.size(-1),
				self.hidden_channels
			)
			for node_type in self._node_types
		})

		# 每层每关系独立卷积
		self.rel_convs = nn.ModuleList()
		for _ in range(self.num_layers):
			conv_dict = nn.ModuleDict()
			for edge_type in self._edge_types:
				edge_key = self._edge_type_to_key(edge_type)
				conv_dict[edge_key] = GraphConv(
					in_channels=self.hidden_channels,
					out_channels=self.hidden_channels
				)
			self.rel_convs.append(conv_dict)

		# 每层、每目标节点类型：node-level relation attention
		# 对每个 dst 类型，基于 x_dst 预测其对各 incoming relation 的权重（每个节点一套权重）
		self.rel_att_mlps = nn.ModuleList()
		for _ in range(self.num_layers):
			mlp_dict = nn.ModuleDict()
			for dst in self._node_types:
				rels_to_dst = self._incoming_rels(dst)
				if len(rels_to_dst) > 0:
					mlp_dict[dst] = nn.Linear(self.hidden_channels, len(rels_to_dst))
			self.rel_att_mlps.append(mlp_dict)

		self.norms = nn.ModuleList([
			nn.ModuleDict({
				node_type: nn.LayerNorm(self.hidden_channels)
				for node_type in self._node_types
			})
			for _ in range(self.num_layers)
		])

		# 两层分类头：与 HeteroGCN 保持一致风格
		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_channels, self.hidden_channels),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.Linear(self.hidden_channels, self.num_classes),
		)

	def _edge_type_to_key(self, edge_type):
		src, rel, dst = edge_type
		return f"{src}__{rel}__{dst}"

	def _incoming_rels(self, dst_node_type):
		return [et for et in self._edge_types if et[2] == dst_node_type]

	def _adaptive_aggregate(self, layer_idx, x_dict, edge_index_dict):
		"""
		按目标节点类型聚合 relation 消息：
		h_dst = sum_j alpha_{v,j} * msg_j，其中 alpha_{v,:} = softmax(MLP(x_dst[v]))。
		"""
		conv_dict = self.rel_convs[layer_idx]
		att_mlp_dict = self.rel_att_mlps[layer_idx]
		out_dict = {ntype: None for ntype in self._node_types}

		for dst in self._node_types:
			rels = self._incoming_rels(dst)
			if len(rels) == 0:
				# 没有入边关系时，保留原表示
				out_dict[dst] = x_dict[dst]
				continue

			msgs = []
			for edge_type in rels:
				src, _, _ = edge_type
				edge_key = self._edge_type_to_key(edge_type)
				edge_index = edge_index_dict[edge_type]
				msg = conv_dict[edge_key]((x_dict[src], x_dict[dst]), edge_index)
				msgs.append(msg)

			# node-level attention: alpha shape [num_dst_nodes, num_rels_to_dst]
			alpha = F.softmax(att_mlp_dict[dst](x_dict[dst]), dim=-1)

			# msgs -> [num_dst_nodes, num_rels_to_dst, hidden]
			msg_stack = torch.stack(msgs, dim=1)
			out_dict[dst] = (alpha.unsqueeze(-1) * msg_stack).sum(dim=1)

		return out_dict

	def forward(self, hetero_data: HeteroData) -> torch.Tensor:
		x_dict = {
			node_type: self.input_proj[node_type](hetero_data[node_type].x)
			for node_type in hetero_data.node_types
		}
		edge_index_dict = hetero_data.edge_index_dict

		for layer_idx in range(self.num_layers):
			h_dict = self._adaptive_aggregate(layer_idx, x_dict, edge_index_dict)
			new_x_dict = {}
			for node_type in self._node_types:
				h = h_dict[node_type]
				# 残差稳定训练，防止深层过度退化
				h = h + x_dict[node_type]
				h = self.norms[layer_idx][node_type](h)
				h = F.relu(h)
				h = self._dropout(h)
				new_x_dict[node_type] = h
			x_dict = new_x_dict

		return self.classifier(x_dict["address"])
