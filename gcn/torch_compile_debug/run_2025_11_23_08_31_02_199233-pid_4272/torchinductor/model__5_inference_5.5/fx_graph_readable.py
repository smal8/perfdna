class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[7, 16]", arg1_1: "f32[7]", arg2_1: "i64[2, 13264]", arg3_1: "f32[13264]", arg4_1: "f32[2708, 16]"):
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/nn/dense/linear.py:127 in forward, code: return F.linear(x, self.weight, self.bias)
        permute: "f32[16, 7]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "f32[2708, 7]" = torch.ops.aten.mm.default(arg4_1, permute);  arg4_1 = permute = None
        
        # File: /tmp/torch_geometric.nn.conv.gcn_conv_GCNConv_propagate_wlw1dik6.py:62 in collect, code: edge_index_i = edge_index[i]
        select: "i64[13264]" = torch.ops.aten.select.int(arg2_1, 0, 1)
        
        # File: /tmp/torch_geometric.nn.conv.gcn_conv_GCNConv_propagate_wlw1dik6.py:63 in collect, code: edge_index_j = edge_index[j]
        select_1: "i64[13264]" = torch.ops.aten.select.int(arg2_1, 0, 0);  arg2_1 = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/nn/conv/message_passing.py:265 in _index_select, code: return src.index_select(self.node_dim, index)
        index: "f32[13264, 7]" = torch.ops.aten.index.Tensor(mm, [select_1]);  mm = select_1 = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/nn/conv/gcn_conv.py:271 in message, code: return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        view: "f32[13264, 1]" = torch.ops.aten.view.default(arg3_1, [-1, 1]);  arg3_1 = None
        mul: "f32[13264, 7]" = torch.ops.aten.mul.Tensor(view, index);  view = index = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/utils/_scatter.py:144 in broadcast, code: return src.view(size).expand_as(ref)
        view_1: "i64[13264, 1]" = torch.ops.aten.view.default(select, [-1, 1]);  select = None
        expand: "i64[13264, 7]" = torch.ops.aten.expand.default(view_1, [13264, 7]);  view_1 = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/utils/_scatter.py:70 in scatter, code: return src.new_zeros(size).scatter_add_(dim, index, src)
        full_default: "f32[2708, 7]" = torch.ops.aten.full.default([2708, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter_add: "f32[2708, 7]" = torch.ops.aten.scatter_add.default(full_default, 0, expand, mul);  full_default = expand = mul = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/nn/conv/gcn_conv.py:266 in torch_dynamo_resume_in_forward_at_241, code: out = out + self.bias
        add: "f32[2708, 7]" = torch.ops.aten.add.Tensor(scatter_add, arg1_1);  scatter_add = arg1_1 = None
        return (add,)
        