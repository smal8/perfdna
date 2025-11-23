class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[2, 10556]"):
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/utils/loop.py:624 in add_remaining_self_loops, code: mask = edge_index[0] != edge_index[1]
        select: "i64[10556]" = torch.ops.aten.select.int(arg0_1, 0, 0)
        select_1: "i64[10556]" = torch.ops.aten.select.int(arg0_1, 0, 1);  arg0_1 = None
        ne: "b8[10556]" = torch.ops.aten.ne.Tensor(select, select_1);  select = select_1 = None
        
        # File: /usr/local/lib/python3.12/dist-packages/torch_geometric/utils/loop.py:634 in add_remaining_self_loops, code: loop_index = torch.arange(0, N, device=device).view(1, -1).repeat(2, 1)
        iota: "i64[2708]" = torch.ops.prims.iota.default(2708, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view: "i64[1, 2708]" = torch.ops.aten.view.default(iota, [1, -1]);  iota = None
        repeat: "i64[2, 2708]" = torch.ops.aten.repeat.default(view, [2, 1]);  view = None
        return (ne, repeat)
        