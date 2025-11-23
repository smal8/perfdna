class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2708, 16]"):
        # File: /content/perfdna/gcn/gcn.py:13 in torch_dynamo_resume_in_forward_at_12, code: x = F.relu(x)
        relu: "f32[2708, 16]" = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        return (relu,)
        