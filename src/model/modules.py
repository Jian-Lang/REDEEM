import torch.nn as nn
import torch
import torch.nn.functional as F

class MACRouter(nn.Module):
    def __init__(self, hs=768, **kwargs) -> None:
        super().__init__()
        self.linear_rem = nn.Linear(hs, hs)
        self.linear_ret = nn.Linear(hs, hs)

    def forward(self, rem_fea, ret_fea, **kwargs):
        rem_fea = self.linear_rem(rem_fea)
        ret_fea = self.linear_ret(ret_fea)
        avg_rem_fea = rem_fea.mean(dim=1)
        avg_ret_fea = ret_fea.mean(dim=2)
        routing_score = torch.matmul(avg_rem_fea.unsqueeze(1), avg_ret_fea.transpose(2,1))
        return nn.Softmax(dim=1)(routing_score.squeeze(1))


class CMoEGenerator(nn.Module):
    def __init__(self, k, hs=768, **kwargs):
        super().__init__()
        self.router = MACRouter(hs=hs)
        self.num_expert = k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hs, hs),
                nn.ReLU(),
                nn.Linear(hs, hs)
            ) for _ in range(self.num_expert)
        ])

    def forward(self, rem_fea, ret_fea, **kwargs):
        routing_score = self.router(rem_fea, ret_fea)  # shape: [B, K]
        expert_outs = []
        for k in range(self.num_expert):
            expert_outs.append(self.experts[k](ret_fea[:, k]))  # [B, S, hs]
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, K, S, hs]
        gen_fea = torch.einsum('bksd,bk->bsd', expert_outs, routing_score)
        print(gen_fea.size())
        return gen_fea

class SAIMPrompter(nn.Module):
    def __init__(self, prompt_len,dim=768):
        super(SAIMPrompter, self).__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)
        self.pooling = nn.AdaptiveAvgPool2d((prompt_len, dim))

    def forward(self, V, r_t):
        prompt = self.attention(V.mean(dim=1), r_t)
        return prompt

    def attention(self, query, key_value):
        b, k, s, _ = key_value.shape

        q = self.q_proj(query).unsqueeze(1).expand(b, k, -1, -1)  
        k = self.k_proj(key_value)  
        v = self.v_proj(key_value) 

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)  
        attn_probs = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, v)  
        output = self.pooling(output)
        output = self.linear_out(output)
        output = output.mean(dim=1)
        return output
    
