import torch
import torch.nn as nn


class DCFF(nn.Module):
    def __init__(self, d_model, hidden_size,
                 use_modulation=True,
                 use_gating=True,
                 use_residual=True):
        super().__init__()
        self.use_modulation = use_modulation
        self.use_gating = use_gating
        self.use_residual = use_residual

        # Context encoder
        self.ctx_encoder = nn.Linear(hidden_size, d_model)

        # Modulation parameters
        if use_modulation:
            self.modulator = nn.Linear(d_model, 4 * d_model)  # γ_s,β_s,γ_e,β_e

        # Gating mechanism
        if use_gating:
            self.gate = nn.Linear(2 * d_model, d_model)

        # Residual projection
        if use_residual:
            self.res_proj = nn.Linear(hidden_size, d_model)
            self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, S, E, H):
        # Context encoding
        C = nn.functional.relu(self.ctx_encoder(H))

        # Feature modulation
        if self.use_modulation:
            γ_s, β_s, γ_e, β_e = self.modulator(C).chunk(4, dim=-1)
            S_prime = γ_s * S + β_s
            E_prime = γ_e * E + β_e
        else:
            S_prime, E_prime = S, E  # 消融调制功能

        # Feature fusion
        if self.use_gating:
            gate_input = torch.cat([S_prime, E_prime], dim=-1)
            G = torch.sigmoid(self.gate(gate_input))
            F = G * S_prime + (1 - G) * E_prime
        else:
            F = (S_prime + E_prime) / 2  # 消融门控功能

        # Residual enhancement
        if self.use_residual:
            H_res = self.res_proj(H)
            output = self.layer_norm(F + H_res)
        else:
            output = F  # 消融残差功能

        return output


if __name__ == '__main__':
    # 使用示例
    d_model = 512
    hidden_size = 256
    batch_size = 32

    S = torch.randn(batch_size, d_model)
    E = torch.randn(batch_size, d_model)
    H = torch.randn(batch_size, hidden_size)

    # 完整模型
    full_model = DCFF(d_model=512, hidden_size=256)
    output = full_model(S, E, H)
    print(output.shape)  # torch.Size([32, 512])

    # 消融调制功能
    ablation1 = DCFF(512, 256, use_modulation=False)
    output = ablation1(S, E, H)
    print(output.shape)  # torch.Size([32, 512])

    # 消融门控功能
    ablation2 = DCFF(512, 256, use_gating=False)
    output = ablation2(S, E, H)
    print(output.shape)  # torch.Size([32, 512])

    # 消融残差功能
    ablation3 = DCFF(512, 256, use_residual=False)
    output = ablation3(S, E, H)
    print(output.shape)  # torch.Size([32, 512])

    # 同时消融多个组件
    ablation4 = DCFF(512, 256,
                     use_modulation=False,
                     use_gating=False,
                     use_residual=False)
    output = ablation4(S, E, H)
    print(output.shape)  # torch.Size([32, 512])
