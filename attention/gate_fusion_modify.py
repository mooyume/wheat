import torch
import torch.nn as nn
import torch.nn.functional as F


class BCFM(nn.Module):
    def __init__(self, d_model, hidden_size):
        super().__init__()
        # Modulation for F_r (using F_h)
        self.proj_h = nn.Linear(hidden_size, d_model)
        self.alpha_r = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.beta_r = nn.Linear(d_model, d_model)

        # Modulation for F_h (using F_r)
        self.proj_r = nn.Linear(d_model, hidden_size)
        self.alpha_h = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.beta_h = nn.Linear(hidden_size, hidden_size)

        # Fusion components
        self.fuse_proj = nn.Linear(hidden_size, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, F_r, F_h):
        # Modulate F_r with F_h
        F_h_proj = self.proj_h(F_h)
        alpha_r = self.alpha_r(F_h_proj)
        beta_r = self.beta_r(F_h_proj)
        F_r_mod = alpha_r * F_r + beta_r

        # Modulate F_h with F_r
        F_r_proj = self.proj_r(F_r)
        alpha_h = self.alpha_h(F_r_proj)
        beta_h = self.beta_h(F_r_proj)
        F_h_mod = alpha_h * F_h + beta_h

        # Project and fuse
        F_h_mod_proj = self.fuse_proj(F_h_mod)
        combined = torch.cat([F_r_mod, F_h_mod_proj], dim=-1)
        gate = self.gate(combined)

        return gate * F_r_mod + (1 - gate) * F_h_mod_proj


# 测试代码
if __name__ == "__main__":
    B, d_model, hidden_size = 32, 64, 32
    F_r = torch.randn(B, d_model)
    F_h = torch.randn(B, hidden_size)
    model = BCFM(d_model, hidden_size)
    output = model(F_r, F_h)
    print("Fused feature shape:", output.shape)  # Expected: (32, 64)