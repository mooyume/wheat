import torch
import torch.nn as nn

class CGHYIRevisedFusionModule(nn.Module):
    def __init__(self, d_model, hidden_size):
        super(CGHYIRevisedFusionModule, self).__init__()
        self.gate_ref_layer = nn.Linear(2 * d_model, 1)
        self.gate_env_layer = nn.Linear(2 * d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.ref_proj = nn.Linear(d_model, hidden_size) # Project gated ref feature to hidden_size
        self.env_proj = nn.Linear(d_model, hidden_size) # Project gated env feature to hidden_size


        # CGHYI Module Layers
        self.gate_yield_replace_layer = nn.Linear(2 * d_model, 1) # Gate for yield replacement
        self.yield_generate_layer = nn.Linear(2 * d_model, hidden_size) # Generate alternative yield feature


    def forward(self, feat_ref, feat_env, feat_yield_lstm=None): # feat_yield_lstm can be None
        # Context-Gated Mechanism for Ref and Env
        context = torch.cat([feat_ref, feat_env], dim=-1) # (B, 2*d_model)
        gate_ref = self.sigmoid(self.gate_ref_layer(context)) # (B, 1)
        gate_env = self.sigmoid(self.gate_env_layer(context)) # (B, 1)

        feat_ref_gated = gate_ref * feat_ref # (B, d_model)
        feat_env_gated = gate_env * feat_env # (B, d_model)

        # CGHYI: Conditional Gating for Yield Feature
        gate_yield_replace = self.sigmoid(self.gate_yield_replace_layer(context)) # (B, 1)
        feat_yield_alternative = self.yield_generate_layer(context) # (B, hidden_size)

        if feat_yield_lstm is not None: # Historical yield data exists (e.g., 2002-2022)
            feat_yield_processed = (1 - gate_yield_replace) * feat_yield_lstm + gate_yield_replace * feat_yield_alternative # Conditional selection
        else: # Historical yield data is missing (e.g., 2001)
            feat_yield_processed = feat_yield_alternative # Use alternative generated feature

        # Feature Fusion (Summation) - Project Ref and Env to hidden_size before fusion
        feat_ref_proj = self.ref_proj(feat_ref_gated) # (B, hidden_size)
        feat_env_proj = self.env_proj(feat_env_gated) # (B, hidden_size)
        feat_fuse = feat_ref_proj + feat_env_proj + feat_yield_processed # (B, hidden_size)

        return feat_fuse

if __name__ == '__main__':
    batch_size = 4
    d_model = 128
    hidden_size = 64

    feat_ref = torch.randn(batch_size, d_model)
    feat_env = torch.randn(batch_size, d_model)
    feat_yield_lstm = torch.randn(batch_size, hidden_size) # LSTM output, shape (B, hidden_size)

    # Example usage when historical yield data is available (e.g., 2002-2022)
    fusion_module_cghyi_revised = CGHYIRevisedFusionModule(d_model, hidden_size)
    fused_feat_with_yield_revised = fusion_module_cghyi_revised(feat_ref, feat_env, feat_yield_lstm)
    print("Fused feature (Revised CGHYI, with yield data):", fused_feat_with_yield_revised.shape) # Output: torch.Size([4, 64])

    # Example usage when historical yield data is missing (e.g., 2001)
    fused_feat_no_yield_revised = fusion_module_cghyi_revised(feat_ref, feat_env, feat_yield_lstm=None) # feat_yield_lstm = None
    print("Fused feature (Revised CGHYI, no yield data):", fused_feat_no_yield_revised.shape) # Output: torch.Size([4, 64])