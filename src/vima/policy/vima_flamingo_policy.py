import torch
from einops import rearrange
from torch import nn

import vima.nn as vnn


class VIMAFlamingoPolicy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        dt_n_layers: int,
        dt_n_heads: int,
        xattn_n_heads: int,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=dt_n_layers,
            n_head=dt_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )

        self.obj_encoder = vnn.MultiViewRGBPerceiverEncoder(
            emb_dim=embed_dim,
            views=["front", "top"],
            img_size=(64, 128),
            vit_patch_size=32,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            perceiver_num_queries=4,
            perceiver_num_blocks=4,
            perceiver_num_self_attends_per_block=4,
            perceiver_num_self_attention_heads=8,
            perceiver_num_cross_attention_heads=8,
            perceiver_attention_probs_dropout_prob=0.1,
        )
        self._obj_xf_num_queries = 4

        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, embedding_dim=2)

        obs_feat_dim = self.obj_encoder.output_dim + 2
        self.obs_fusion_layer = (
            nn.Identity() if obs_feat_dim == embed_dim else nn.Linear(obs_feat_dim, embed_dim)
        )

        self.action_encoder = vnn.ActionEmbedding(
            output_dim=embed_dim,
            embed_dict={
                "pose0_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            },
        )
        self.action_decoder = vnn.ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        self.prompt_embedding = vnn.WordEmbedding()
        self.t5_prompt_encoder = vnn.T5PromptEncoder()
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )

        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )

        self._views = ["front", "top"]
        self._n_discrete_x_bins = 50
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 50
        self._n_discrete_rot_bins = 50

    def forward(
        self,
        obs_token: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ):
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        L = L_obs * self._obj_xf_num_queries + L_action

        tokens = torch.empty(L, B, self.embed_dim, dtype=torch.float32, device=self.device)
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        for q in range(self._obj_xf_num_queries):
            tokens[q :: self._obj_xf_num_queries + 1] = obs_token[q :: self._obj_xf_num_queries]
        if action_token is not None:
            tokens[self._obj_xf_num_queries :: self._obj_xf_num_queries + 1] = action_token
        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
        )
        predicted_action_tokens = tokens_out[
            self._obj_xf_num_queries - 1 :: self._obj_xf_num_queries + 1
        ]
        return predicted_action_tokens

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += self._obj_xf_num_queries
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)
        word_batch.shape[0]
        batch_word_emb = self.prompt_embedding(word_batch)
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        prompt_tokens, prompt_masks = [], []
        word_ptr, img_ptr = 0, 0
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[word_ptr])
                    word_ptr += 1
                elif item == 1:
                    for q in range(self._obj_xf_num_queries):
                        assembled_prompt.append(batch_image_emb[img_ptr][q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            valid_tokens = len(assembled_prompt)
            num_padding = L_max - valid_tokens
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)
            prompt_masks.append(
                torch.cat(
                    [
                        torch.ones(valid_tokens, dtype=torch.bool, device=self.device),
                        torch.zeros(num_padding, dtype=torch.bool, device=self.device),
                    ],
                    dim=0,
                )
            )
        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)
        prompt_tokens = self.t5_prompt_encoder(
            prompt_tokens, attention_mask=prompt_masks, batch_first=False
        )
        prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):
        rgbs, ee = obs["rgb"], obs["ee"]
        leading_dims = ee.shape[:2]
        rgbs = rgbs.map_structure(func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(rgb=rgbs)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])
        ee_feats = self.end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(1, 1, self._obj_xf_num_queries, 1)
        obs_feats = self.obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))
        return obs_feats

    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):
        device = action["pose0_position"].device
        boundary_x = torch.linspace(start=0, end=1, steps=self._n_discrete_x_bins, device=device)
        boundary_y = torch.linspace(start=0, end=1, steps=self._n_discrete_y_bins, device=device)
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = actions["pose0_rotation"] / self._n_discrete_rot_bins

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = actions["pose1_rotation"] / self._n_discrete_rot_bins
        return actions
