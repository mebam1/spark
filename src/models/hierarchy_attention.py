"""
Hierarchy Attention Module for Parent-Child Cross-Attention.

This module implements cross-attention between child parts and their parent parts,
enabling the model to learn hierarchical relationships in articulated objects.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from diffusers.models.attention import Attention
from diffusers.models.normalization import AdaLayerNormZero


class HierarchyAttentionBlock(nn.Module):
    """
    Cross-attention block for parent-child hierarchy.
    Child parts query parent features to incorporate hierarchical context.

    This implements a simple unidirectional cross-attention where children attend to parents.
    For bidirectional attention (parents also attending to children), set bidirectional=True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bidirectional: bool = False,
    ):
        """
        Args:
            dim: Hidden dimension size
            num_heads: Number of attention heads
            bidirectional: If True, also allow parents to query children (default: False)
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.dim = dim

        # Child -> Parent attention
        self.norm_child = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_parent = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.child_to_parent_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=True,
            out_bias=True,
        )

        # Optional: Parent -> Child attention
        if bidirectional:
            self.norm_parent_q = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm_child_kv = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.parent_to_child_attn = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                bias=True,
                out_bias=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        parent_indices: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply hierarchy attention.

        Args:
            hidden_states: [N, T, D] - features for all parts in the batch
                          N = total number of parts
                          T = number of tokens per part (e.g., T+1 time tokens + spatial tokens)
                          D = hidden dimension
            parent_indices: [N] - parent index for each part (-1 for root nodes)
                          Can be torch.Tensor or list (will be converted to tensor)
                          Must be global indices (already offset-adjusted for batched data)
            num_parts: [B] - number of parts per object in batch (for boundary validation)

        Returns:
            hidden_states: [N, T, D] - updated features after hierarchy attention
        """
        N, T, D = hidden_states.shape

        # Convert parent_indices to tensor if it's a list
        if isinstance(parent_indices, list):
            parent_indices = torch.tensor(parent_indices, device=hidden_states.device, dtype=torch.long)
        elif isinstance(parent_indices, torch.Tensor):
            parent_indices = parent_indices.to(device=hidden_states.device, dtype=torch.long)

        # Step 1: Gather parent features for each part (with boundary validation)
        parent_features = self._gather_parent_features(hidden_states, parent_indices, num_parts)  # [N, T, D]

        # Step 2: Child parts query parent features (unidirectional)
        # Normalize features
        child_query = self.norm_child(hidden_states)  # [N, T, D]
        parent_kv = self.norm_parent(parent_features)  # [N, T, D]

        # Apply cross-attention: child attends to parent
        attn_output = self.child_to_parent_attn(
            hidden_states=child_query,
            encoder_hidden_states=parent_kv,
        )  # [N, T, D]

        # Residual connection
        hidden_states = hidden_states + attn_output

        # Step 3 (Optional): Parent parts query child features (bidirectional)
        if self.bidirectional:
            # Gather aggregated children features for each part
            children_features = self._gather_children_features(
                hidden_states, parent_indices
            )  # [N, T, D]

            # Normalize
            parent_query = self.norm_parent_q(hidden_states)
            children_kv = self.norm_child_kv(children_features)

            # Apply cross-attention: parent attends to children
            attn_output = self.parent_to_child_attn(
                hidden_states=parent_query,
                encoder_hidden_states=children_kv,
            )  # [N, T, D]

            # Residual connection
            hidden_states = hidden_states + attn_output

        return hidden_states

    def _gather_parent_features(
        self,
        hidden_states: torch.Tensor,
        parent_indices: torch.Tensor,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gather parent features for each part with boundary validation.

        Args:
            hidden_states: [N, T, D]
            parent_indices: [N] - parent index for each part (global indices)
            num_parts: [B] - number of parts per object (for validation)

        Returns:
            parent_features: [N, T, D]
        """
        N, T, D = hidden_states.shape
        device = hidden_states.device

        # Boundary check: ensure all parent indices are within valid range
        valid_parents = parent_indices >= 0
        if valid_parents.any():
            max_parent = parent_indices[valid_parents].max().item()
            if max_parent >= N:
                raise ValueError(
                    f"Invalid parent index detected: max_parent={max_parent} >= N={N}. "
                    f"This indicates cross-sample parent reference or data corruption."
                )

        # Optional: Check that parents are within the same object (if num_parts provided)
        if num_parts is not None and len(num_parts) > 1:
            # Build object_id for each part: [0,0,0,1,1,2,2,2,...]
            object_ids = torch.repeat_interleave(
                torch.arange(len(num_parts), device=device),
                num_parts
            )  # [N]

            # For each part with a valid parent, check they're in the same object
            for i in range(N):
                parent_id = parent_indices[i].item()
                if parent_id >= 0:
                    if object_ids[i] != object_ids[parent_id]:
                        raise ValueError(
                            f"Cross-sample parent reference detected: "
                            f"part {i} (object {object_ids[i].item()}) has parent {parent_id} "
                            f"(object {object_ids[parent_id].item()})"
                        )

        # Handle root nodes: for root nodes (parent_indices == -1), use self as parent
        parent_indices_safe = parent_indices.clone()
        parent_indices_safe[parent_indices < 0] = torch.arange(N, device=device)[parent_indices < 0]

        # Gather parent features using advanced indexing
        # parent_indices_safe: [N] contains indices in range [0, N-1]
        parent_features = hidden_states[parent_indices_safe]  # [N, T, D]

        return parent_features

    def _gather_children_features(
        self,
        hidden_states: torch.Tensor,
        parent_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each part, gather and aggregate features of all its children.
        Optimized version using scatter_add for vectorized aggregation.

        Args:
            hidden_states: [N, T, D]
            parent_indices: [N] - parent index for each part

        Returns:
            children_features: [N, T, D] - aggregated children features for each part
        """
        N, T, D = hidden_states.shape
        device = hidden_states.device

        # Initialize accumulator and counter
        children_sum = torch.zeros_like(hidden_states)  # [N, T, D]
        children_count = torch.zeros(N, 1, 1, device=device)  # [N, 1, 1] for broadcasting

        # Mask for parts with valid parents (exclude roots)
        valid_mask = parent_indices >= 0  # [N]

        if valid_mask.any():
            # Get valid parent indices and corresponding features
            valid_parents = parent_indices[valid_mask]  # [M] where M = num of non-root parts
            valid_features = hidden_states[valid_mask]  # [M, T, D]

            # Expand indices for scatter_add: [M] -> [M, T, D]
            parent_indices_expanded = valid_parents.view(-1, 1, 1).expand(-1, T, D)

            # Accumulate children features to their parents using scatter_add
            children_sum.scatter_add_(0, parent_indices_expanded, valid_features)

            # Count children for each parent
            count_indices = valid_parents.view(-1, 1, 1).expand(-1, 1, 1)
            children_count.scatter_add_(0, count_indices, torch.ones_like(count_indices, dtype=torch.float32))

        # Compute average: divide sum by count
        # For parts with children: use average; for leaf nodes: use self features
        has_children = children_count > 0  # [N, 1, 1]
        children_features = torch.where(
            has_children.expand_as(children_sum),
            children_sum / children_count.expand_as(children_sum),
            hidden_states  # Leaf nodes use self features
        )

        return children_features


class HierarchyAttentionBlockWithModulation(nn.Module):
    """
    Hierarchy Attention Block with Adaptive Layer Norm (for DiT-style models).

    This version uses AdaLayerNormZero for modulation, similar to DiT blocks.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.dim = dim

        # Modulation for child->parent attention
        self.norm_child = AdaLayerNormZero(dim)
        self.norm_parent = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        self.child_to_parent_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            bias=True,
            out_bias=True,
        )

        if bidirectional:
            self.norm_parent_q = AdaLayerNormZero(dim)
            self.norm_child_kv = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

            self.parent_to_child_attn = Attention(
                query_dim=dim,
                cross_attention_dim=dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                bias=True,
                out_bias=True,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        parent_indices: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        num_parts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply hierarchy attention with timestep modulation.

        Args:
            hidden_states: [N, T, D]
            parent_indices: [N] - Can be torch.Tensor or list (will be converted)
                          Must be global indices (already offset-adjusted for batched data)
            temb: [N, D] - timestep embedding for modulation (optional)
            num_parts: [B] - number of parts per object in batch (for boundary validation)

        Returns:
            hidden_states: [N, T, D]
        """
        N, T, D = hidden_states.shape

        # Convert parent_indices to tensor if it's a list
        if isinstance(parent_indices, list):
            parent_indices = torch.tensor(parent_indices, device=hidden_states.device, dtype=torch.long)
        elif isinstance(parent_indices, torch.Tensor):
            parent_indices = parent_indices.to(device=hidden_states.device, dtype=torch.long)

        # Gather parent features (with boundary validation)
        parent_features = self._gather_parent_features(hidden_states, parent_indices, num_parts)

        # Child -> Parent attention with modulation
        if temb is not None:
            # Expand temb to match hidden_states shape
            temb_expanded = temb.unsqueeze(1).expand(-1, T, -1)  # [N, T, D]
            child_query, child_gate_msa, child_shift_mlp, child_scale_mlp, child_gate_mlp = \
                self.norm_child(hidden_states, temb_expanded)
        else:
            child_query = self.norm_child.norm(hidden_states)
            child_gate_msa = 1.0

        parent_kv = self.norm_parent(parent_features)

        attn_output = self.child_to_parent_attn(
            hidden_states=child_query,
            encoder_hidden_states=parent_kv,
        )

        hidden_states = hidden_states + child_gate_msa * attn_output

        # Bidirectional attention (if enabled)
        if self.bidirectional:
            children_features = self._gather_children_features(hidden_states, parent_indices)

            if temb is not None:
                parent_query, parent_gate_msa, _, _, _ = \
                    self.norm_parent_q(hidden_states, temb_expanded)
            else:
                parent_query = self.norm_parent_q.norm(hidden_states)
                parent_gate_msa = 1.0

            children_kv = self.norm_child_kv(children_features)

            attn_output = self.parent_to_child_attn(
                hidden_states=parent_query,
                encoder_hidden_states=children_kv,
            )

            hidden_states = hidden_states + parent_gate_msa * attn_output

        return hidden_states

    def _gather_parent_features(self, hidden_states, parent_indices, num_parts=None):
        """Same as HierarchyAttentionBlock._gather_parent_features (optimized)"""
        N, T, D = hidden_states.shape
        device = hidden_states.device

        # Boundary check: ensure all parent indices are within valid range
        valid_parents = parent_indices >= 0
        if valid_parents.any():
            max_parent = parent_indices[valid_parents].max().item()
            if max_parent >= N:
                raise ValueError(
                    f"Invalid parent index detected: max_parent={max_parent} >= N={N}. "
                    f"This indicates cross-sample parent reference or data corruption."
                )

        # Optional: Check that parents are within the same object (if num_parts provided)
        if num_parts is not None and len(num_parts) > 1:
            # Build object_id for each part
            object_ids = torch.repeat_interleave(
                torch.arange(len(num_parts), device=device),
                num_parts
            )

            # For each part with a valid parent, check they're in the same object
            for i in range(N):
                parent_id = parent_indices[i].item()
                if parent_id >= 0:
                    if object_ids[i] != object_ids[parent_id]:
                        raise ValueError(
                            f"Cross-sample parent reference detected: "
                            f"part {i} (object {object_ids[i].item()}) has parent {parent_id} "
                            f"(object {object_ids[parent_id].item()})"
                        )

        parent_indices_safe = parent_indices.clone()
        parent_indices_safe[parent_indices < 0] = torch.arange(N, device=device)[parent_indices < 0]

        parent_features = hidden_states[parent_indices_safe]
        return parent_features

    def _gather_children_features(self, hidden_states, parent_indices):
        """Same as HierarchyAttentionBlock._gather_children_features (optimized)"""
        N, T, D = hidden_states.shape
        device = hidden_states.device

        # Initialize accumulator and counter
        children_sum = torch.zeros_like(hidden_states)
        children_count = torch.zeros(N, 1, 1, device=device)

        # Mask for parts with valid parents (exclude roots)
        valid_mask = parent_indices >= 0

        if valid_mask.any():
            # Get valid parent indices and corresponding features
            valid_parents = parent_indices[valid_mask]
            valid_features = hidden_states[valid_mask]

            # Expand indices for scatter_add
            parent_indices_expanded = valid_parents.view(-1, 1, 1).expand(-1, T, D)

            # Accumulate children features to their parents
            children_sum.scatter_add_(0, parent_indices_expanded, valid_features)

            # Count children for each parent
            count_indices = valid_parents.view(-1, 1, 1).expand(-1, 1, 1)
            children_count.scatter_add_(0, count_indices, torch.ones_like(count_indices, dtype=torch.float32))

        # Compute average: divide sum by count
        has_children = children_count > 0
        children_features = torch.where(
            has_children.expand_as(children_sum),
            children_sum / children_count.expand_as(children_sum),
            hidden_states
        )

        return children_features
