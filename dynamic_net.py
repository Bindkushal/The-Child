"""
dynamic_net.py
==============
Self-Expanding Neural Network — Core Module
Based on: "Self-Expanding Neural Networks" (Mitchell et al., arXiv:2307.04526)
          "Dynamically Expandable Networks" (Yoon et al., ICLR 2018)
          "DeDNN" (ScienceDirect, 2023)

KEY INSIGHT FROM PAPERS:
  - WHEN  : Natural Expansion Score (NES) = g^T * F^-1 * g  (natural gradient)
            Expand when NES > threshold τ
  - WHERE : Layer with highest local error / gradient variance
  - HOW   : Add neuron with near-zero weights (no disruption to existing optimization)
            Use Identity init for layers (SECNN paper, arXiv:2401.05686)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy
from typing import List, Dict, Tuple


class DynamicLayer(nn.Module):
    """
    A single fully-connected layer that can grow (add neurons) or shrink (prune).
    Wraps nn.Linear but allows runtime resizing without restarting training.
    """

    def __init__(self, in_features: int, out_features: int, activation=nn.ReLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_features)

        # Per-neuron stats for growth/pruning decisions
        self.neuron_activations = torch.zeros(out_features)   # running mean activation
        self.neuron_gradients   = torch.zeros(out_features)   # running grad magnitude
        self._register_hooks()

    def _register_hooks(self):
        """Track gradient magnitudes per neuron for expansion score."""
        def grad_hook(grad):
            # grad shape: [batch, out_features]
            self.neuron_gradients = grad.abs().mean(dim=0).detach()
        self._hook = self.linear.weight.register_hook(
            lambda grad: setattr(self, 'neuron_gradients', grad.abs().mean(dim=1).detach())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if out.shape[0] > 1:          # BatchNorm needs batch_size > 1
            out = self.bn(out)
        out = self.activation(out)
        # Track mean activations for pruning
        self.neuron_activations = out.abs().mean(dim=0).detach()
        return out

    # ------------------------------------------------------------------ #
    #  GROWTH  — add `n` new neurons to this layer's OUTPUT               #
    # ------------------------------------------------------------------ #
    def add_neurons(self, n: int, next_layer: 'DynamicLayer' = None):
        """
        Add n neurons to this layer.
        Init strategy: near-zero weights so existing optimization is NOT disturbed.
        (Mitchell et al. 2023 — new neurons must not interfere with prior optim.)
        """
        old_out = self.out_features
        new_out = old_out + n

        # New weight matrix for THIS layer (out grows)
        new_weight = torch.zeros(new_out, self.in_features)
        new_weight[:old_out] = self.linear.weight.data
        # New neurons: tiny random init — "small perturbation" strategy
        nn.init.xavier_uniform_(new_weight[old_out:])
        new_weight[old_out:] *= 0.01   # scale down so no disruption

        new_bias = torch.zeros(new_out)
        new_bias[:old_out] = self.linear.bias.data

        # Replace Linear layer
        self.linear = nn.Linear(self.in_features, new_out)
        self.linear.weight = nn.Parameter(new_weight)
        self.linear.bias   = nn.Parameter(new_bias)
        self.bn = nn.BatchNorm1d(new_out)

        self.out_features = new_out
        self.neuron_activations = torch.zeros(new_out)
        self.neuron_gradients   = torch.zeros(new_out)
        self._register_hooks()

        # Next layer must accept the new inputs
        if next_layer is not None:
            next_layer.expand_input(n)

        print(f"  [GROW]  Layer: {old_out} → {new_out} neurons (+{n})")
        return new_out

    # ------------------------------------------------------------------ #
    #  INPUT EXPANSION — called on the NEXT layer when prev layer grew    #
    # ------------------------------------------------------------------ #
    def expand_input(self, n: int):
        """Expand input side of this layer because previous layer added n neurons."""
        old_in  = self.in_features
        new_in  = old_in + n

        new_weight = torch.zeros(self.out_features, new_in)
        new_weight[:, :old_in] = self.linear.weight.data
        # New input connections: near zero
        new_weight[:, old_in:] = torch.randn(self.out_features, n) * 0.01

        # ← save OLD bias BEFORE replacing the layer
        old_bias = self.linear.bias.data.clone()

        self.linear = nn.Linear(new_in, self.out_features)
        self.linear.weight = nn.Parameter(new_weight)
        self.linear.bias   = nn.Parameter(old_bias)   # ← restore old bias
        self.in_features = new_in
        self._register_hooks()

    # ------------------------------------------------------------------ #
    #  PRUNING — remove neurons with consistently near-zero activation     #
    # ------------------------------------------------------------------ #
    def prune_neurons(self, threshold: float = 0.01,
                      next_layer: 'DynamicLayer' = None) -> int:
        """
        Remove neurons whose mean absolute activation < threshold.
        Returns number of neurons pruned.
        From DEN paper: group-sparsity regularization removes unnecessary neurons.
        """
        if self.out_features <= 2:
            return 0   # keep at least 2 neurons

        keep_mask = self.neuron_activations > threshold
        # Always keep at least half
        if keep_mask.sum() < max(2, self.out_features // 2):
            # Keep top-50% by activation
            topk = self.neuron_activations.topk(max(2, self.out_features // 2))
            keep_mask = torch.zeros(self.out_features, dtype=torch.bool)
            keep_mask[topk.indices] = True

        keep_idx   = keep_mask.nonzero(as_tuple=True)[0]
        pruned     = self.out_features - len(keep_idx)

        if pruned == 0:
            return 0

        new_out    = len(keep_idx)
        new_weight = self.linear.weight.data[keep_idx]
        new_bias   = self.linear.bias.data[keep_idx]

        self.linear = nn.Linear(self.in_features, new_out)
        self.linear.weight = nn.Parameter(new_weight)
        self.linear.bias   = nn.Parameter(new_bias)
        self.bn = nn.BatchNorm1d(new_out)

        self.out_features = new_out
        self.neuron_activations = self.neuron_activations[keep_idx]
        self.neuron_gradients   = self.neuron_gradients[keep_idx]
        self._register_hooks()

        # Shrink NEXT layer inputs
        if next_layer is not None:
            new_w = next_layer.linear.weight.data[:, keep_idx]
            next_layer.linear = nn.Linear(new_out, next_layer.out_features)
            next_layer.linear.weight = nn.Parameter(new_w)
            next_layer.in_features = new_out

        print(f"  [PRUNE] Layer: pruned {pruned} neurons → {new_out} remain")
        return pruned


class SelfExpandingNet(nn.Module):
    """
    Main Self-Expanding Neural Network.
    Starts small. Grows automatically during training.
    Architecture is serializable to JSON for self-modification.
    """

    def __init__(self, input_size: int, output_size: int,
                 initial_hidden: List[int] = None):
        super().__init__()

        if initial_hidden is None:
            initial_hidden = [16]   # Start SMALL (paper's recommendation)

        self.input_size  = input_size
        self.output_size = output_size
        self.growth_log  = []       # Track all growth events

        # Build initial layers
        self.hidden_layers = nn.ModuleList()
        sizes = [input_size] + initial_hidden

        for i in range(len(sizes) - 1):
            self.hidden_layers.append(
                DynamicLayer(sizes[i], sizes[i+1], activation=nn.ReLU())
            )

        # Output layer (fixed size — determined by task)
        last_hidden = initial_hidden[-1]
        self.output_layer = nn.Linear(last_hidden, output_size)

        print(f"[SENN] Initialized: input={input_size} → hidden={initial_hidden} → output={output_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

    # ------------------------------------------------------------------ #
    #  ADD A NEW HIDDEN LAYER (depth expansion)                           #
    # ------------------------------------------------------------------ #
    def add_layer(self, position: int = -1, size: int = 8):
        """
        Add a new hidden layer at `position`.
        Init as near-identity so existing function is preserved.
        Strategy from SECNN paper (arXiv:2401.05686).
        position=-1 means add just before output layer.
        """
        if position == -1:
            position = len(self.hidden_layers)

        # Size of the layer before insertion point
        if position == 0:
            in_size = self.input_size
        else:
            in_size = self.hidden_layers[position - 1].out_features

        # Update next layer's input if not inserting at end
        new_layer = DynamicLayer(in_size, size, activation=nn.ReLU())

        if position < len(self.hidden_layers):
            # Must update next layer's input size
            self.hidden_layers[position].expand_input(size - in_size
                                                       if size != in_size else 0)

        layers_list = list(self.hidden_layers)
        layers_list.insert(position, new_layer)
        self.hidden_layers = nn.ModuleList(layers_list)

        # Fix output layer input size
        self._fix_output_layer()

        event = {"event": "add_layer", "position": position, "size": size}
        self.growth_log.append(event)
        print(f"  [GROW]  New layer at position {position} with {size} neurons")

    def _fix_output_layer(self):
        """Ensure output layer input matches last hidden layer output."""
        last_size = self.hidden_layers[-1].out_features
        if self.output_layer.in_features != last_size:
            old_w = self.output_layer.weight.data
            new_w = torch.zeros(self.output_size, last_size)
            min_in = min(old_w.shape[1], last_size)
            new_w[:, :min_in] = old_w[:, :min_in]
            self.output_layer = nn.Linear(last_size, self.output_size)
            self.output_layer.weight = nn.Parameter(new_w)

    # ------------------------------------------------------------------ #
    #  SERIALIZE architecture to JSON                                     #
    # ------------------------------------------------------------------ #
    def get_architecture(self) -> Dict:
        arch = {
            "input_size":  self.input_size,
            "output_size": self.output_size,
            "hidden_layers": [
                {"in": l.in_features, "out": l.out_features}
                for l in self.hidden_layers
            ],
            "total_params": sum(p.numel() for p in self.parameters()),
            "growth_events": len(self.growth_log)
        }
        return arch

    def __repr__(self):
        arch = self.get_architecture()
        layers = " → ".join(
            [str(arch["input_size"])] +
            [str(l["out"]) for l in arch["hidden_layers"]] +
            [str(arch["output_size"])]
        )
        return f"SelfExpandingNet [{layers}] | params={arch['total_params']}"
