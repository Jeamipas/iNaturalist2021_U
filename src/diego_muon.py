"""Muon with the same coupled L2 rule as Diego's Adam baseline.

Only internal convolution/linear weights use Muon. The classifier, biases and
normalization parameters use Adam. Learning rates describe fixed recipes, not
an optimizer-wide hyperparameter search. Sources and limitations live in the
single project notebook.
"""
import torch

from .full_experiment import MatrixMuon


class CoupledMuon(MatrixMuon):
    @torch.no_grad()
    def step(self, closure=None):
        # Diego applies L2 to tensors with ndim > 1 before the optimizer update.
        # Muon receives the same regularized gradient, rather than introducing
        # decoupled decay as a second experimental change.
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None and group.get("weight_decay", 0):
                    p.grad.add_(p, alpha=group["weight_decay"])
        return super().step(closure)


class DiegoMuon(torch.optim.Optimizer):
    def __init__(self, model, lr=0.001, muon_lr=0.02, weight_decay=5e-5, factor_lr_backbone=1.0):
        classifier = model.fc if hasattr(model, "fc") else model.classifier
        head_ids = {id(p) for p in classifier.parameters()}
        matrices, auxiliary = [], {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("weight") and p.ndim >= 2 and id(p) not in head_ids:
                matrices.append(p)
            else:
                # ConvNeXt layer_scale is [C,1,1], but is a learned vector.
                # It uses auxiliary Adam without L2, like normalization/bias.
                no_decay = p.ndim <= 1 or name.endswith("layer_scale")
                auxiliary.setdefault((id(p) not in head_ids, no_decay), []).append(p)
        if not matrices or not auxiliary:
            raise ValueError("Muon needs backbone matrices and auxiliary Adam parameters")
        matrix_optimizer = CoupledMuon(matrices, lr=muon_lr * factor_lr_backbone)
        for group in matrix_optimizer.param_groups:
            group["weight_decay"] = weight_decay
        adam_groups = [{"params": ps, "lr": lr * (factor_lr_backbone if backbone else 1.0),
                        "weight_decay": 0.0 if no_decay else weight_decay}
                       for (backbone, no_decay), ps in auxiliary.items()]
        self.children = [matrix_optimizer, torch.optim.Adam(adam_groups, lr=lr)]
        super().__init__([g for child in self.children for g in child.param_groups], {})

    def step(self, closure=None):
        for child in self.children:
            child.step()

    def state_dict(self):
        return {"children": [child.state_dict() for child in self.children]}

    def load_state_dict(self, state):
        if len(state["children"]) != len(self.children):
            raise ValueError("Optimizer checkpoint group mismatch")
        for child, saved in zip(self.children, state["children"]):
            child.load_state_dict(saved)
        self.param_groups = [g for child in self.children for g in child.param_groups]
