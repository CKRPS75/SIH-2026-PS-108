from types import SimpleNamespace

import pytest

from app.services.reranker_training import (
    CheckpointMetrics,
    configure_last_n_layers_trainable,
    pairwise_ranknet_loss,
    select_best_checkpoint,
)
from scripts import train_standardwise_reranker as trainer


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.requires_grad = True
        self._count = count

    def numel(self) -> int:
        return self._count


class FakeLayer:
    def __init__(self, parameter: FakeParameter) -> None:
        self._parameter = parameter

    def parameters(self):
        return [self._parameter]


class FakeTorchModel:
    def __init__(self) -> None:
        self.layer_params = [FakeParameter(10) for _ in range(6)]
        self.classifier_param = FakeParameter(5)
        self.roberta = SimpleNamespace(
            encoder=SimpleNamespace(
                layer=[FakeLayer(parameter) for parameter in self.layer_params]
            )
        )

    def parameters(self):
        return [*self.layer_params, self.classifier_param]

    def named_parameters(self):
        for index, parameter in enumerate(self.layer_params):
            yield f"roberta.encoder.layer.{index}.weight", parameter
        yield "classifier.weight", self.classifier_param


def test_partial_unfreeze_keeps_only_last_n_layers_and_head_trainable() -> None:
    model = SimpleNamespace(model=FakeTorchModel())

    report = configure_last_n_layers_trainable(model, last_n=4)

    assert report["trainable_parameters"] == 45
    assert model.model.classifier_param.requires_grad
    assert [parameter.requires_grad for parameter in model.model.layer_params] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_ranknet_loss_orders_positive_and_negative_scores() -> None:
    assert pairwise_ranknet_loss([3.0], [0.0]) < pairwise_ranknet_loss([0.0], [3.0])


def test_ranknet_tensor_loss_matches_pairwise_direction() -> None:
    torch = pytest.importorskip("torch")

    good = trainer._ranknet_loss_tensor(torch.tensor([3.0]), torch.tensor([0.0]))
    bad = trainer._ranknet_loss_tensor(torch.tensor([0.0]), torch.tensor([3.0]))

    assert good.item() < bad.item()


def test_gradient_accumulation_steps_at_expected_interval() -> None:
    steps = [
        index
        for index in range(1, 8)
        if trainer._should_optimizer_step(
            index,
            total_batches=7,
            gradient_accumulation_steps=4,
        )
    ]

    assert steps == [4, 7]


def test_seed_is_deterministic_for_python_and_torch() -> None:
    torch = pytest.importorskip("torch")
    import random

    trainer._set_deterministic_seed(5108)
    first_random = random.random()
    first_tensor = torch.rand(3)
    trainer._set_deterministic_seed(5108)

    assert random.random() == first_random
    assert torch.equal(torch.rand(3), first_tensor)


def test_best_checkpoint_selection_uses_mrr_hit1_then_loss(tmp_path) -> None:
    first = CheckpointMetrics(
        epoch=1,
        checkpoint_path=tmp_path / "epoch-1",
        train_loss=0.2,
        validation_loss=0.4,
        validation_hit_at_1=0.8,
        validation_hit_at_3=1.0,
        validation_hit_at_5=1.0,
        validation_mrr=0.9,
    )
    second = CheckpointMetrics(
        epoch=2,
        checkpoint_path=tmp_path / "epoch-2",
        train_loss=0.2,
        validation_loss=0.3,
        validation_hit_at_1=0.8,
        validation_hit_at_3=1.0,
        validation_hit_at_5=1.0,
        validation_mrr=0.9,
    )

    assert select_best_checkpoint([first, second]) == second


def test_cpu_import_and_mixed_precision_mode_are_safe() -> None:
    assert trainer._mixed_precision_mode("cpu", enabled=True) == "off"
