"""The pure parts of the classifiers track: labels, splits, hygiene, metrics, calibration, loss, LLRD."""

import math
from pathlib import Path

import pytest

from stagetrain.core import Pipeline, kinds, load_config
from stagetrain.tracks.classifiers.labels import (
    MULTI,
    SINGLE,
    LabelSpace,
    class_weights,
    dedupe,
    infer_names,
    label_counts,
    remove_leaks,
    row_target,
    split_labels,
    stratified_split,
    stratum,
    text_key,
)

CONFIGS = sorted((Path(__file__).parent.parent / "configs" / "classifiers").glob("*.yaml"))


def test_stage_kinds_are_registered():
    assert set(kinds("classifiers")) == {
        "classifiers.prepare_data", "classifiers.mlm_adapt", "classifiers.finetune", "classifiers.pseudo_label",
        "classifiers.calibrate", "classifiers.eval", "classifiers.export", "classifiers.select",
        "classifiers.baseline", "classifiers.audit_labels",
    }


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_validate(path):
    pipeline = Pipeline(load_config(path))
    pipeline.validate()
    assert pipeline.config.track == "classifiers"


def test_there_is_a_shipped_config():
    assert CONFIGS


# ------------------------------------------------------------------ labels


def test_label_space_round_trip(tmp_path):
    space = LabelSpace(["neg", "pos"], SINGLE, class_weights=[1.0, 2.0], counts=[4, 2])
    assert space.index("pos") == 1 and space.index(" neg ") == 0 and space.index(1) == 1
    with pytest.raises(KeyError, match="unknown label 'meh'"):
        space.index("meh")
    with pytest.raises(KeyError):
        space.index(2)
    path = space.save(tmp_path / "m" / "labels.json")
    again = LabelSpace.load(path.parent)  # a model directory works as well as the file
    assert (again.names, again.problem_type, again.class_weights, again.counts) == (["neg", "pos"], SINGLE, [1.0, 2.0], [4, 2])
    with pytest.raises(ValueError, match="unique"):
        LabelSpace(["a", "a"])
    with pytest.raises(ValueError, match="problem_type"):
        LabelSpace(["a"], "regression")
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        LabelSpace.load(tmp_path)


def test_infer_names_and_split_labels():
    assert infer_names(["b", "a", "b"]) == ["a", "b"]
    assert infer_names([10, 2, "1"]) == ["1", "2", "10"]  # integer names sort as numbers
    assert split_labels("joy, love ,") == ["joy", "love"]
    assert split_labels("a|b", "|") == ["a", "b"]
    assert split_labels(["x", 3]) == ["x", 3] and split_labels(None) == [] and split_labels(4) == [4]


def test_dedupe_and_conflicts():
    rows = [
        {"text": "Hello  world", "label": 0},
        {"text": "hello world", "label": 0},  # same text after normalisation
        {"text": "mixed", "label": 0},
        {"text": "MIXED", "label": 1},  # the same text labelled two ways
        {"text": "  ", "label": 1},
        {"text": "pair", "text_pair": "a", "label": 0},
        {"text": "pair", "text_pair": "b", "label": 1},  # a different pair is a different example
    ]
    kept, stats = dedupe(rows)
    assert [r["text"] for r in kept] == ["Hello  world", "pair", "pair"]
    assert stats == {"duplicates": 1, "conflicts": 2, "empty": 1}
    kept, stats = dedupe(rows, drop_conflicts=False)
    assert [r["text"] for r in kept] == ["Hello  world", "mixed", "pair", "pair"] and stats["duplicates"] == 2
    multi, _ = dedupe([{"text": "t", "labels": [0, 1]}, {"text": "t", "labels": [0, 1]}, {"text": "u", "labels": []}])
    assert len(multi) == 2
    assert text_key({"text": " A\tb "}) == "a b"


def test_remove_leaks():
    train = [{"text": "a"}, {"text": "B"}, {"text": "c"}]
    kept, n = remove_leaks(train, [{"text": "b"}, {"text": "zzz"}])
    assert [r["text"] for r in kept] == ["a", "c"] and n == 1


def test_stratified_split_is_deterministic_and_keeps_every_class():
    rows = [{"text": f"t{i}", "label": 0 if i < 80 else 1 if i < 98 else 2} for i in range(100)] + [{"text": "solo", "label": 3}]
    key = lambda r: r["label"]  # noqa: E731
    rest, held = stratified_split(rows, 0.1, seed=7, key=key)
    assert len(rest) + len(held) == len(rows) and not {r["text"] for r in rest} & {r["text"] for r in held}
    assert label_counts(held, 4) == [8, 2, 1, 0]  # 10% of each class, at least one, never a class's only row
    assert label_counts(rest, 4) == [72, 16, 1, 1]
    assert stratified_split(rows, 0.1, seed=7, key=key) == (rest, held)
    assert stratified_split(rows, 0.1, seed=8, key=key)[1] != held
    assert stratified_split(rows, 0.0, seed=7, key=key) == (rows, [])
    with pytest.raises(ValueError, match="fraction"):
        stratified_split(rows, 1.0, seed=7, key=key)


def test_multi_label_stratum_is_the_rarest_label():
    counts = [50, 3, 10]
    assert stratum({"labels": [0, 2]}, counts) == 2 and stratum({"labels": [0, 1, 2]}, counts) == 1
    assert stratum({"labels": []}, counts) == -1 and stratum({"label": 4}, counts) == 4


def test_class_weights():
    assert class_weights([50, 25, 25], 100, SINGLE) == [round(100 / 150, 6), round(100 / 75, 6), round(100 / 75, 6)]
    assert class_weights([1, 99], 100, SINGLE, max_weight=5.0) == [5.0, round(100 / 198, 6)]
    assert class_weights([20, 0], 100, MULTI, max_weight=8.0) == [4.0, 8.0]  # negatives / positives


def test_row_target():
    assert row_target({"label": 1}, 4, SINGLE) == [0, 1, 0, 0]
    assert row_target({"label": 1}, 4, SINGLE, 0.2) == pytest.approx([0.05, 0.85, 0.05, 0.05])
    assert row_target({"labels": [0, 2]}, 3, MULTI) == [1, 0, 1]
    assert row_target({"labels": [0]}, 2, MULTI, 0.1) == pytest.approx([0.95, 0.05])
    assert row_target({"label": 0, "probs": [0.2, 0.8]}, 2, SINGLE, 0.3) == [0.2, 0.8]  # soft rows are not smoothed
    with pytest.raises(ValueError, match="3 probabilities for 2 labels"):
        row_target({"probs": [0.1, 0.2, 0.7]}, 2, SINGLE)


# ----------------------------------------------------------------- metrics

np = pytest.importorskip("numpy")


def test_report_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.metrics")
    from stagetrain.tracks.classifiers.metrics import classification_report, confusion_matrix

    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 5, 300) % 4
    y[y == 3] = 0  # class 3 never occurs in the targets, only in the predictions
    names = ["a", "b", "c", "d"]
    got = classification_report(p, y, names, SINGLE)
    assert got["metrics"]["accuracy"] == pytest.approx(sk.accuracy_score(y, p))
    for avg in ("macro", "micro", "weighted"):
        assert got["metrics"][f"{avg}_f1"] == pytest.approx(sk.f1_score(y, p, average=avg, zero_division=0))
    ref = sk.classification_report(y, p, labels=[0, 1, 2, 3], target_names=names, output_dict=True, zero_division=0)
    for name in names:
        for k in ("precision", "recall", "f1"):
            assert got["per_class"][name][k] == pytest.approx(ref[name]["f1-score" if k == "f1" else k])
        assert got["per_class"][name]["support"] == ref[name]["support"]
    assert confusion_matrix(p, y, 4) == sk.confusion_matrix(y, p, labels=[0, 1, 2, 3]).tolist()

    ym, pm = rng.random((200, 3)) < 0.3, rng.random((200, 3)) < 0.3
    got = classification_report(pm, ym, names[:3], MULTI)
    assert got["metrics"]["accuracy"] == pytest.approx(sk.accuracy_score(ym, pm))  # exact-match
    assert got["metrics"]["hamming_loss"] == pytest.approx(sk.hamming_loss(ym, pm))
    for avg in ("macro", "micro", "weighted"):
        assert got["metrics"][f"{avg}_f1"] == pytest.approx(sk.f1_score(ym, pm, average=avg, zero_division=0))


def test_probabilities_and_nll():
    from stagetrain.tracks.classifiers.metrics import nll, probabilities

    logits = np.array([[2.0, 0.0, -1.0], [0.0, 3.0, 0.0]])
    p = probabilities(logits, SINGLE)
    assert p.sum(-1) == pytest.approx([1, 1]) and p.argmax(-1).tolist() == [0, 1]
    assert probabilities(logits, SINGLE, 100.0) == pytest.approx(np.full((2, 3), 1 / 3), abs=0.02)
    assert nll(logits, [0, 1], SINGLE) == pytest.approx(-np.log(p[[0, 1], [0, 1]]).mean())
    assert probabilities(np.array([[0.0, 800.0, -800.0]]), MULTI).tolist() == [[0.5, 1.0, 0.0]]
    y = np.array([[1, 0, 1], [0, 1, 0]])
    pm = probabilities(logits, MULTI)
    assert nll(logits, y, MULTI) == pytest.approx(-(y * np.log(pm) + (1 - y) * np.log(1 - pm)).mean())
    assert math.isfinite(nll(np.array([[900.0, -900.0]]), np.array([[0, 1]]), MULTI))


def test_ece():
    from stagetrain.tracks.classifiers.metrics import ece

    # 10 predictions at confidence 0.8, 8 of them right: perfectly calibrated.
    probs = np.tile([0.8, 0.2], (10, 1))
    assert ece(probs, [0] * 8 + [1] * 2, SINGLE) == pytest.approx(0.0, abs=1e-12)
    assert ece(probs, [1] * 10, SINGLE) == pytest.approx(0.8)  # always wrong at 0.8
    assert ece(np.array([[1.0, 0.0]]), [0], SINGLE) == 0.0  # confidence 1.0 falls in the last bin
    # Multi-label: four decisions at confidence 0.9, three of them right.
    assert ece(np.array([[0.9, 0.1], [0.9, 0.1]]), np.array([[1, 0], [0, 0]]), MULTI) == pytest.approx(0.9 - 0.75)
    assert ece(np.zeros((0, 2)), [], SINGLE) == 0.0


def test_temperature_scaling_recovers_the_temperature():
    from stagetrain.tracks.classifiers.metrics import ece, fit_temperature, nll, probabilities

    rng = np.random.default_rng(1)
    clean = rng.normal(size=(4000, 5)) * 2.0
    y = np.array([rng.choice(5, p=row) for row in probabilities(clean, SINGLE)])
    for true_t in (0.4, 3.0):
        logits = clean * true_t  # an over- or under-confident model: the right temperature is true_t
        t = fit_temperature(logits, y, SINGLE)
        assert t == pytest.approx(true_t, rel=0.08)
        assert nll(logits, y, SINGLE, t) < nll(logits, y, SINGLE)
        assert ece(probabilities(logits, SINGLE, t), y, SINGLE) < ece(probabilities(logits, SINGLE), y, SINGLE)
    ym = rng.random((4000, 5)) < probabilities(clean, MULTI)
    assert fit_temperature(clean * 2.5, ym, MULTI) == pytest.approx(2.5, rel=0.08)


def test_thresholds_and_decide():
    from stagetrain.tracks.classifiers.metrics import decide, fit_thresholds

    probs = np.array([[0.30, 0.9, 0.1], [0.35, 0.8, 0.2], [0.10, 0.7, 0.3], [0.05, 0.2, 0.4]])
    y = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    thresholds = fit_thresholds(probs, y)
    # label 0: a cut between 0.10 and 0.30 separates it; label 1: 0.5 already does; label 2: no positive.
    assert 0.10 < thresholds[0] <= 0.30 and thresholds[1] == 0.5 and thresholds[2] == 0.5
    assert decide(probs, MULTI, thresholds).tolist() == y.astype(bool).tolist()
    assert decide(probs, MULTI).tolist()[0] == [False, True, False]
    assert decide(probs, SINGLE).tolist() == [1, 1, 1, 2]


def test_select_confident():
    from stagetrain.tracks.classifiers.post import select_confident

    probs = np.array([[0.95, 0.05], [0.6, 0.4], [0.02, 0.98], [0.91, 0.09], [0.08, 0.92]])
    assert select_confident(probs, SINGLE, 0.9, 0.4) == [(0, 0.95), (3, 0.91), (2, 0.98), (4, 0.92)]
    assert select_confident(probs, SINGLE, 0.9, 0.4, max_per_class=1) == [(0, 0.95), (2, 0.98)]
    assert select_confident(probs, SINGLE, 0.99, 0.4) == []
    kept = select_confident(np.array([[0.95, 0.03], [0.95, 0.6], [0.99, 0.99]]), MULTI, 0.9, 0.4)
    assert [i for i, _ in kept] == [2, 0] and kept[0][1] == pytest.approx(0.99)  # row 1 has an unsure label


# ------------------------------------------------------------------- torch

torch = pytest.importorskip("torch")


def test_loss_equals_torch_cross_entropy_on_hard_targets():
    import torch.nn.functional as F

    from stagetrain.tracks.classifiers.modeling import classification_loss

    g = torch.Generator().manual_seed(0)
    logits, y = torch.randn(16, 4, generator=g), torch.randint(0, 4, (16,), generator=g)
    onehot, w = F.one_hot(y, 4).float(), torch.tensor([1.0, 2.0, 0.5, 3.0])
    assert classification_loss(logits, onehot) == pytest.approx(F.cross_entropy(logits, y).item(), rel=1e-5)
    assert classification_loss(logits, onehot, class_weights=w) == pytest.approx(F.cross_entropy(logits, y, weight=w).item(), rel=1e-5)
    smooth = onehot * 0.9 + 0.1 / 4
    assert classification_loss(logits, smooth) == pytest.approx(F.cross_entropy(logits, y, label_smoothing=0.1).item(), rel=1e-5)
    soft = torch.softmax(torch.randn(16, 4, generator=g), -1)  # soft targets: the cross-entropy H(target, model)
    assert classification_loss(logits, soft) == pytest.approx(-(soft * F.log_softmax(logits, -1)).sum(-1).mean().item(), rel=1e-5)
    # Focal loss down-weights easy rows, so it is below the cross-entropy; gamma=0 is the cross-entropy.
    assert classification_loss(logits, onehot, focal_gamma=2.0) < classification_loss(logits, onehot)
    # A zero sample weight removes a row.
    sw = torch.ones(16)
    sw[8:] = 0
    assert classification_loss(logits, onehot, sw) == pytest.approx(F.cross_entropy(logits[:8], y[:8]).item(), rel=1e-5)


def test_multi_label_loss_equals_bce():
    import torch.nn.functional as F

    from stagetrain.tracks.classifiers.modeling import classification_loss

    g = torch.Generator().manual_seed(0)
    logits, y = torch.randn(8, 3, generator=g), (torch.rand(8, 3, generator=g) < 0.4).float()
    pw = torch.tensor([1.0, 4.0, 0.5])
    assert classification_loss(logits, y, problem_type=MULTI) == pytest.approx(F.binary_cross_entropy_with_logits(logits, y).item(), rel=1e-5)
    assert classification_loss(logits, y, problem_type=MULTI, class_weights=pw) == pytest.approx(
        F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw).item(), rel=1e-5)
    assert classification_loss(logits, y, problem_type=MULTI, focal_gamma=2.0) < classification_loss(logits, y, problem_type=MULTI)


def test_llrd_groups():
    from stagetrain.tracks.classifiers.modeling import layer_depths, llrd_groups

    def param(*shape, grad=True):
        return torch.nn.Parameter(torch.zeros(*shape), requires_grad=grad)

    named = [
        ("bert.embeddings.word_embeddings.weight", param(10, 4)),
        ("bert.embeddings.LayerNorm.weight", param(4)),
        ("bert.encoder.layer.0.attention.self.query.weight", param(4, 4)),
        ("bert.encoder.layer.0.attention.self.query.bias", param(4)),
        ("bert.encoder.layer.1.output.dense.weight", param(4, 4)),
        ("bert.pooler.dense.weight", param(4, 4)),
        ("classifier.weight", param(2, 4)),
        ("classifier.bias", param(2)),
        ("bert.encoder.layer.1.frozen.weight", param(4, 4, grad=False)),
    ]
    depths = layer_depths([n for n, _ in named], "bert")
    assert [depths[n] for n, _ in named[:8]] == [0, 0, 1, 1, 2, 3, None, None]

    groups = llrd_groups(named, lr=1e-3, decay=0.5, weight_decay=0.1, head_lr_multiplier=10, backbone_prefix="bert")
    by_param = {id(p): g for g in groups for p in g["params"]}
    lr = {n: by_param[id(p)]["lr"] for n, p in named[:8]}
    wd = {n: by_param[id(p)]["weight_decay"] for n, p in named[:8]}
    assert lr["bert.pooler.dense.weight"] == pytest.approx(1e-3)  # the top of the backbone trains at lr
    assert lr["bert.encoder.layer.1.output.dense.weight"] == pytest.approx(5e-4)
    assert lr["bert.encoder.layer.0.attention.self.query.weight"] == pytest.approx(2.5e-4)
    assert lr["bert.embeddings.word_embeddings.weight"] == pytest.approx(1.25e-4)
    assert lr["classifier.weight"] == pytest.approx(1e-2)
    assert wd["classifier.weight"] == 0.1 and wd["classifier.bias"] == 0.0
    assert wd["bert.embeddings.LayerNorm.weight"] == 0.0 and wd["bert.encoder.layer.0.attention.self.query.bias"] == 0.0
    assert id(named[-1][1]) not in by_param  # frozen parameters are not optimised
    assert sum(len(g["params"]) for g in groups) == 8
    torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups])  # valid optimizer groups

    # ModernBERT-style names: `model.layers.N.`, head outside the `model.` prefix.
    modern = ["model.embeddings.tok_embeddings.weight", "model.layers.0.attn.Wqkv.weight", "model.layers.21.mlp.Wo.weight",
              "model.final_norm.weight", "head.dense.weight", "classifier.weight"]
    assert list(layer_depths(modern, "model").values()) == [0, 1, 22, 23, None, None]
    flat = llrd_groups(named, lr=1e-3, decay=1.0, weight_decay=0.0)
    assert len(flat) == 1 and flat[0]["lr"] == 1e-3


# ------------------------------------------------- what the research added


def test_logit_adjusted_loss_adds_the_prior_in_training_only():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    from stagetrain.tracks.classifiers.modeling import classification_loss, log_prior

    bias = log_prior([80, 10, 10], tau=1.0)
    assert bias == pytest.approx([math.log(0.8), math.log(0.1), math.log(0.1)])
    assert log_prior([80, 10, 10], tau=2.0) == pytest.approx([2 * b for b in bias])
    assert log_prior([9, 1, 0])[2] == pytest.approx(math.log(0.1))  # an unseen class gets the rarest seen frequency
    with pytest.raises(ValueError, match="no class counts"):
        log_prior([])

    torch.manual_seed(0)
    logits, y = torch.randn(6, 3), torch.tensor([0, 0, 0, 1, 2, 0])
    onehot = F.one_hot(y, 3).float()
    adjusted = classification_loss(logits, onehot, logit_bias=torch.tensor(bias))
    assert adjusted.item() == pytest.approx(F.cross_entropy(logits + torch.tensor(bias), y).item(), rel=1e-6)
    # With equal raw logits the rare classes cost more, so training pushes their logits up.
    flat = torch.zeros(1, 3)
    rare = classification_loss(flat, torch.tensor([[0.0, 1.0, 0.0]]), logit_bias=torch.tensor(bias))
    common = classification_loss(flat, torch.tensor([[1.0, 0.0, 0.0]]), logit_bias=torch.tensor(bias))
    assert rare.item() > common.item()
    with pytest.raises(ValueError, match="single-label"):
        classification_loss(torch.zeros(2, 3), torch.zeros(2, 3), problem_type=MULTI, logit_bias=torch.tensor(bias))


def test_headline_matches_the_full_report():
    np = pytest.importorskip("numpy")
    from stagetrain.tracks.classifiers.metrics import classification_report, headline

    rng = np.random.default_rng(0)
    y, pred = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    full = classification_report(pred, y, list("abcd"), SINGLE)["metrics"]
    assert headline(pred, y, SINGLE, 4) == pytest.approx((full["accuracy"], full["macro_f1"]))
    ym, pm = rng.random((200, 5)) < 0.3, rng.random((200, 5)) < 0.3
    full = classification_report(pm, ym, list("abcde"), MULTI)["metrics"]
    assert headline(pm, ym, MULTI, 5) == pytest.approx((full["accuracy"], full["macro_f1"]))
    assert headline(np.array([0, 0]), np.array([0, 0]), SINGLE, 3) == (1.0, 1.0)  # classes that never occur are left out


def test_bootstrap_intervals_and_paired_difference():
    np = pytest.importorskip("numpy")
    from stagetrain.tracks.classifiers.metrics import bootstrap_intervals, headline, paired_bootstrap

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 600)
    base = np.where(rng.random(600) < 0.8, y, (y + 1) % 3)  # right four times in five
    better = np.where((base != y) & (rng.random(600) < 0.25), y, base)  # the same answers, a quarter of the errors fixed
    accuracy, macro = headline(base, y, SINGLE, 3)
    ci = bootstrap_intervals(base, y, SINGLE, 3, samples=400, seed=0)
    assert ci["accuracy"][0] < accuracy < ci["accuracy"][1] and ci["macro_f1"][0] < macro < ci["macro_f1"][1]
    assert 0.04 < ci["accuracy"][1] - ci["accuracy"][0] < 0.09  # about +-3 points at 80 % on 600 rows
    assert bootstrap_intervals(base, y, SINGLE, 3, samples=400, seed=0) == ci  # the same seed, the same draw

    diff = paired_bootstrap(better, base, y, SINGLE, 3, samples=400, seed=0)
    assert diff["accuracy"]["delta"] == pytest.approx(headline(better, y, SINGLE, 3)[0] - accuracy)
    assert diff["accuracy"]["ci_low"] > 0 and diff["macro_f1"]["ci_low"] > 0  # better beyond chance...
    own = bootstrap_intervals(better, y, SINGLE, 3, samples=400, seed=0)
    assert own["accuracy"][0] < ci["accuracy"][1]  # ...although the two models' own intervals overlap
    same = paired_bootstrap(base, base, y, SINGLE, 3, samples=100)
    assert same["macro_f1"] == {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_rare_labels_share_one_threshold():
    np = pytest.importorskip("numpy")
    from stagetrain.tracks.classifiers.metrics import fit_thresholds

    rng = np.random.default_rng(2)
    y = np.zeros((200, 3), dtype=int)
    y[:100, 0] = 1  # a common label: 100 positives
    y[:3, 1] = 1  # a rare one: 3 positives
    probs = np.clip(0.2 + 0.25 * y + rng.normal(0, 0.05, y.shape), 0.01, 0.99)  # positives sit near 0.45
    own = fit_thresholds(probs, y)
    assert own[2] == 0.5 and own[0] < 0.5 and own[1] < 0.5  # every label with a positive gets its own cut
    shared = fit_thresholds(probs, y, min_positives=10)
    assert shared[0] == own[0] and shared[1] == shared[2] and shared[1] < 0.5  # the rare labels take the common cut
    assert fit_thresholds(probs, np.zeros_like(y), min_positives=10) == [0.5, 0.5, 0.5]


def test_row_losses_and_probability_nll():
    np = pytest.importorskip("numpy")
    from stagetrain.tracks.classifiers.metrics import (
        nll,
        nll_of_probabilities,
        probabilities,
        row_losses,
        targets_array,
    )

    logits = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0]])
    y = np.array([0, 1, 2])
    losses = row_losses(logits, y, SINGLE)
    assert losses[2] > 3.9 and losses[0] == pytest.approx(losses[1]) and losses.mean() == pytest.approx(nll(logits, y, SINGLE))
    assert nll_of_probabilities(probabilities(logits, SINGLE), y, SINGLE) == pytest.approx(nll(logits, y, SINGLE))
    ym = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    assert row_losses(logits, ym, MULTI).mean() == pytest.approx(nll(logits, ym, MULTI))
    assert nll_of_probabilities(probabilities(logits, MULTI), ym, MULTI) == pytest.approx(nll(logits, ym, MULTI), rel=1e-6)
    rows = [{"labels": [0]}, {"labels": [1, 2]}, {"labels": []}]
    assert targets_array(rows, 3, MULTI).tolist() == ym.tolist()
    assert targets_array([{"label": 2}, {"label": 0}], 3, SINGLE).tolist() == [2, 0]


def test_folds_and_suspect_ranking():
    from stagetrain.tracks.classifiers.post import rank_suspects, row_id
    from stagetrain.tracks.classifiers.train import fold_of

    rows = [{"text": f"text number {i}"} for i in range(300)]
    folds = [fold_of(r, 5) for r in rows]
    assert set(folds) == {0, 1, 2, 3, 4} and min(folds.count(f) for f in range(5)) > 35
    assert folds == [fold_of(r, 5) for r in rows]  # the text decides, nothing else
    assert fold_of({"text": "Same  Text"}, 5) == fold_of({"text": "same text"}, 5)

    oof = [
        {"text": "fine", "label": 0, "loss": 0.1, "probs": [0.9, 0.1], "fold": 0},
        {"text": "wrong label", "label": 0, "loss": 3.0, "probs": [0.05, 0.95], "fold": 1},
        {"text": "wrong label", "label": 0, "loss": 2.0, "probs": [0.15, 0.85], "fold": 1},  # a second seed's run
        {"text": "multi", "labels": [0], "loss": 1.0, "probs": [0.9, 0.7], "fold": 2},
    ]
    ranked = rank_suspects(oof)
    assert [r["text"] for r in ranked] == ["wrong label", "multi", "fine"]
    top = ranked[0]
    assert top["loss"] == 2.5 and top["runs"] == 2 and top["probs"] == pytest.approx([0.1, 0.9])
    assert top["predicted"] == 1 and top["disagrees"] and top["id"] == row_id({"text": "WRONG label"})
    assert ranked[1]["predicted"] == [0, 1] and ranked[1]["disagrees"] and not ranked[2]["disagrees"]


@pytest.mark.parametrize("params, message", [
    ({"data": "pool.jsonl"}, "needs a teacher"),
    ({"data": "pool.jsonl", "models": ["a"], "model": "b"}, "models, or group, or a single model"),
    ({"data": "pool.jsonl", "model": "a", "ensemble": "winner"}, "needs group"),
    ({"data": "pool.jsonl", "group": "g", "ensemble": "winner"}, "needs model"),
    ({"data": "pool.jsonl", "models": ["a", "b"], "calibration": "c.json"}, "single teacher"),
])
def test_pseudo_label_teacher_parameters(params, message):
    from pydantic import ValidationError

    from stagetrain.tracks.classifiers.post import AuditParams, PseudoLabelParams

    with pytest.raises(ValidationError, match=message):
        PseudoLabelParams(**params)
    assert PseudoLabelParams(data="p.jsonl", models=["a", "b"]).ensemble == "all"
    with pytest.raises(ValidationError, match="needs train_data"):
        AuditParams(group="cv", labels="l.json", drop_fraction=0.1)

