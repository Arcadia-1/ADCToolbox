"""Held-out physical forward-model check of the oversampling example."""

from importlib import import_module


def test_inband_fit_improves_held_out_noise_shaping_capture():
    example = import_module(
        "adctoolbox.examples.10_oversampling.exp_o04_inband_weight_calibration")
    report, _ = example.run_comparison()
    metrics = report["metrics"]
    assert report["overload_samples"] == 0
    assert report["train_frequency_normalized"] != report["test_frequency_normalized"]
    assert metrics["In-band fit"]["sndr_dbc"] > metrics["Full-band fit"]["sndr_dbc"] + 15
    assert metrics["In-band fit"]["sndr_dbc"] > metrics["Analog-weight reference"]["sndr_dbc"] - 2
