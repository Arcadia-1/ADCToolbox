"""Public Dither contracts: observables, causality, state and validation."""
import importlib
import json
import subprocess
import sys

import numpy as np
import pytest

from adctoolbox import DitherGainTracker, DitherHDCTracker, apply_dither_gain


def capture(size=12288):
    example = importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed')
    x = .4*np.sin(np.arange(size)*.137)
    c, _ = example.Plant(seed=81, noise=0).convert(x)
    args = (2*c['r'], c['p'], c['q'], c['s'], c['rails'])
    return c, args


def test_public_imports_do_not_load_examples_or_optional_numba():
    result = subprocess.run([sys.executable, '-c', """
import sys
import adctoolbox as adc
from adctoolbox import calibration
for name in ('DitherGainTracker', 'DitherHDCTracker', 'apply_dither_gain'):
    assert name in adc.__all__ and name in calibration.__all__
    assert getattr(adc, name) is getattr(calibration, name)
assert not any(name.startswith('adctoolbox.examples') for name in sys.modules)
assert 'numba' not in sys.modules
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gain_update_and_frozen_application_have_distinct_causal_roles():
    pn = np.tile([-1., 1.], 64)
    signal = np.repeat(np.linspace(-.1, .1, 64), 2)
    coarse = np.full(len(pn), .25)
    backend = 3.8*(signal+.0625*pn)
    codes = np.floor((backend+1)*2048)
    tracker = DitherGainTracker(learning_rate=1.)
    before = tracker.apply(coarse, backend, pn)
    assert before['gain_used'] == 4.
    update = tracker.update(backend, pn, codes)
    assert update['gain_next'] == pytest.approx(3.8)
    actual = tracker.apply(coarse, backend, pn)
    np.testing.assert_allclose(actual['calibrated_signal'], coarse+signal, atol=1e-15)
    np.testing.assert_array_equal(actual['calibrated_signal'], apply_dither_gain(
        coarse, backend, pn, gain_estimate=tracker.gain_estimate,
        probe_amplitude=.0625)['calibrated_signal'])
    saved = tracker.gain_estimate
    with pytest.raises(ValueError):
        tracker.update(backend, pn, np.full(len(pn), np.nan))
    assert tracker.gain_estimate == saved
    tracker.reset()
    assert tracker.gain_estimate == 4. and tracker.updates == 0


@pytest.mark.parametrize('fixed', [False, True])
def test_hdc_public_matches_existing_kernel_and_arbitrary_block_partitions(fixed):
    example = importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed')
    c, args = capture()
    kernel = example.Tracker(fixed=fixed, window_log2=10, controller_latency=17)
    expected = kernel.process(c)
    one = DitherHDCTracker(fixed=fixed, window_log2=10, controller_latency=17)
    split = DitherHDCTracker(fixed=fixed, window_log2=10, controller_latency=17)
    actual = one.process(*args)
    edges = (0, 1, 1024, 1041, 1042, 4099, len(args[0]))
    parts = [split.process(*(a[start:stop] for a in args))['calibrated_signal']
             for start, stop in zip(edges, edges[1:])]
    np.testing.assert_array_equal(actual['calibrated_signal'], expected)
    np.testing.assert_array_equal(np.concatenate(parts), expected)
    assert split.coefficients == one.coefficients
    assert actual['updates'] == kernel.updates
    assert actual['held_updates'] == kernel.held


def test_frozen_hdc_matches_independent_recurrence_and_resets_history():
    _, args = capture(257)
    initial = dict(linear=1.03, cubic=-.12, memory=.06, dac_mismatch=.01)
    tracker = DitherHDCTracker(fixed=False, initial_coefficients=initial)
    expected = []
    previous = 0.
    for r, p, q, s in zip(*args[:4]):
        linear = initial['linear']*r/2
        v = linear-initial['cubic']*linear**3-initial['memory']*previous
        expected.append(q/2+v-p.sum()/15+initial['dac_mismatch']*s*q/2)
        previous = v
    output = tracker.process(*args, learn=False)
    np.testing.assert_allclose(output['calibrated_signal'], expected, atol=1e-15)
    assert output['coefficients'] == initial
    assert output['updates'] == 0 and not output['pending_update']
    # Public coefficient dictionaries are detached and directly JSON round-trip.
    restored = DitherHDCTracker(fixed=False, initial_coefficients=json.loads(json.dumps(tracker.coefficients)))
    tracker.coefficients['linear'] = 9
    tracker.reset()
    np.testing.assert_array_equal(tracker.process(*args, learn=False)['calibrated_signal'],
                                  restored.process(*args, learn=False)['calibrated_signal'])


def test_freeze_cancels_pending_update_and_discards_partial_learning_window():
    # Balanced PN with a zero residue gives a well-excited gain-only window.
    import itertools
    p = np.tile(list(itertools.product((-1, 1), repeat=3)), (512, 1))
    # Keep exact current-PN moments without a correlated periodic lag reference.
    np.random.default_rng(91).shuffle(p, axis=0)
    args = (2*1.02*p.sum(axis=1)/15, p, np.zeros(len(p)),
            np.ones(len(p)), np.zeros(len(p), dtype=bool))
    tracker = DitherHDCTracker(fixed=False, window_log2=12, controller_latency=17)
    learned = tracker.process(*args)
    assert learned['pending_update'] and learned['updates'] == 1
    active = tracker.coefficients
    frozen = tracker.process(*(a[:64] for a in args), learn=False)
    assert not frozen['pending_update'] and tracker.coefficients == active
    tracker.process(*(a[:100] for a in args), learn=True)
    tracker.process(*(a[:100] for a in args), learn=False)
    resumed = tracker.process(*(a[:3996] for a in args), learn=True)
    assert resumed['updates'] == 1 and tracker.coefficients == active


@pytest.mark.parametrize('index,bad', [
    (0, np.full(12, np.nan)), (0, np.ones((12, 1))), (0, np.full(12, 1.1)),
    (1, np.ones((12, 2))), (1, np.zeros((12, 3))),
    (2, np.full(12, 2)), (3, np.zeros(12)), (4, np.zeros(12)),
])
def test_bad_capture_rejected_before_any_state_changes(index, bad):
    _, args = capture(12)
    tracker = DitherHDCTracker()
    invalid = list(args)
    invalid[index] = bad
    with pytest.raises(ValueError):
        tracker.process(*invalid)
    fresh = DitherHDCTracker()
    np.testing.assert_array_equal(tracker.process(*args)['calibrated_signal'],
                                  fresh.process(*args)['calibrated_signal'])


@pytest.mark.parametrize('kwargs', [
    dict(fixed='yes'), dict(window_log2=10.5), dict(accumulator_bits=0),
    dict(controller_latency=True), dict(fraction=8), dict(dither_peak=0),
    dict(initial_coefficients={'linear': 1}),
    dict(initial_coefficients=dict(linear=1, cubic=0, memory=1, dac_mismatch=0)),
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DitherHDCTracker(**kwargs)
