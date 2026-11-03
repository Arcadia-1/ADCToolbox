"""Small-input controls must distinguish bias, freezing and sample variance."""
import importlib
import numpy as np
import pytest

hdc = importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed')
small = importlib.import_module('adctoolbox.examples.11_calibration.hdc_small_signal')


def test_balanced_quantization_bias_is_an_expectation_not_sampling_error():
    rows = small.balanced_scan(offsets=[0., .07, .137], amplitudes=[0., 1e-5], phase_count=32)
    for row in rows:
        assert row['headroom'] > 0
        assert row['cubic_error'] == pytest.approx(row['quantization_cubic_bias'], abs=1e-13)
        if row['backend_bits'] == 0:
            assert row['cubic_error'] == pytest.approx(0, abs=1e-13)
            assert row['gain_error'] == pytest.approx(row['eta_gain_bias'], abs=1e-13)
    assert max(abs(r['cubic_error']) for r in rows if r['backend_bits']==13) > .005
    assert max(abs(r['cubic_error']) for r in rows if r['backend_bits']==24) < 1e-5
    # Repeated DC phase rows cannot average away a deterministic PN correlation.
    repeated = small.balanced_scan(offsets=[.137], amplitudes=[0.], phase_count=512)
    original = [r for r in rows if r['offset']==.137 and r['amplitude']==0]
    np.testing.assert_allclose([r['estimated_cubic'] for r in repeated],
                               [r['estimated_cubic'] for r in original], atol=1e-13)


def test_snapshot_discards_pending_update_and_preserves_active_coefficients():
    tracker = hdc.Tracker(window_log2=12)
    tracker.coefficients[0] += 1234
    tracker.pending = tracker.coefficients+500
    tracker.commit_at, tracker.count, tracker.samples = 512, 111, 0
    tracker.sums[:] = 99
    frozen = small.branch_tracker(tracker)
    assert frozen.pending is None and frozen.count == frozen.samples == 0
    assert not np.any(frozen.sums)
    np.testing.assert_array_equal(frozen.coefficients, tracker.coefficients)
    capture, _ = hdc.Plant(55).convert(np.full(8192, .137))
    before = frozen.values().copy()
    frozen.process(capture, learn=False)
    np.testing.assert_array_equal(frozen.values(), before)
    assert frozen.updates == 0 and frozen.events == []


def test_dc_score_has_bias_and_noise_but_no_sndr():
    result = small.score(np.full(1024, .1002), np.full(1024, .1), [])
    assert result['sndr_db'] is result['sfdr_db'] is None
    assert result['mean_error'] == pytest.approx(.0002)


def test_streaming_methods_share_capture_and_frozen_coefficients():
    plant, trackers, _ = small.train(66, 0., 16, 12, .2)
    rows, result = small.stream_case(plant, trackers, 'small', .137, 1e-5, 16, 12, 10, 66, 0., .2)
    assert result['overloads'] == result['heldout_overloads'] == 0
    assert len(rows) == 5
    assert result['methods']['frozen_fixed']['parameters'] == trackers['fixed'].physical()
    assert all(row['saturations']==row['accumulator_saturations']==row['controller_saturations']==0 for row in rows)


def test_small_signal_cli_refuses_existing_outputs(tmp_path):
    (tmp_path/'keep.txt').write_text('keep')
    with pytest.raises(SystemExit): small.main(['--output', str(tmp_path)])
    assert (tmp_path/'keep.txt').read_text() == 'keep'
