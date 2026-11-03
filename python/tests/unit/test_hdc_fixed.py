"""Independent HDC moment identities, fixed arithmetic, and causal scheduling."""
import importlib
import itertools
import numpy as np
import pytest

hdc = importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed')
ex = importlib.import_module('adctoolbox.examples.11_calibration.exp_cal08_hdc_fixed')
controls = importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed_controls')


def balanced_capture(g=1.02, k=-.24, size=4096):
    signs = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=np.int64)
    p = np.tile(signs, (size//8, 1))
    u = np.repeat(np.linspace(-.25, .25, size//8), 8)
    d = p.sum(axis=1)/15
    r = g*(u+d)+k*(u+d)**3
    c = dict(r=r, p=p, pp=np.zeros(size,dtype=np.int64), s=np.ones(size,dtype=np.int64),
             q=np.zeros(size,dtype=np.int64), rail=False)
    return c, u


def sliced(c, start, stop):
    return {k:(v[start:stop] if isinstance(v,np.ndarray) else v) for k,v in c.items()}


def test_three_pn_moments_and_explicit_eta_approximation():
    c, u = balanced_capture()
    a, g, k = 1/15, 1.02, -.24
    ss = c['r']-a*c['p'].sum(axis=1)
    gamma1 = np.mean(ss*c['p'][:,0])/a
    gamma3 = np.mean(ss*np.prod(c['p'],axis=1))/(6*a**3)
    assert gamma3 == pytest.approx(k, abs=1e-13)
    assert gamma1 == pytest.approx(g-1+(7*a*a+3*np.mean(u*u))*k, abs=1e-13)
    tracker = hdc.Tracker(False, window_log2=12)
    tracker.process(c)
    estimated_g = 1/tracker.pending[0]
    # The published eta approximation produces this deterministic gain bias.
    assert estimated_g-g == pytest.approx(3*k*(np.mean(u*u)-np.mean(ss*ss)), abs=1e-13)
    assert abs(estimated_g-g) > 1e-5


def test_signed_restoring_divider_against_integer_reference():
    rng = np.random.default_rng(918)
    for width in (8, 28, 56):
        for _ in range(80):
            n = int(rng.integers(-(1 << (width-1)), 1 << (width-1)))
            d = int(rng.integers(1, 1 << min(width-1,28)))
            for divisor in (d, -d):
                assert hdc.serial_divide_round(n, divisor, width) == hdc.divide_round(n, divisor)
    assert hdc.serial_divide_round(-3, 2, 8) == -2
    with pytest.raises(OverflowError):
        hdc.serial_divide_round(256, 3, 8)
    with pytest.raises(ZeroDivisionError):
        hdc.serial_divide_round(1, 0, 8)


def test_update_latency_and_stream_partition_invariance():
    c, _ = balanced_capture(k=0)
    c = {k:(np.tile(v, (3,1)) if v.ndim == 2 else np.tile(v,3)) if isinstance(v,np.ndarray) else v for k,v in c.items()}
    one, split = hdc.Tracker(window_log2=12), hdc.Tracker(window_log2=12)
    expected = one.process(c)
    first = split.process(sliced(c,0,4096))
    np.testing.assert_array_equal(split.values(), [1,0,0,0])
    assert split.pending is not None
    second = split.process(sliced(c,4096,4608))
    np.testing.assert_array_equal(split.values(), [1,0,0,0])
    third = split.process(sliced(c,4608,4609))
    assert split.values()[0] == pytest.approx(1/1.02, abs=2e-6)
    fourth = split.process(sliced(c,4609,len(c['r'])))
    np.testing.assert_array_equal(expected, np.concatenate([first,second,third,fourth]))
    assert split.events == one.events
    np.testing.assert_array_equal(split.sums,one.sums)
    assert split.saturations == split.accumulator_saturations == split.controller_saturations == 0


def test_accumulator_overflow_and_backend_rail_hold():
    c, _ = balanced_capture(k=0)
    tracker = hdc.Tracker(window_log2=12, accumulator_bits=16)
    tracker.process(c)
    assert tracker.held == 1 and tracker.accumulator_saturations > 0
    np.testing.assert_array_equal(tracker.values(),[1,0,0,0])
    other = hdc.Tracker(window_log2=12)
    other.process({**c,'rail':True})
    assert other.held == 1 and other.pending is None
    assert other.events[0]['rails_or_overflow']


def test_rail_flag_only_invalidates_its_acquisition_window():
    c, _ = balanced_capture(k=0,size=8192)
    rails = np.zeros(8192,dtype=bool); rails[4000] = True
    c.update(rail=True,rails=rails)
    tracker = hdc.Tracker(window_log2=12)
    tracker.process(c)
    assert tracker.held == 1 and tracker.updates == 1
    assert [e['rails_or_overflow'] for e in tracker.events] == [True,False]


def test_three_equal_probes_physical_peak_and_stream_continuity():
    a, b = hdc.Plant(197), hdc.Plant(197)
    x = .8*np.sin(np.arange(16384)*.861)
    one, d = a.convert(x)
    left, _ = b.convert(x[:5555]); right, _ = b.convert(x[5555:])
    for key in ('r','p','pp','s','q'):
        np.testing.assert_array_equal(one[key],np.concatenate([left[key],right[key]]))
    assert d['probe_peak'] == pytest.approx(.2)
    assert set(one['p'].sum(axis=1)) == {-3,-1,1,3}
    assert d['overloads'] == 0 and d['headroom'] > 0


def test_exact_oracle_separates_cubic_inverse_residual():
    x, bins = ex.waveform(np.arange(65536),32768,'sine')
    c, _ = hdc.Plant(918, bits=24, noise=0).convert(x)
    exact = ex.oracle(c,hdc.FINAL,True)
    approximate = ex.oracle(c,hdc.FINAL)
    assert np.max(abs(exact[64:]-x[64:])) < 5e-8
    assert np.std(approximate[64:]-x[64:]) > 1e-4


def test_resource_inventory_is_explicit_and_bounded():
    resources = hdc.resource_budget()
    assert resources['total_multipliers'] == 7
    assert resources['correlation_accumulators'] == 5
    assert resources['total_register_bits'] == sum(resources['registers'].values())
    assert resources['divider_width'] == 56
    assert resources['division_iterations_per_update'] < resources['controller_latency_samples']


def test_joint_cartesian_controls_separate_moment_bias_from_sample_error():
    for row in controls.balanced_moments():
        if row['backend_bits'] == 0:
            assert row['cubic_error'] == pytest.approx(0,abs=1e-13)
            assert row['gain_error'] == pytest.approx(row['eta_gain_bias_prediction'],abs=1e-13)


def test_dc_overload_at_unchanged_probe_peak_is_detected():
    rows, bounds = controls.headroom_controls()
    dc = next(r for r in rows if r['signal'] == 'dc_pos')
    assert dc['probe_peak'] == pytest.approx(.2)
    assert dc['headroom'] < 0 and dc['overloads'] > 0
    assert dc['held'] == 1 and dc['accepted'] == 0
    reduced = next(r for r in rows if r['signal'] == 'dc_pos_derated')
    assert reduced['headroom'] > .01 and reduced['overloads'] == 0
    assert .78 < bounds[1]['maximum_input_peak_six_sigma'] < .8


def test_cli_refuses_overwrite(tmp_path):
    (tmp_path/'existing.txt').write_text('retain')
    with pytest.raises(SystemExit):
        ex.main(['--output',str(tmp_path)])
    assert (tmp_path/'existing.txt').read_text() == 'retain'


@pytest.mark.parametrize('fraction', [8, 9, 10])
def test_reject_unrepresentable_cubic_denominator_before_capture(fraction):
    with pytest.raises(ValueError, match='moment denominator'):
        hdc.Tracker(fraction=fraction, bits=fraction+4)


@pytest.mark.parametrize('fraction', [11, 16, 24, 26])
def test_supported_denominator_survives_a_balanced_update(fraction):
    c, _ = balanced_capture()
    tracker = hdc.Tracker(fraction=fraction, bits=fraction+4, window_log2=12)
    tracker.process(c)
    assert tracker.updates+tracker.held == 1
    assert np.all(np.isfinite(tracker.values()))


def test_denominator_validation_depends_on_probe_scale_and_arithmetic():
    with pytest.raises(ValueError, match='moment denominator'):
        hdc.Tracker(dither_peak=.001)
    # Float mode has no staged fixed-point underflow at these settings.
    assert hdc.Tracker(False, fraction=8, bits=12).a > 0
    assert hdc.Tracker(fraction=10, bits=14, dither_peak=.4).a > 0


@pytest.mark.parametrize('peak', [0., -1., 1., np.nan, np.inf])
def test_invalid_probe_scale_rejected(peak):
    with pytest.raises(ValueError): hdc.Tracker(dither_peak=peak)
    with pytest.raises(ValueError): hdc.Plant(peak=peak)


def test_custom_probe_scale_shared_by_plant_tracker_and_oracle():
    peak = .15
    x = .3*np.sin(np.arange(4096)*.137)
    c, d = hdc.Plant(9, peak=peak, bits=24, noise=0).convert(x)
    assert d['probe_peak'] == pytest.approx(peak)
    with pytest.raises(ValueError, match='injection scale'):
        hdc.Tracker().process(c)
    tracker = hdc.Tracker(dither_peak=peak, window_log2=12)
    y = tracker.process(c, learn=False)
    np.testing.assert_allclose(y, c['q']/2+c['r']-peak/3*c['p'].sum(axis=1), atol=2e-7)
    exact = ex.oracle(c, hdc.FINAL, True, dither_peak=peak)
    np.testing.assert_allclose(exact[64:], x[64:], atol=5e-8)


def test_cli_custom_peak_propagates_to_every_method(tmp_path):
    _, result = ex.run_case(tmp_path/'custom', 71, 'sine', 16, 12,
        formats=True, scenario='constant', peak=.4, dither_peak=.15, noise=0)
    assert result['dither_peak'] == .15
    assert result['diagnostics']['maximum_probe_peak'] == pytest.approx(.15)
    assert result['methods']['oracle_exact']['error_snr_db'] > 65
