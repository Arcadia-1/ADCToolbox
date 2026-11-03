"""Physical path, causal learning and tracking checks for the packaged example."""
import importlib

import numpy as np
import pytest

model = importlib.import_module('adctoolbox.examples.11_calibration.pipeline_dither_model')
experiment = importlib.import_module('adctoolbox.examples.11_calibration.exp_cal01_dither_background')


def noiseless(**kwargs):
    return model.PipelineConfig(input_noise_rms=0., residue_noise_rms=0., **kwargs)


def test_nominal_pipeline_resolution_and_probe_removal():
    config = noiseless()
    x = np.linspace(-.85,.85,20000)
    capture = model.DitherPipeline(config,19).convert(x,4.)
    corrected = model.reconstruct(capture['coarse'],capture['backend'],capture['pn'],4.,config.probe_amplitude)
    assert np.max(abs(corrected-x)) <= 1/(2**config.backend_bits*4)+1e-14
    assert capture['backend_overloads'] == 0


def test_probe_passes_through_gain_but_not_coarse_decision():
    config = noiseless(backend_bits=24)
    x = np.linspace(-.8,.8,4096)
    on = model.DitherPipeline(config,2).convert(x,3.88)
    off = model.DitherPipeline(config,2).convert(x,3.88,probe_enabled=False)
    np.testing.assert_array_equal(on['coarse'],off['coarse'])
    np.testing.assert_allclose(on['backend']-off['backend'],3.88*config.probe_amplitude*on['pn'],atol=2/2**24)


def test_streaming_partition_preserves_probe_noise_and_drift():
    config = model.PipelineConfig()
    x = np.sin(np.arange(20003)*.113)*.85
    gains = np.linspace(3.88,4.12,len(x))
    whole = model.DitherPipeline(config,91).convert(x,gains)
    adc = model.DitherPipeline(config,91)
    chunks = [adc.convert(x[a:b],gains[a:b]) for a,b in ((0,137),(137,8193),(8193,len(x)))]
    for key in ('coarse','backend','pn','backend_code'):
        np.testing.assert_array_equal(whole[key],np.concatenate([c[key] for c in chunks]))
    assert adc.sample_count == len(x)
    adc.reset()
    np.testing.assert_array_equal(adc.convert(x,gains)['backend'],whole['backend'])


def test_correlation_estimator_with_independent_nuisance():
    rng = np.random.default_rng(13)
    pn = rng.choice([-1.,1.],4096)
    centered = pn-pn.mean()
    nuisance = .1*np.sin(np.arange(len(pn))*.731)
    nuisance -= centered*(centered@nuisance)/(centered@centered)
    backend = 3.88*.0625*pn+nuisance+.03
    code = np.floor((backend+1)*2048).astype(int)
    tracker = model.gain_tracker(learning_rate=.1)
    for _ in range(100):
        result = tracker.update(backend,pn,code)
    assert result['status'] == 'updated'
    assert tracker.gain_estimate == pytest.approx(3.88,abs=4e-6)
    tracker.reset()
    assert tracker.gain_estimate == 4.


@pytest.mark.parametrize('failure',['no_probe','constant_probe','rail','out_of_range'])
def test_unreliable_blocks_hold_estimate(failure):
    tracker = model.gain_tracker()
    pn = np.tile([-1.,1.],64)
    backend = .25*pn
    code = np.full(len(pn),2048)
    if failure == 'no_probe':pn.fill(0.)
    elif failure == 'constant_probe':pn.fill(1.)
    elif failure == 'rail':code[17] = 0
    else:backend *= 2
    result = tracker.update(backend,pn,code)
    assert result['status'].startswith('held_')
    assert tracker.gain_estimate == 4.
    assert tracker.updates == 0


def test_foreground_sine_fit_recovers_gain_without_amplitude_or_truth_label():
    config = noiseless(backend_bits=24)
    size = 32768
    adc = model.DitherPipeline(config,8)
    n = np.arange(size)
    x = .81*np.sin(2*np.pi*4511*n/size+.391)+.01
    capture = adc.convert(x,3.88,probe_enabled=False)
    fitted = model.fit_foreground_gain([capture],4511/size)
    assert fitted['gain_estimate'] == pytest.approx(3.88,abs=1e-7)


def test_foreground_rejects_enabled_probe():
    capture = model.DitherPipeline(noiseless(),9).convert(np.sin(np.arange(4096))*.7,3.88)
    with pytest.raises(ValueError,match='probe off'):
        model.fit_foreground_gain([capture],.137)


def test_physical_probe_overload_is_detected_and_holds_tracker():
    config = noiseless(probe_amplitude=.1875)
    x = .85*np.sin(2*np.pi*561*np.arange(4096)/4096)
    capture = model.DitherPipeline(config,11).convert(x,3.88)
    assert capture['backend_overloads'] > 0
    tracker = model.gain_tracker(config)
    result = tracker.update(capture['backend'],capture['pn'],capture['backend_code'])
    assert result['status'] == 'held_backend_rail'
    assert tracker.gain_estimate == 4.


def test_gain_and_probe_amplitude_are_not_separately_identifiable():
    pn = np.tile([-1.,1.],2048)
    backend = 4.*.07*pn
    code = np.floor((backend+1)*2048)
    tracker = model.gain_tracker(learning_rate=1.)
    tracker.update(backend,pn,code)
    assert tracker.gain_estimate == pytest.approx(4.*.07/.0625)


def test_spectrum_evaluator_matches_toolbox_for_single_tone():
    from adctoolbox import analyze_spectrum
    n = np.arange(16384)
    clean = .8*np.sin(2*np.pi*137*n/len(n))
    y = clean+.0008*np.cos(2*np.pi*411*n/len(n))
    actual = experiment.metrics(y,clean,[137])
    expected = analyze_spectrum(y,fs=1.,win_type='rectangular',side_bin=0,create_plot=False)
    assert actual['spectral_ratio_db'] == pytest.approx(expected['sndr_dbc'],abs=1e-8)
    assert actual['spur_ratio_db'] == pytest.approx(expected['sfdr_dbc'],abs=1e-8)


def test_step_tracking_is_causal_and_beats_frozen_foreground():
    rows,result = experiment.run_case(model.PipelineConfig(),seed=42,scenario='step',kind='sine',
        blocks=128,block_size=8192,learning_rate=.15,foreground_blocks=8)
    assert rows[0]['background_gain_used'] == 4.
    assert rows[0]['background_spectral_ratio_db'] == rows[0]['uncalibrated_spectral_ratio_db']
    for previous,current in zip(rows,rows[1:]):
        assert current['background_gain_used'] == previous['background_gain_next']
    tail = result['summary']['windows']['final_quarter']
    assert tail['background']['spectral_ratio_db'] > tail['foreground_frozen']['spectral_ratio_db']+10
    assert result['summary']['backend_overloads'] == 0
    assert result['summary']['held_updates'] == 0


def test_example_cli_exports_trajectories_and_figure(tmp_path):
    import json
    experiment.main(['--output',str(tmp_path),'--blocks','64','--block-size','2048',
                     '--foreground-blocks','2','--seeds','11','--scenarios','constant','--waveforms','sine'])
    result = json.loads((tmp_path/'summary.json').read_text())
    assert len(result['runs']) == 1
    configuration = json.loads((tmp_path/'configuration.json').read_text())
    assert {'dither_gain.py', 'dither.py'} <= configuration['source_sha256'].keys()
    assert (tmp_path/'sine_constant_seed11'/'trajectory.csv').is_file()
    assert (tmp_path/'tracking_sine.png').stat().st_size > 1000


def test_examples_reuse_public_gain_api_and_keep_sine_baseline_local():
    import adctoolbox
    import adctoolbox.calibration as calibration
    helpers = importlib.import_module("adctoolbox.examples.11_calibration.dither_gain")
    assert helpers.DitherGainTracker is adctoolbox.DitherGainTracker
    assert helpers.apply_gain_calibration is calibration.apply_dither_gain
    for name in ("apply_gain_calibration", "fit_pipeline_gain_sine"):
        assert not hasattr(adctoolbox, name)
        assert not hasattr(calibration, name)
        assert name not in adctoolbox.__all__
        assert name not in calibration.__all__
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("adctoolbox.calibration.background")
    assert model.DitherGainTracker is helpers.DitherGainTracker
    assert model.fit_foreground_gain is helpers.fit_pipeline_gain_sine


def test_hybrid_gain_snapshot_reset_and_hold():
    tracker = model.gain_tracker(initial_gain=3.9, learning_rate=1)
    p = np.tile([-1., 1.], 100)
    r = 4.1*.0625*p
    codes = np.floor((r+1)*2048).astype(int)
    before = tracker.apply(np.zeros(len(p)), r, p)
    assert before['gain_used'] == 3.9
    update = tracker.update(r, p, codes)
    assert update['gain_next'] == pytest.approx(4.1)
    assert before['gain_used'] == 3.9
    np.testing.assert_allclose(tracker.apply(np.zeros(len(p)), r, p)['calibrated_signal'], 0, atol=1e-15)
    tracker.update(r, p, np.zeros(len(p)))
    assert tracker.updates == 1
    tracker.reset()
    assert tracker.gain_estimate == 3.9 and tracker.updates == 0


def test_foreground_qr_fit_is_capture_partition_invariant():
    n = np.arange(8192)
    f = 137/8192
    x = .8*np.sin(2*np.pi*f*n+.71)
    q = (np.searchsorted(np.arange(-5, 6, 2)/8, x)-3)/4
    r = 3.88*(x-q)
    capture = {'coarse': q, 'backend': r, 'pn': np.zeros(len(n))}
    whole = model.fit_foreground_gain([capture], f)
    pieces = [{key: value[i:i+113] for key, value in capture.items()} for i in range(0, len(n), 113)]
    streamed = model.fit_foreground_gain(pieces, f)
    assert whole['gain_estimate'] == pytest.approx(3.88, abs=1e-12)
    assert streamed['gain_estimate'] == pytest.approx(whole['gain_estimate'], abs=1e-12)
    np.testing.assert_allclose(model.apply_gain_calibration(q, r, np.zeros(len(n)), gain_estimate=3.88,
                                                    probe_amplitude=.0625)['calibrated_signal'], x, atol=1e-15)
    np.testing.assert_allclose(model.apply_gain_calibration(q, r, np.zeros(len(n)), gain_estimate=3.88,
                                                    probe_amplitude=0.)['calibrated_signal'], x, atol=1e-15)


def test_foreground_rejects_no_excitation_or_nonzero_probe():
    with pytest.raises(ValueError):
        model.fit_foreground_gain([], .1)
    c = {'coarse': np.zeros(100), 'backend': np.zeros(100), 'pn': np.zeros(100)}
    with pytest.raises(ValueError):
        model.fit_foreground_gain([c], .1)
    c['pn'][:] = 1
    with pytest.raises(ValueError):
        model.fit_foreground_gain([c], .1)
