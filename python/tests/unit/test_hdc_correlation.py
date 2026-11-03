"""Exact error attribution and controls that preserve only the intended path."""
import importlib
import numpy as np
import pytest

h=importlib.import_module('adctoolbox.examples.11_calibration.hdc_fixed')
c=importlib.import_module('adctoolbox.examples.11_calibration.hdc_correlation')


def test_diagnostic_stage_matches_existing_physical_model_and_stream_state():
    x=.65*np.sin(np.arange(8192)*.17)
    diagnostic=c.DiagnosticPlant(56,noise=.0002)
    plant=h.Plant(56,noise=.0002)
    for start,stop in [(0,3073),(3073,8192)]:
        raw=diagnostic.capture(x[start:stop]); observed,_=plant.convert(x[start:stop])
        np.testing.assert_array_equal(raw['p'],observed['p'])
        quantized=c.quantize(2*(raw['r0']+raw['noise_r']),13)[0]/2
        np.testing.assert_array_equal(quantized,observed['r'])


@pytest.mark.parametrize('mode',['isolated','joint'])
@pytest.mark.parametrize('noise',[0.,.0002])
def test_finite_record_cubic_and_gain_error_partitions_are_exact(mode,noise):
    rows=c.streaming(56,mode,c.CASES[1],noise,[12,14])
    for r in rows:
        assert abs(r['cubic_decomposition_residual'])<1e-12
        assert abs(r['gain_decomposition_residual'])<1e-12
        assert abs(r['finite_partition_residual'])<1e-12
        assert r['overloads']==0
        assert r['estimated_cubic']==pytest.approx(h.FINAL.cubic+r['finite_cubic_error']+r['noise_cubic_error']+r['quant_cubic_error'],abs=1e-12)


def test_balanced_repetition_keeps_bias_but_error_permutation_removes_it():
    rows=c.balanced_controls(repeats=(1,64))
    dc=[r for r in rows if r['case']=='dc_bias_control' and r['backend_bits']==13]
    assert dc[0]['quant_cubic_error']==pytest.approx(dc[1]['quant_cubic_error'],abs=1e-12)
    assert abs(dc[1]['quant_cubic_error'])>.02
    assert abs(dc[1]['shuffled_quant_cubic_error'])<.001
    assert abs(dc[1]['independent_reference_estimated_cubic'])<.02
    for r in dc:
        assert abs(r['finite_cubic_error'])<1e-12
        assert r['mean_c3']==0
        assert r['mean_error_times_c3']==pytest.approx(
            (r['mean_error_given_c3_positive']-r['mean_error_given_c3_negative'])/2,abs=1e-14)


def test_unquantized_reference_has_no_quantization_contribution():
    rows=c.streaming(78,'joint',c.CASES[0],.0002,[12])
    row=next(r for r in rows if r['backend_bits']==0)
    assert row['quant_cubic_error']==row['quant_error_rms']==0
    assert row['error_c3_correlation'] is None


def test_cli_artifacts_and_refusal_to_overwrite(tmp_path):
    args=['--output',str(tmp_path),'--quick','--seeds','44','--noise-rms','0']
    assert c.main(args)==0
    for name in ('configuration.json','balanced.csv','streaming.csv','completion.json'):
        assert (tmp_path/name).stat().st_size>100
    with pytest.raises(SystemExit): c.main(args)
