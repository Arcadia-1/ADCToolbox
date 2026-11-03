"""Probe visibility, observable SAR learning, and loop transfer identities."""
import importlib
import itertools
import json
import numpy as np
import pytest

pa=importlib.import_module('adctoolbox.examples.11_calibration.pn_architectures')
ex=importlib.import_module('adctoolbox.examples.11_calibration.exp_cal09_pn_architectures')


def test_three_pn_levels_and_lookahead_partition_invariance():
    one,split=pa.ThreePN(19),pa.ThreePN(19)
    expected=one.adjacent(8192)
    chunks=[split.adjacent(n) for n in (1,17,4096,4078)]
    for j in range(3):
        np.testing.assert_array_equal(expected[j],np.concatenate([c[j] for c in chunks]))
    np.testing.assert_array_equal(expected[2][:-1],expected[1][1:])
    assert max(abs(expected[1]))==pytest.approx(.2)
    p,d=pa.ThreePN(19).draw(131072)
    np.testing.assert_allclose([np.mean(p.sum(axis=1)==v) for v in (-3,-1,1,3)], [.125,.375,.375,.125],atol=.004)
    assert np.var(d)==pytest.approx(.2**2/3,rel=.02)


def test_pipeline_post_amplifier_probe_has_no_cubic_information():
    rows={r['mode']:r for r in ex.pipeline_boundaries()}
    assert rows['before_amplifier']['estimated_cubic']==pytest.approx(-.24,abs=1e-12)
    assert rows['after_amplifier']['estimated_cubic']==pytest.approx(0,abs=1e-12)
    assert rows['after_amplifier']['corrected_error_rms'] > rows['before_amplifier']['corrected_error_rms']
    assert rows['fifth_order']['estimated_cubic']==pytest.approx(rows['fifth_order']['expected_cubic'],abs=1e-12)
    assert abs(rows['fifth_order']['estimated_cubic']+.24)>.05


def test_subbinary_decisions_retain_resolution_but_binary_gap_loses_information():
    w=pa.nominal_sar_weights()
    x=np.linspace(-.9,.9,10003)
    b=pa.sar_decisions(x,w)
    assert max(abs(x-b@w)) <= w[-1]+1e-14
    binary=pa.nominal_sar_weights(decisions=12,radix=2.)
    binary[0] *= 1.01
    bits=pa.sar_decisions(np.array([.0001,.0004]),binary)
    np.testing.assert_array_equal(bits[0],bits[1])


def test_sar_paired_learning_and_causal_partitioning():
    rng=np.random.default_rng(121)
    x=rng.uniform(-.7,.7,131072)
    _,d=pa.ThreePN(121).draw(len(x))
    actual=pa.nominal_sar_weights(); actual[:4] *= [1.008,.994,1.005,.996]
    bp,bm=pa.sar_decisions(x+d,actual),pa.sar_decisions(x-d,actual)
    one,split=pa.SARWeights(),pa.SARWeights()
    expected=one.process(bp,bm,d)
    first=(bp[0]@split.weights+bm[0]@split.weights)/2
    output=np.r_[split.process(bp[:12345],bm[:12345],d[:12345]),split.process(bp[12345:],bm[12345:],d[12345:])]
    assert output[0]==pytest.approx(first,abs=1e-14)
    np.testing.assert_array_equal(output,expected)
    np.testing.assert_array_equal(one.weights,split.weights)
    assert np.max(abs((one.weights[:4]-actual[:4])/actual[:4]))<.0005
    assert one.updated>0 and one.rejected==0
    saved=one.weights.copy(); one.process(bp[:100],bm[:100],d[:100],learn=False)
    np.testing.assert_array_equal(one.weights,saved)


def test_sar_same_input_is_a_real_observation_requirement():
    w=pa.nominal_sar_weights(); x=np.linspace(-.7,.7,4096)
    _,d=pa.ThreePN(82).draw(len(x))
    bp=pa.sar_decisions(x+d,w)
    correct=pa.sar_decisions(x-d,w)
    broken=pa.sar_decisions(-x-d,w)
    assert np.std((bp-correct)@w-2*d) < 2*w[-1]
    assert np.std((bp-broken)@w-2*d)>.5


@pytest.mark.parametrize('placement,expected_gain,expected_skew',[
    ('linear_before',1.02,.04),('flat_before',1.02,0.),('after_sampler',1.,0.)])
def test_ti_timing_visibility_depends_on_slope_and_location(placement,expected_gain,expected_skew):
    grid=np.array(list(itertools.product((-1,1),repeat=6)))
    d0,d1=grid[:,:3].sum(axis=1)/15,grid[:,3:].sum(axis=1)/15
    probe=.96*d0+.04*d1 if placement=='linear_before' else d0
    y=1.02*(.3+probe) if placement!='after_sampler' else 1.02*.3+probe
    tracker=pa.TIMoments(channels=1)
    tracker.update(y,d0,d1,np.zeros(len(y),dtype=int))
    estimate=tracker.estimate()
    assert estimate['valid']
    assert estimate['gain'][0]==pytest.approx(expected_gain,abs=1e-14)
    assert estimate['skew'][0]==pytest.approx(expected_skew,abs=1e-14)


@pytest.mark.parametrize('at_quantizer',[True,False])
def test_sigma_delta_transfer_identity_and_streaming_state(at_quantizer):
    x=.4*np.sin(np.arange(10000)*.01)
    _,d=pa.ThreePN(59).draw(len(x))
    y,e,state,maximum=pa.sigma_delta(x,d,at_quantizer)
    path=pa.difference(d) if at_quantizer else d
    np.testing.assert_allclose(y,x+path+pa.difference(e),atol=1e-14)
    left,_,memory,_=pa.sigma_delta(x[:3123],d[:3123],at_quantizer)
    right,_,end,_=pa.sigma_delta(x[3123:],d[3123:],at_quantizer,memory)
    np.testing.assert_array_equal(y,np.r_[left,right]); assert state==end
    assert maximum<1.3
    if at_quantizer:
        np.testing.assert_allclose((y-d)-(y-path),-np.r_[0,d[:-1]],atol=1e-14)


def test_dc_does_not_get_a_misleading_sndr_label():
    clean=np.full(32768,1/64)
    y,_,_,_=pa.sigma_delta(clean,np.zeros(len(clean)))
    m=pa.inband_metrics(y,clean,[])
    assert 'sndr_db' not in m and 'sfdr_db' not in m
    assert m['largest_inband_spur_dbfs'] == pytest.approx(-30.1029995664,abs=1e-8)


def test_architecture_cli_artifacts_and_no_overwrite(tmp_path):
    args=['--quick','--seeds','44','--output',str(tmp_path)]
    assert ex.main(args)==0
    for name in ('configuration.json','summary.json','pipeline.csv','heldout_spectra.csv','mechanisms.png'):
        assert (tmp_path/name).stat().st_size>100
    assert json.loads((tmp_path/'completion.json').read_text())['source_hashes_match']
    with pytest.raises(SystemExit): ex.main(args)
