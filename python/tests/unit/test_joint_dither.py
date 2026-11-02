"""Joint-model charge/inverse identities, observable identification and failure guards."""
import importlib

import numpy as np
import pytest

jm = importlib.import_module('adctoolbox.examples.11_calibration.joint_dither_model')
ex = importlib.import_module('adctoolbox.examples.11_calibration.exp_cal06_joint_dither')


def test_joint_plant_and_inverse_preserve_streams_across_parameter_step():
    x = .8*np.sin(np.arange(10003)*.317)
    whole, _ = jm.JointPlant(41).convert(x)
    plant = jm.JointPlant(41)
    pieces = [plant.convert(x[a:b])[0] for a,b in ((0,13),(13,4139),(4139,len(x)))]
    for key in ('q','qp','r','p','pp','s'):
        np.testing.assert_array_equal(whole[key], np.concatenate([c[key] for c in pieces]))
    plant = jm.JointPlant(41,bits=24,noise=0.)
    corrector = jm.JointCorrector()
    result=[]
    for a,b,parameters in ((0,4139,jm.INITIAL),(4139,len(x),jm.FINAL)):
        c,d=plant.convert(x[a:b],parameters)
        result.append(corrector.apply(c,parameters.inverse())['signal'])
        assert d['overloads']==0
    assert np.max(abs(np.concatenate(result)-x)) < 5e-8


def test_joint_electronics_moments_identify_three_coefficients_without_truth():
    # Exact Cartesian independence: current/previous p, cap roles and business x.
    x,p,pp,s,sp=np.meshgrid(np.linspace(-.8,.8,101),[-1.,-1/3,1/3,1.],
                           [-1.,-1/3,1/3,1.],[-1.,1.],[-1.,1.],indexing='ij')
    x,p,pp,s,sp=[a.ravel() for a in (x,p,pp,s,sp)]
    xp=.6-.2*x
    q=np.where(x>.25,1.,np.where(x<-.25,-1.,0.))
    qp=np.where(xp>.25,1.,np.where(xp<-.25,-1.,0.))
    t=jm.FINAL
    v=(2*x-(1+t.delta*s)*q)/(2*(1-t.delta*s))+.1*p
    vp=(2*xp-(1+t.delta*sp)*qp)/(2*(1-t.delta*sp))+.1*pp
    r=jm.compressed_amplifier(t.gain*v+t.history*vp,1.,t.beta)
    # Only the 3 electronics moments are tested; these Cartesian rows are
    # deliberately not a continuous stream for the recursive cap deembedding.
    tracker=jm.JointTracker()
    c=dict(r=r,p=p,pp=pp,s=s,q=q,qp=qp,rail=False)
    assert tracker.update(c)['electronics']=='updated'
    np.testing.assert_allclose(tracker.coefficients[:3],t.inverse()[:3],atol=1e-11)


def test_capacitor_moment_uses_corrected_electronics_and_no_clean_labels():
    plant=jm.JointPlant(7,bits=24,noise=0.)
    tracker=jm.JointTracker(rate=.001,cap_rate=.1)
    # Constant within-code business input removes signal variation as a
    # nuisance, while exercising both estimators on a continuous stream.
    x=np.full(32768,.5)
    for _ in range(4):
        c,_=plant.convert(x,jm.FINAL)
        status=tracker.update(c)
    assert status['capacitor']=='updated'
    assert tracker.coefficients[3] == pytest.approx(jm.FINAL.delta,abs=5e-5)


@pytest.mark.parametrize('kwargs,peak,expected', [
    ({'binary':True},.8,('held_rank','held_electronics')),
    ({},.1,('updated','held_excitation')),
    ({'amplitude':.4},.8,('held_rail','held_rail')),
])
def test_joint_guards(kwargs,peak,expected):
    plant=jm.JointPlant(11,**kwargs)
    tracker=jm.JointTracker(plant.amplitude)
    for _ in range(4):
        x,_=ex.waveform(np.arange(32768),32768,'sine',peak)
        c,_=plant.convert(x,jm.FINAL)
        status=tracker.update(c)
    assert (status['electronics'],status['capacitor'])==expected
    if expected[0] in ('held_rank','held_rail'):
        np.testing.assert_array_equal(tracker.coefficients,jm.NOMINAL.inverse())


def test_rail_is_observable_and_diagnostics_are_separate():
    c,d=jm.JointPlant(11,amplitude=.4).convert(.8*np.sin(np.arange(4096)),jm.FINAL)
    assert set(c)=={'q','qp','r','p','pp','s','rail'}
    assert d['overloads']>0 and d['headroom']<0 and c['rail']
    assert jm.JointTracker(.4).update(c)['electronics']=='held_rail'


def test_metrics_match_known_harmonic_power():
    n=np.arange(4096)
    clean=.8*np.sin(2*np.pi*37*n/4096)
    distorted=clean+.008*np.sin(2*np.pi*74*n/4096)
    result=ex.metrics(distorted,clean,[37])
    assert result['spectral_ratio_db']==pytest.approx(40.,abs=1e-10)
    assert result['spur_ratio_db']==pytest.approx(40.,abs=1e-10)
    assert result['error_snr_db']==pytest.approx(40.,abs=1e-10)


def test_joint_run_is_causal_and_reports_real_limits():
    rows,summary=ex.run_case(seed=11,blocks=64,size=16384)
    assert summary['max_causality_gap']==0.
    assert summary['overloads']==0
    assert rows[0]['joint_spectral_ratio_db']==rows[0]['nominal_spectral_ratio_db']
    assert summary['windows']['final_quarter']['oracle']['spectral_ratio_db']>70
    final=summary['windows']['final_quarter']
    assert final['joint']['spectral_ratio_db'] > final['nominal']['spectral_ratio_db']+5
    assert all(r['used_gain']==rows[i-1]['next_gain'] for i,r in enumerate(rows) if i)
    assert summary['final_in_band_fraction']<=1


def test_cli_exports_and_refuses_nonempty_destination(tmp_path):
    args=['--output',str(tmp_path),'--blocks','16','--block-size','1024',
          '--seeds','11','--waveforms','sine','--scenarios','constant']
    ex.main(args)
    for filename in ('configuration.json','summary.json','aggregate.csv','headroom.csv','diagnostics.json','joint_headroom.png'):
        assert (tmp_path/filename).stat().st_size>100
    with pytest.raises(SystemExit):
        ex.main(args)
