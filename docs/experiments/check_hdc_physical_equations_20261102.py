"""Exact charge/state identity checks for a proposed phase table; not circuit simulation."""
from fractions import Fraction as F
from pathlib import Path
from random import Random
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-audit', type=Path, required=True,
                    help='Fresh model-basis JSON generated from this checkout')
parser.add_argument('--output', type=Path, required=True, help='Fresh JSON output path')
args = parser.parse_args()
out = args.output
assert not out.exists(), 'Use a fresh audit file; do not overwrite prior evidence'
model_audit = json.loads(args.model_audit.read_text(encoding='utf-8'))
for path, digest in model_audit['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path

rng = Random(20261102)
cf, cx = F(15), F(30)
cp = [F(2)] * 3
delta = F(3, 250)
cd = [15 * (1 + delta), 15 * (1 - delta)]
caps = [cx, cf, *cd, *cp]
ct = sum(caps)
assert ct == 81
beta = cf/ct

samples = []
for n in range(128):
    x = F(rng.randrange(-70, 71), 100)
    q = 1 if x > F(1,4) else -1 if x < -F(1,4) else 0
    selected = rng.randrange(2)
    p = [rng.choice((-1, 1)) for _ in range(3)]
    u = [F(q if j == selected else 0) for j in range(2)]
    samples.append((x, u, p))

settings = {
    'ideal': [F(0)]*7,
    'feedback_only': [F(0), F(1,17), F(0), F(0), F(0), F(0), F(0)],
    'equal_reset': [F(1,50)]*7,
    'unequal_reset': [F(1,100), F(1,17), F(1,80), F(1,60), F(1,90), F(1,70), F(1,50)],
}
counts = {}
for name, rho in settings.items():
    previous_voltage = [F(0)]*7  # plate alpha minus plate beta at end of A
    previous_b = F(0)
    previous_u = [F(0),F(0)]
    previous_p = [F(0)]*3
    input_filter = F(0)
    for x, u, p in samples:
        target = [-x, F(0), F(0), F(0), F(0), F(0), F(0)]
        reset_voltage = [t+r*(w-t) for t,r,w in zip(target,rho,previous_voltage)]
        q_initial = sum(c*w for c,w in zip(caps,reset_voltage))
        dac = sum(c*w for c,w in zip(cd,u))
        pn = sum(c*w for c,w in zip(cp,p))
        b_from_charge = (-q_initial-dac+pn)/cf
        b_from_state = (cx/cf*(1-rho[0])*x-dac/cf+pn/cf+rho[1]*previous_b
            +sum(c*r*w for c,r,w in zip(cd,rho[2:4],previous_u))/cf
            -sum(c*r*w for c,r,w in zip(cp,rho[4:],previous_p))/cf)
        assert b_from_charge == b_from_state
        if name == 'equal_reset':
            rr = rho[0]
            input_filter = cx/cf*(1-rr)*x + rr*input_filter
            assert b_from_charge == input_filter-dac/cf+pn/cf
        if name == 'ideal':
            for aol in (F(20),F(100),F(1000)):
                b_finite = (-q_initial-dac+pn)/(cf+ct/aol)
                m = -b_finite/aol
                assert ct*m-cf*b_finite-dac+pn == q_initial
        previous_voltage = [F(0),-b_from_charge,*[-z for z in u],*[F(z) for z in p]]
        previous_b, previous_u, previous_p = b_from_charge,u,p
    counts[name] = len(samples)

finite_state_cases = 0
for aol in (F(20),F(100),F(1000)):
    den = cf+ct/aol
    for name,rho in settings.items():
        previous_voltage = [F(0)]*7
        previous_b = F(0)
        previous_u = [F(0)]*2
        previous_p = [F(0)]*3
        rho_weight = sum(c*r for c,r in zip(caps,rho))
        input_filter = F(0)
        for x,u,p in samples:
            target = [-x,F(0),F(0),F(0),F(0),F(0),F(0)]
            reset_voltage = [t+r*(w-t) for t,r,w in zip(target,rho,previous_voltage)]
            qi = sum(c*w for c,w in zip(caps,reset_voltage))
            dac = sum(c*w for c,w in zip(cd,u))
            pn = sum(c*w for c,w in zip(cp,p))
            bb = (-qi-dac+pn)/den
            expected = (cx*(1-rho[0])*x-dac+pn
                +(cf*rho[1]+rho_weight/aol)*previous_b
                +sum(c*r*w for c,r,w in zip(cd,rho[2:4],previous_u))
                -sum(c*r*w for c,r,w in zip(cp,rho[4:],previous_p)))/den
            assert bb == expected
            m = -bb/aol
            assert ct*m-cf*bb-dac+pn == qi
            if name == 'equal_reset':
                rr = rho[0]
                input_filter = cx/den*(1-rr)*x + rr*input_filter
                assert bb == input_filter-dac/den+pn/den
            previous_voltage = [m,m-bb,*[m-z for z in u],*[m+z for z in p]]
            previous_b,previous_u,previous_p = bb,u,p
            finite_state_cases += 1

numerics = {}
for name,gain,history in [('initial',F(194,100),F(6,100)),('final',F(204,100),F(12,100))]:
    h=history/gain
    rho = 2*h/gain  # conditional static-map convention r=f(v+rho*r_previous)
    numerics[name] = dict(h=float(h), rho_feedback_under_static_map=float(rho),
        hypothetical_reset_time_over_tau_if_rho_is_RC_retention=-math.log(float(rho)),
        forward_tail_l1=float(h*h/(1-h)),
        residual_after_current_inverse_l1=float(h*h/(1-h*h)),
        next_impulse_coefficient=float(h*h))

result = dict(scope='Proposed phase-table charge conservation and linear RC-state algebra only; no ADC calibration or transistor simulation',
    state_charge_identity_samples=counts,
    equal_reset_PN_and_DAC_tail_cancellation_samples=128,
    finite_open_loop_gain_charge_identity_cases=384,
    finite_open_loop_gain_state_identity_samples=finite_state_cases,
    total_summing_node_capacitance_Cu=float(ct),feedback_factor=float(beta),
    conditional_memory_mapping=numerics,
    source_sha256=model_audit['source_sha256'],
    model_audit_sha256=hashlib.sha256(args.model_audit.read_bytes()).hexdigest(),
    audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
