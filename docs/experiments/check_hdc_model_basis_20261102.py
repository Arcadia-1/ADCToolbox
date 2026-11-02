"""Read-only model review: exact identities and approximation bounds, not ADC validation."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
paths = [
    'python/src/adctoolbox/siggen/nonidealities.py',
    'python/src/adctoolbox/models/sar.py',
    'python/src/adctoolbox/examples/11_calibration/hdc_fixed.py',
    'python/src/adctoolbox/examples/11_calibration/dither_mechanisms.py',
    'python/src/adctoolbox/examples/11_calibration/joint_dither_model.py',
]
errors = {}
checks = 0
for delta in (F(-3, 100), F(0), F(3, 250)):
    for s in (-1, 1):
        for q in (-1, 0, 1):
            for x in (F(-4, 5), F(137, 1000), F(4, 5)):
                cf, cs = 1-delta*s, 1+delta*s
                residue = ((cf+cs)*x-cs*q)/cf
                y = (q+residue)/2
                assert y+delta*s*(q-y) == x
                delta_ratio = cs/cf-1
                assert delta_ratio == 2*delta*s/(1-delta*s)
                for psum in (-3, -1, 1, 3):
                    fixed_charge = (30*x-15*(1+delta*s)*q+2*psum)/15
                    d = F(psum, 15)
                    assert fixed_charge == 2*(x-(1+delta*s)*q/2+d)
                    yy = F(q, 2)+fixed_charge/2-d
                    assert yy+delta*s*q/2 == x
                    checks += 1

memory_bounds = {}
for name, gain, history in [('initial', F(194,100), F(6,100)),
                            ('final', F(204,100), F(12,100))]:
    h = history/gain
    # Difference between 1/(1-h*z^-1) and 1+h*z^-1 on |z|=1.
    worst_tail = h*h/(1-abs(h))
    memory_bounds[name] = dict(h=float(h), second_lag=float(h*h),
                             maximum_transfer_difference=float(worst_tail),
                             dc_relative_difference=float(h*h))

result = dict(
    scope='Algebraic consistency only; no calibration, SNDR, SFDR, or circuit validation',
    exact_charge_identity_cases=checks,
    hypothetical_one_pole_to_FIR_truncation=memory_bounds,
    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True, help='Fresh JSON output path')
out = parser.parse_args().output
assert not out.exists(), 'Use a fresh audit file; do not overwrite prior evidence'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2, ensure_ascii=False))
