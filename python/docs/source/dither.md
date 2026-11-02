# Dither mechanisms

Dither can supply a known excitation for parameter calibration, or randomize
quantization error without estimating a parameter. Its physical injection
point and digital processing depend on the ADC architecture.

| Architecture | Injection and observation | Example |
|---|---|---|
| Pipeline | Before residue amplifier; current/triple/delayed PN correlations | `exp_cal08_hdc_fixed.py` |
| SAR | Held-sample/CDAC offset; paired raw decisions and bit weights | `exp_cal09_pn_architectures.py` |
| Time interleaved | Before sampler with known analog slope; channel correlations | `exp_cal09_pn_architectures.py` |
| Delta-sigma | Input or quantizer node; transfer-aware subtraction and in-band noise | `exp_cal09_pn_architectures.py` |

The common probe is a sum of three equal PN sequences. This does not make the
estimators interchangeable. The SAR example needs two conversions of one held
input, the TI model needs an explicitly defined analog probe slope, and the
Delta-sigma example does not identify an amplifier polynomial.

The canonical guide ships with the examples as
`11_calibration/DITHER_GUIDE.md`. Download the
{download}`complete guide <../../src/adctoolbox/examples/11_calibration/DITHER_GUIDE.md>`
and the
{download}`fixed-point HDC specification <../../src/adctoolbox/examples/11_calibration/HDC_FIXED.md>`.
The guides are also rendered below from the same packaged sources.

```{toctree}
:maxdepth: 1

DITHER_GUIDE
HDC_FIXED
DITHER_COVERAGE
JOINT_DITHER
JOINT_CONVERGENCE
```

```bash
adctoolbox-get-examples
cd adctoolbox_examples
python 11_calibration/exp_cal09_pn_architectures.py --quick --output output/pn_smoke
```

The quick run checks execution, not convergence. Full-size runs benefit from
optional Numba acceleration. Output directories must be new. The examples save
source hashes, configurations, per-case measurements and figures, including
controls that deliberately violate the observation assumptions.

Only the HDC example has a tested fixed-point arithmetic/resource contract;
none of these examples claims verified transistor behavior, RTL timing, or
general success on every ADC or input. The public `DitherGainTracker`,
`apply_dither_gain` and `DitherHDCTracker` interfaces reuse the example
algorithms; see the [API reference](api/dither.rst). The sine-fit baseline
and behavioral plants remain example-only helpers.

The repository also includes the {download}`model-basis review <../../../docs/experiments/2026-11-02-hdc-model-basis-review.md>`
and {download}`physical-state and switch-phase derivation <../../../docs/experiments/2026-11-02-hdc-physical-state-and-phases.md>`.
The one-tap memory extension is a conditional behavioral approximation, not a
validated model of every reset/settling mechanism.

The {download}`results summary <../../../docs/experiments/dither-results.md>`
contains representative measurements and commit-pinned links to the detailed
experiment archive.
