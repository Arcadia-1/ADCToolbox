"""Capture plumbing for the repository's signal-generator nonidealities.

Source-monitor labels are observable reference-channel measurements projected
onto the known training tone. Clean input is returned separately for evaluation.
"""
from dataclasses import dataclass, asdict

import numpy as np

from adctoolbox import build_calibration_features
from adctoolbox.siggen import ADC_Signal_Generator


MODEL_ID = 'repository_siggen_static_coarse_memory_v1'


@dataclass(frozen=True)
class FrontendConfig:
    gain: float = 1.
    quadratic: float = .015
    cubic: float = .04
    memory: float = .04
    bits: int = 14
    noise_rms: float = 1e-5
    monitor_noise_rms: float = 2e-4
    monitor_gain: float = 1.
    model_id: str = MODEL_ID

    @classmethod
    def from_manifest(cls, values):
        if values.get('model_id') != MODEL_ID:
            raise ValueError('Legacy/custom foreground plant: rerun exp_cal03 with the repository model')
        return cls(**values)


def tone_projection(observation, frequencies):
    """Fit amplitudes, phases and DC; only frequencies are supplied."""
    n = np.arange(len(observation))
    basis = np.column_stack([np.ones(len(n))] +
                            [f(2*np.pi*frequency*n) for frequency in frequencies for f in (np.sin, np.cos)])
    return basis@np.linalg.lstsq(basis, observation, rcond=None)[0]


def capture(config, *, size, seed, frequencies, amplitudes, phases, chirp=False, history=3):
    """One independent capture, including warm-up and history before sample zero.

    Call the original static/memory/quantizer APIs without reimplementing them.
    The memory API uses np.roll: its wrapped first sample is discarded, together
    with the warm-up. This helper does not promise stateful block conversion.
    A zero memory setting bypasses that API, including its coarse/fine quantizer.
    """
    if config.model_id != MODEL_ID:
        raise ValueError('Unsupported foreground model identity')
    warmup = 64
    n = np.arange(-warmup-history, size)
    if chirp:
        # Sweep stays within Nyquist over the evaluated record.
        clean = .75*np.sin(2*np.pi*(.03*n+.40*n*n/(2*size))+.2)
    else:
        clean = sum(a*np.sin(2*np.pi*f*n+p) for f, a, p in zip(frequencies, amplitudes, phases))
    adc_seed, monitor_seed = np.random.SeedSequence(seed).spawn(2)
    generator = ADC_Signal_Generator(len(clean), 1., 0., 1., 0.)
    analog = generator.apply_static_nonlinearity(
        input_signal=config.gain*clean, k2=config.quadratic, k3=config.cubic)
    if config.memory:
        analog = generator.apply_memory_effect(input_signal=analog, memory_strength=config.memory)
    analog += np.random.default_rng(adc_seed).normal(0, config.noise_rms, len(clean))
    raw = generator.apply_quantization_noise(analog, n_bits=config.bits, quant_range=(-1, 1))
    # The repository API reconstructs the lower edge; express ADC codes at midpoints.
    raw += 2.**-config.bits
    rail = (raw == -1+2.**-config.bits) | (raw == 1-2.**-config.bits)
    monitor = config.monitor_gain*clean + np.random.default_rng(monitor_seed).normal(0, config.monitor_noise_rms, len(clean))
    monitor = generator.apply_quantization_noise(monitor, n_bits=16, quant_range=(-1, 1))+2.**-16
    raw = raw[warmup:]
    result = {"raw": raw, "truth": clean[warmup:], "monitor": monitor[warmup:],
              "overloads": int((abs(analog[warmup+history:]) >= 1).sum()),
              "rails": int(rail[warmup+history:].sum())}
    if not chirp:
        result['reference'] = tone_projection(result['monitor'], frequencies)
        result['output_fit_reference'] = tone_projection(raw, frequencies)
    return result


def dataset(config, *, records, size, seed, stride, exclude_bins=()):
    rng = np.random.default_rng(seed)
    candidates = np.setdiff1d(np.arange(max(3, int(.02*size)) | 1, int(.45*size), 2), exclude_bins)
    if records > len(candidates):
        raise ValueError('Too many records for distinct training tone bins')
    bins = rng.choice(candidates, records, replace=False)
    features, labels, biased_labels, manifest = [], [], [], []
    for i, k in enumerate(bins):
        amplitude, phase = rng.uniform(.05, .90), rng.uniform(0, 2*np.pi)
        capture_seed = int(rng.integers(0, 2**31-1))
        c = capture(config, size=size, seed=capture_seed, frequencies=[k/size],
                    amplitudes=[amplitude], phases=[phase])
        built = build_calibration_features(c['raw'], history=3)
        select = built['sample_indices'][::stride]
        features.append(built['features'][::stride])
        labels.append(c['reference'][select])
        biased_labels.append(c['output_fit_reference'][select])
        manifest.append({'record': i, 'tone_bin': int(k), 'amplitude': float(amplitude),
                         'phase': float(phase), 'capture_seed': capture_seed,
                         'overloads': c['overloads'], 'rails': c['rails']})
    return {'features': np.vstack(features), 'reference': np.concatenate(labels),
            'output_fit_reference': np.concatenate(biased_labels), 'manifest': manifest,
            'plant': asdict(config)}
