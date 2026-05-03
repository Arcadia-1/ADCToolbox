# 09_downsample — Subsample-without-filter (IO rate adapter)

This folder demonstrates the **subsample-only downsampler** pattern used in
chip-level ADC monitor / debug output ports — *not* a DSP downsampler with
anti-alias filter. The examples cover the **signal-domain** behavior you
can observe in Python before committing to RTL.

## Examples

| File | Topic |
|---|---|
| `exp_d00_subsample_aliasing.py` | 4-column grid (fin = 50 / 70 / 100 / 140 MHz) at N = 3. Verifies the harmonic-alias formula and demonstrates spur-height conservation (SFDR diff < 0.1 dB across all four cases). |

## Why this pattern matters

A real ADC core can run at GS/s rates internally but a chip pad can
typically only drive ~32–100 MS/s. The standard solution is a
**subsample-only** block: discard N−1 of every N samples, send the
survivor to the pad. No anti-alias filter — *intentionally* — because
we want to see the ADC's raw codes for offline characterization
(INL/DNL, code histograms, channel match), not a filtered/reconstructed
waveform.

Two consequences worth understanding before depending on this pattern:

1. **Harmonics alias** by the standard rule
   `f_alias = | ((f + fs_out/2) mod fs_out) − fs_out/2 |`.
   Spur energy is preserved (Parseval); only the frequency position
   moves. This is what `exp_d00` makes quantitative.

2. **NSD** (noise spectral density, dBFS/Hz) **degrades by 10·log₁₀(N)**
   — the same total noise power packed into 1/N the bandwidth raises
   per-Hz density. Not a sign of design failure; it is the inevitable
   cost of removing the anti-alias filter.

## Choosing the downsample ratio N

For time-interleaved ADCs (K-way interleaving), N must be **coprime
to K** so every channel is visited evenly during long characterization
runs. Common choices: N = 31 (prime, safe for any K ≤ 30), N = 15
(works for binary K ∈ {2, 4, 8, 16}), N = 7 (works for any non-7 K).
Avoid powers of 2 when the ADC is binary-interleaved — they lock to a
single channel.

## Running

```bash
cd python/src/adctoolbox/examples/09_downsample
python exp_d00_subsample_aliasing.py
# saves output/exp_d00_subsample_aliasing.png
```
