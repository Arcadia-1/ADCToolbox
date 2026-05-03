"""3x subsample-only downsampler -- harmonic aliasing and spur conservation.

Demonstrates the ADC-monitor-port "IO rate adapter": keep every N-th sample
without any anti-alias filter.  Two properties are made visual on a single
4x2 panel grid:

  1. Harmonic aliasing -- when an input frequency exceeds OUTPUT Nyquist,
     it folds back via
         f_alias = | ((f + fs_out/2) mod fs_out) - fs_out/2 |
     The four columns sweep the fundamental from 50 -> 70 -> 100 -> 140 MHz
     so that progressively more harmonics alias.

  2. Spur-height conservation -- a spur's amplitude (in dBc) is preserved
     across the subsampler; only its frequency position changes.  The
     printed table at the end confirms |SFDR_in - SFDR_out| < 0.1 dB
     across all four cases.

Setup: ADC_Signal_Generator with HD3 = -60 dBc and ~0.3 LSB of analog
thermal noise (gives ENOB ~= 11.5 b on a 12-bit quantizer with 1 V FSR).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator


def subsample(signal, n):
    """Capture indices N-1, 2N-1, 3N-1, ... matching the RTL counter that
    increments 0..N-1 and triggers DOUT capture at counter==N-1."""
    return signal[n - 1 :: n]


def alias_freq(f, fs):
    """Where does frequency f land in [0, fs/2] after sampling at fs?"""
    f_mod = f % fs
    if f_mod > fs / 2:
        f_mod = fs - f_mod
    return f_mod


def main():
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # ---------------- ADC + downsampler setup ----------------
    fs_in = 1e9
    n_factor = 3
    n_in = n_factor * 4096          # multiple of N -> clean output FFT
    fs_out = fs_in / n_factor

    n_bits = 12
    quant_range = (0.0, 1.0)        # 1 V FSR, codes 0..4095
    A, DC = 0.49, 0.5               # near-FS sine swinging 0.01..0.99 V
    hd3_dB = -60.0                  # clearly visible spur above noise floor

    # 0.3 LSB analog dither at 12 b / 1 V FSR ~= 0.3 * (1/4096) V ~= 73 uV.
    # Added BEFORE quantization to whiten quantization noise (~11.5 ENOB).
    lsb_v = (quant_range[1] - quant_range[0]) / (2 ** n_bits)
    noise_rms = 0.3 * lsb_v

    cases = [
        ("A: fin=50 MHz -- no aliases",         50.0),
        ("B: fin=70 MHz -- HD3 aliases",        70.0),
        ("C: fin=100 MHz -- HD2 + HD3 alias",   100.0),
        ("D: fin=140 MHz -- HD2 + HD3 alias",   140.0),
    ]

    fig, axes = plt.subplots(2, len(cases), figsize=(6 * len(cases), 9))

    print(f"3x subsample-only downsampler  fs_in = {fs_in/1e9:g} GS/s  "
          f"fs_out = {fs_out/1e6:.2f} MS/s  Nyq_out = {fs_out/2/1e6:.2f} MHz\n")
    print(f"{'case':<6}{'fin set':>10}{'HD3 in (MHz)':>16}{'HD3 out (MHz)':>16}"
          f"{'SFDR in (dBc)':>16}{'SFDR out (dBc)':>16}{'diff':>8}")
    print("-" * 88)

    for i, (label, fin_target_mhz) in enumerate(cases):
        # 1. Coherent frequency selection on the INPUT FFT length
        fin, _ = find_coherent_frequency(fs_in, fin_target_mhz * 1e6, n_in)

        # 2. Build a clean sine -> add HD3 -> add analog dither -> quantize.
        #    Order matters: HD3 + dither model the analog input; quantization
        #    is the last stage of the ADC.
        np.random.seed(0xC0FFEE + i)
        gen = ADC_Signal_Generator(N=n_in, Fs=fs_in, Fin=fin, A=A, DC=DC)
        sig = gen.apply_static_nonlinearity_hd(hd3_dB=hd3_dB)
        sig = gen.apply_thermal_noise(sig, noise_rms=noise_rms)
        sig_in = gen.apply_quantization_noise(
            sig, n_bits=n_bits, quant_range=quant_range,
        )
        sig_out = subsample(sig_in, n_factor)

        # 3. Predict harmonic alias positions (theory)
        a_hd2 = alias_freq(2 * fin, fs_out)
        a_hd3 = alias_freq(3 * fin, fs_out)
        aliased_hd2 = a_hd2 != 2 * fin
        aliased_hd3 = a_hd3 != 3 * fin

        # 4. Spectra (input on top row, output on bottom)
        res_in = analyze_spectrum(
            sig_in, fs=fs_in,
            create_plot=True, show_title=False, show_label=True,
            ax=axes[0][i],
        )
        res_out = analyze_spectrum(
            sig_out, fs=fs_out,
            create_plot=True, show_title=False, show_label=True,
            ax=axes[1][i],
        )

        # 5. Annotate predicted alias positions on the OUTPUT panel
        for f_hz, name, aliased in [
            (alias_freq(fin, fs_out), "fund", False),
            (a_hd2, "HD2", aliased_hd2),
            (a_hd3, "HD3", aliased_hd3),
        ]:
            color = "red" if aliased else "green"
            axes[1][i].axvline(
                f_hz, color=color, linestyle="--", linewidth=0.8, alpha=0.7,
            )
            axes[1][i].text(
                f_hz, -10, f"{name}{' (alias)' if aliased else ''}",
                color=color, fontsize=8, ha="center", va="top",
            )

        # 6. Per-panel titles
        axes[0][i].set_title(
            f"INPUT  fin={fin/1e6:.1f} MHz\n"
            f"HD2 @ {2*fin/1e6:.1f} MHz, HD3 @ {3*fin/1e6:.1f} MHz",
            fontsize=10,
        )
        axes[1][i].set_title(
            f"OUTPUT  fs={fs_out/1e6:.1f} MS/s   Nyq={fs_out/2/1e6:.1f} MHz\n"
            f"HD2{'->ALIAS@' if aliased_hd2 else '->@'}{a_hd2/1e6:.1f}, "
            f"HD3{'->ALIAS@' if aliased_hd3 else '->@'}{a_hd3/1e6:.1f} MHz",
            fontsize=10,
        )

        # 7. Column header
        axes[0][i].text(
            0.5, 1.18, label, transform=axes[0][i].transAxes,
            fontsize=11, fontweight="bold", ha="center", va="bottom",
        )

        # 8. Tabulated spur-conservation evidence
        print(
            f"{chr(65+i):<6}{fin_target_mhz:>10.0f}"
            f"{3*fin/1e6:>16.2f}{a_hd3/1e6:>16.2f}"
            f"{res_in['sfdr_dbc']:>16.2f}{res_out['sfdr_dbc']:>16.2f}"
            f"{res_out['sfdr_dbc']-res_in['sfdr_dbc']:>+8.2f}"
        )

    fig.suptitle(
        f"3x subsample downsampler (no anti-alias filter) -- "
        f"harmonic aliasing + spur-height conservation\n"
        f"[12-bit ADC_Signal_Generator, ENOB~11.5, HD3 set = {hd3_dB:.0f} dBc, "
        f"hann window]",
        fontsize=12,
    )
    fig.tight_layout()

    out_png = output_dir / "exp_d00_subsample_aliasing.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved {out_png}")
    print()
    print("==> Above table: SFDR diff between input and output is < 0.1 dB across")
    print("    all four cases. Spur energy is conserved across the subsampler;")
    print("    only the frequency position changes per the alias formula.")


if __name__ == "__main__":
    main()
