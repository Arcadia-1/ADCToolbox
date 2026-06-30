"""Unit tests for plot_spectrum with figure generation and saving."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum import (
    _noise_floor_axis_min,
    _should_label_harmonic,
    plot_spectrum,
)
from adctoolbox.spectrum.plot_spectrum_virtuoso import plot_spectrum_virtuoso


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def _text_values(ax):
    return [text.get_text() for text in ax.texts]


def _max_spur_text(ax):
    return next(text for text in ax.texts if text.get_text() == 'MaxSpur')


def _max_spur_marker(ax):
    return next(line for line in ax.lines if line.get_marker() == 'd')


def _thermal_noise_demo_result(noise_rms=0.0):
    n_fft = 2**13
    fs = 100e6
    fund_bin = 983
    n = np.arange(n_fft)
    signal = 0.5 * np.sin(2 * np.pi * fund_bin * n / n_fft) + 0.5
    if noise_rms:
        rng = np.random.default_rng(20260628)
        signal = signal + rng.standard_normal(n_fft) * noise_rms
    return compute_spectrum(signal, fs=fs)


def _strong_spur_result():
    n_fft = 2**14
    fs = 1.0
    n = np.arange(n_fft)
    signal = (
        0.49 * np.sin(2 * np.pi * 997 * n / n_fft)
        + 0.42 * np.sin(2 * np.pi * 2501 * n / n_fft)
    )
    return compute_spectrum(signal, fs=fs, win_type='rectangular', side_bin=0)


def _assert_spectrum_axes(ax, result, *, show_label=True, expected_title='Power Spectrum'):
    """Verify that plot_spectrum rendered the expected plot structure."""
    spectrum_line = ax.lines[0]
    assert len(spectrum_line.get_xdata()) == len(result['plot_data']['freq'])
    assert len(spectrum_line.get_ydata()) == len(result['plot_data']['power_spectrum_db_plot'])

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    assert xmin > 0
    assert xmax == pytest.approx(result['fs'] / 2)
    assert ymin < 0
    assert ymax == 0

    if expected_title is not None:
        assert ax.get_title() == expected_title

    if show_label:
        assert ax.get_xlabel() == 'Freq (Hz)'
        assert ax.get_ylabel() == 'dBFS'
        texts = _text_values(ax)
        for prefix in ('Fin/fs =', 'ENoB =', 'SNDR =', 'Sig ='):
            assert any(text.startswith(prefix) for text in texts)
        assert 'MaxSpur' in texts
        assert any(line.get_marker() == 'o' for line in ax.lines)
        assert any(line.get_marker() == 'd' for line in ax.lines)
    else:
        assert ax.get_xlabel() == ''
        assert ax.get_ylabel() == ''
        assert _text_values(ax) == []


@pytest.mark.parametrize(
    "nf_line_level,expected",
    [
        (-113.1, -140),
        (-80.0, -100),
        (-79.9, -100),
        (-179.9, -200),
        (-220.0, -200),
    ],
)
def test_noise_floor_axis_min_uses_20db_tick_below_nsd_line(nf_line_level, expected):
    assert _noise_floor_axis_min(nf_line_level) == expected


def test_noise_floor_axis_min_falls_back_to_sndr_level_when_nsd_is_nan():
    assert _noise_floor_axis_min(np.nan, fallback_level=-26.5) == -60


def test_plot_spectrum_uses_sndr_fallback_ylim_when_noise_metrics_are_nan():
    n_fft = 8
    fs = 100e6
    n = np.arange(n_fft)
    signal = (
        0.4 * np.sin(2 * np.pi * n / n_fft)
        + 0.4 * 10**(-30 / 20) * np.sin(2 * 2 * np.pi * n / n_fft)
    )

    with pytest.warns(RuntimeWarning, match="No noise bins remain"):
        result = compute_spectrum(
            signal,
            fs=fs,
            max_scale_range=[-0.5, 0.5],
            win_type='rectangular',
            side_bin=0,
            max_harmonic=5,
            nf_method=4,
        )

    fig, ax = plt.subplots()
    plot_spectrum(result, show_title=False, show_label=False, ax=ax)
    ymin, ymax = ax.get_ylim()
    plt.close(fig)

    assert ymin == -60
    assert ymax == 0


def test_plot_spectrum_labels_stay_fixed_when_ylim_changes():
    n_fft = 2**12
    fs = 100e6
    n = np.arange(n_fft)
    signal = 0.5 * np.sin(2 * np.pi * 123 * n / n_fft)
    result = compute_spectrum(signal, fs=fs, win_type="rectangular", side_bin=0)

    fig, ax = plt.subplots()
    plot_spectrum(result, show_title=False, show_label=True, ax=ax)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    prefixes = ("Fin/fs =", "SNDR =", "Sig =")
    labels = [
        next(text for text in ax.texts if text.get_text().startswith(prefix))
        for prefix in prefixes
    ]
    before = np.array([label.get_window_extent(renderer).bounds for label in labels])

    ax.set_ylim(-80, 0)
    fig.canvas.draw()
    after = np.array([label.get_window_extent(renderer).bounds for label in labels])
    plt.close(fig)

    np.testing.assert_allclose(after, before, atol=0.5)


@pytest.mark.parametrize("plotter", [plot_spectrum, plot_spectrum_virtuoso])
def test_max_spur_annotation_hides_when_spur_is_below_user_ylim(plotter):
    result = _thermal_noise_demo_result(noise_rms=0.0)

    fig, ax = plt.subplots()
    plotter(result, show_title=False, show_label=True, ax=ax)
    ax.set_ylim([-140, 0])
    fig.canvas.draw()

    assert not _max_spur_text(ax).get_visible()
    assert not _max_spur_marker(ax).get_visible()
    plt.close(fig)


@pytest.mark.parametrize("plotter", [plot_spectrum, plot_spectrum_virtuoso])
def test_max_spur_annotation_remains_visible_for_in_range_spur_after_user_ylim(plotter):
    result = _thermal_noise_demo_result(noise_rms=50e-6)

    fig, ax = plt.subplots()
    plotter(result, show_title=False, show_label=True, ax=ax)
    ax.set_ylim([-140, 0])
    fig.canvas.draw()

    max_spur_text = _max_spur_text(ax)
    ymin, ymax = ax.get_ylim()
    assert max_spur_text.get_visible()
    assert _max_spur_marker(ax).get_visible()
    assert ymin <= max_spur_text.get_position()[1] <= ymax
    renderer = fig.canvas.get_renderer()
    text_bbox = max_spur_text.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)
    assert text_bbox.x0 >= axes_bbox.x0 - 0.5
    assert text_bbox.x1 <= axes_bbox.x1 + 0.5
    plt.close(fig)


@pytest.mark.parametrize("plotter", [plot_spectrum, plot_spectrum_virtuoso])
def test_max_spur_annotation_stays_inside_axes_for_high_spur(plotter):
    result = _strong_spur_result()

    fig, ax = plt.subplots(figsize=(7, 5))
    plotter(result, show_title=False, show_label=True, ax=ax)
    fig.canvas.draw()

    max_spur_text = _max_spur_text(ax)
    renderer = fig.canvas.get_renderer()
    text_bbox = max_spur_text.get_window_extent(renderer=renderer)
    axes_bbox = ax.get_window_extent(renderer=renderer)

    assert max_spur_text.get_visible()
    assert _max_spur_marker(ax).get_visible()
    assert text_bbox.x0 >= axes_bbox.x0 - 0.5
    assert text_bbox.x1 <= axes_bbox.x1 + 0.5
    assert text_bbox.y0 >= axes_bbox.y0 - 0.5
    assert text_bbox.y1 <= axes_bbox.y1 + 0.5
    plt.close(fig)


@pytest.mark.parametrize(
    "harmonic_power_db,nf_line_level,expected",
    [
        (-132.0, -113.1, True),
        (-133.1, -113.1, True),
        (-133.2, -113.1, False),
        (-150.0, -113.1, False),
        (-120.0, np.nan, True),
        (np.nan, -113.1, False),
    ],
)
def test_should_label_harmonic_skips_bins_buried_below_nsd_line(
    harmonic_power_db,
    nf_line_level,
    expected,
):
    assert _should_label_harmonic(harmonic_power_db, nf_line_level) is expected


@pytest.mark.parametrize("hd2_target,hd3_target", [
    (-80, -80),
    (-70, -85),
])
def test_plot_spectrum_distorted_sine(hd2_target, hd3_target):
    """Test plot_spectrum with distorted sinewave signals."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    noise_rms = 10e-6
    Fin = 123 / N_fft * Fs  # Coherent frequency

    # Compute nonlinearity coefficients for HD2 and HD3
    hd2_amp = 10**(hd2_target/20)
    hd3_amp = 10**(hd3_target/20)
    k2 = hd2_amp / (A / 2)
    k3 = hd3_amp / (A**2 / 4)

    # Generate signal with both HD2 and HD3 distortion
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2 * np.pi * Fin * t)
    rng = np.random.default_rng(2026062246)
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + rng.standard_normal(N_fft) * noise_rms

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, max_harmonic=6, side_bin=1)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_spectrum(result, show_title=True, show_label=True, plot_harmonics_up_to=3, ax=ax)
    _assert_spectrum_axes(ax, result)
    assert len(ax.lines) >= 5

    # Save figure
    fig_path = output_dir / f'test_plot_distorted_hd2_{hd2_target}_hd3_{hd3_target}.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    print(f"\n[Plot Test] HD2={hd2_target} dBc, HD3={hd3_target} dBc -> Saved to {fig_path}")


def test_plot_spectrum_clean_sine():
    """Test plot_spectrum with clean sinewave (minimal distortion)."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    noise_rms = 50e-6
    Fin = 100 / N_fft * Fs  # Coherent frequency

    # Generate clean signal
    t = np.arange(N_fft) / Fs
    rng = np.random.default_rng(2026062247)
    signal = A * np.sin(2 * np.pi * Fin * t) + rng.standard_normal(N_fft) * noise_rms

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, max_harmonic=5, side_bin=1)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_spectrum(result, show_title=True, show_label=True, plot_harmonics_up_to=3, ax=ax)
    _assert_spectrum_axes(ax, result)

    # Save figure
    fig_path = output_dir / 'test_plot_clean_sine.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    print(f"\n[Plot Test] Clean sine -> Saved to {fig_path}")


def test_plot_spectrum_comparison():
    """Test plot_spectrum with multiple subplots comparing different distortion levels."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    noise_rms = 10e-6
    Fin = 123 / N_fft * Fs

    # Create 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    distortion_configs = [
        {'hd2': -80, 'hd3': -80, 'label': 'Low Distortion'},
        {'hd2': -60, 'hd3': -65, 'label': 'High Distortion'},
    ]

    for idx, config in enumerate(distortion_configs):
        # Compute nonlinearity coefficients
        hd2_amp = 10**(config['hd2']/20)
        hd3_amp = 10**(config['hd3']/20)
        k2 = hd2_amp / (A / 2)
        k3 = hd3_amp / (A**2 / 4)

        # Generate signal
        t = np.arange(N_fft) / Fs
        sig_ideal = A * np.sin(2 * np.pi * Fin * t)
        rng = np.random.default_rng(2026062248 + idx)
        signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + rng.standard_normal(N_fft) * noise_rms

        # Compute spectrum
        result = compute_spectrum(signal, fs=Fs, max_harmonic=6, side_bin=1)

        # Plot on specific axis
        plt.sca(axes[idx])
        plot_spectrum(result, show_title=False, show_label=True, plot_harmonics_up_to=3, ax=axes[idx])
        axes[idx].set_title(f'{config["label"]}: HD2={config["hd2"]} dBc, HD3={config["hd3"]} dBc',
                           fontsize=12, fontweight='bold')
        _assert_spectrum_axes(axes[idx], result, expected_title=None)
        assert config['label'] in axes[idx].get_title()

    # Save figure
    fig_path = output_dir / 'test_plot_comparison.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert len(axes) == len(distortion_configs)
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    print(f"\n[Plot Test] Comparison plot -> Saved to {fig_path}")


def test_plot_spectrum_no_labels():
    """Test plot_spectrum with labels disabled."""
    # Signal parameters
    N_fft = 2**12
    Fs = 50e6
    A = 0.4
    Fin = 50 / N_fft * Fs

    # Generate signal
    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2 * np.pi * Fin * t)

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, max_harmonic=5)

    # Create figure and plot without labels
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_spectrum(result, show_title=False, show_label=False, plot_harmonics_up_to=0, ax=ax)
    _assert_spectrum_axes(ax, result, show_label=False, expected_title='')
    assert len(ax.lines) == 1

    # Save figure
    fig_path = output_dir / 'test_plot_no_labels.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    print(f"\n[Plot Test] No labels -> Saved to {fig_path}")


def test_plot_spectrum_high_harmonics():
    """Test plot_spectrum with many harmonics plotted."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    Fin = 100 / N_fft * Fs

    # Generate signal with multiple harmonics
    hd_levels = [-70, -75, -80, -85, -90, -95]  # HD2-HD7
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2 * np.pi * Fin * t)
    signal = sig_ideal.copy()

    for h_order, hd_db in enumerate(hd_levels, start=2):
        hd_amp = 10**(hd_db/20)
        k_h = hd_amp * (2**(h_order-1)) / (A**(h_order-1))
        signal += k_h * sig_ideal**h_order

    # Compute spectrum
    result = compute_spectrum(signal, fs=Fs, max_harmonic=8, side_bin=1)

    # Create figure and plot with many harmonics
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_spectrum(result, show_title=True, show_label=True, plot_harmonics_up_to=7, ax=ax)
    _assert_spectrum_axes(ax, result)
    texts = set(_text_values(ax))
    assert {'2', '3', '4'}.issubset(texts)
    assert sum(line.get_marker() == 's' for line in ax.lines) >= 3
    for text in ax.texts:
        if text.get_text() in {'2', '3', '4'}:
            assert text.get_clip_on()

    # Save figure
    fig_path = output_dir / 'test_plot_high_harmonics.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    print(f"\n[Plot Test] High harmonics (up to HD7) -> Saved to {fig_path}")


if __name__ == '__main__':
    """Run plotting tests standalone"""
    print('='*80)
    print('RUNNING PLOT_SPECTRUM TESTS')
    print('='*80)

    test_plot_spectrum_clean_sine()
    test_plot_spectrum_distorted_sine(-80, -80)
    test_plot_spectrum_distorted_sine(-70, -85)
    test_plot_spectrum_comparison()
    test_plot_spectrum_no_labels()
    test_plot_spectrum_high_harmonics()

    print('\n' + '='*80)
    print(f'** All plot_spectrum tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
