# Documentation Figure Generation (MATLAB)

This directory contains scripts to generate all documentation figures for ADCToolbox.

## Structure

```
matlab/
├── generate_all_figures.m       # Master script - runs everything
├── generate_canonical_data.m    # Creates synthetic test datasets
├── scripts/                     # Per-tool example scripts
│   ├── example_FGCalSine.m
│   ├── example_specPlot.m
│   ├── example_INLsine.m
│   └── ...
└── README.md                    # This file
```

## Quick Start

```matlab
cd d:\ADCToolbox\doc\examples\matlab
generate_all_figures()
```

This will:
1. Generate canonical datasets in `../data/`
2. Run all example scripts
3. Save figures to `../../figures/`

## Output

Figures are organized by tool:

```
doc/figures/
├── FGCalSine/
│   ├── basic_calibration.png
│   ├── rank_deficiency.png
│   └── frequency_search.png
├── specPlot/
│   ├── ideal_spectrum.png
│   ├── with_distortion.png
│   └── before_after_calibration.png
└── ...
```

## Adding New Figures

To add figures for a new tool:

1. Create `scripts/example_YourTool.m`:
```matlab
function example_YourTool()
    outDir = '../../figures/YourTool';
    if ~isfolder(outDir), mkdir(outDir); end

    % Load data
    bits = readmatrix('../../data/ideal_10bit_sine.csv');

    % Run your tool
    output = YourTool(bits);

    % Create figure
    figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
    plot(output);
    title('Your Tool Example');

    % Save
    exportgraphics(gcf, fullfile(outDir, 'example.png'), 'Resolution', 300);
    close(gcf);
end
```

2. Add to `generate_all_figures.m`:
```matlab
examples = {
    ...
    'example_YourTool'
};
```

3. Run `generate_all_figures()`

## Canonical Datasets

See `../data/README.md` for dataset descriptions.

Key datasets:
- `ideal_10bit_sine.csv` - Perfect binary-weighted ADC
- `sar_12bit_redundancy.csv` - With redundancy bits
- `with_inl_dnl.csv` - Nonlinearity for INL/DNL demos
- `with_harmonic_distortion.csv` - HD2/HD3 for spectrum demos

## Best Practices

1. **Use 'Visible', 'off'** to avoid figure pop-ups:
```matlab
figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
```

2. **High resolution export**:
```matlab
exportgraphics(gcf, filename, 'Resolution', 300);
```

3. **Close figures** after saving:
```matlab
close(gcf);
```

4. **Consistent sizing**:
   - Single plot: `[100, 100, 1000, 600]`
   - Multi-panel: `[100, 100, 1200, 800]`

5. **Font sizes**:
   - Axes labels: `FontSize', 12`
   - Titles: `FontSize', 14`
   - Annotations: `FontSize', 11`

## Notes

- Figures are language-agnostic (look the same from MATLAB or Python)
- Documentation shows both MATLAB and Python code examples
- Only need to generate once (commit to git or regenerate as needed)
