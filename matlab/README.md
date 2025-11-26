# ADCToolbox - MATLAB

A comprehensive toolbox for ADC (Analog-to-Digital Converter) testing and debugging, providing functions for spectral analysis, calibration, linearity testing, and signal processing.

## Structure

- `src/` - Source code files (MATLAB functions)
- `toolbox/` - Toolbox packaging files (.mltbx)
- `setupLib.m` - Setup script for adding toolbox to MATLAB path

## Installation

### Option 1: Install Toolbox Package (Recommended)
1. Double-click `toolbox/ADCToolbox_0v12.mltbx` to install
2. The toolbox will be automatically added to MATLAB path
3. You can also download this toolbox from MATLAB Add-Ons

### Option 2: Add to Path Manually
```matlab
addpath(genpath('path/to/ADCToolbox/matlab/src'))
savepath  % Optional: save path for future sessions
```

## Usage

See individual function help for detailed usage:
```matlab
help functionName
```

## Main Functions

### Spectral Analysis
- `specPlot` - Comprehensive spectrum analysis with ENOB, SNDR, SFDR, SNR, THD calculations
- `specPlotPhase` - Spectrum analysis with phase information
- `findFin` - Find input frequency from ADC data
- `findBin` - Find FFT bin for coherent sampling

### Calibration & Correction
- `FGCalSine` - Foreground calibration using sinewave input
- `cap2weight` - Convert capacitor values to bit weights for SAR ADCs

### Linearity Analysis
- `INLsine` - INL/DNL analysis using sine wave histogram method
- `errHistSine` - Error histogram analysis for sine wave testing

### Signal Processing
- `sineFit` - Sine wave fitting with frequency estimation
- `tomDecomp` - Thompson decomposition for signal analysis
- `NTFAnalyzer` - Noise transfer function analysis for delta-sigma ADCs

### Utility Functions
- `alias` - Calculate frequency after aliasing
- `bitInBand` - Bits-wise filter function to extract the ADC data within specified frequency bands (mainly for noise-shaping ADC calibration)
- `overflowChk` - Check for ADC overflow on every segments  
