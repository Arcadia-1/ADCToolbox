# ADCToolbox - MATLAB

A comprehensive toolbox for ADC (Analog-to-Digital Converter) test and debug.

## Structure

- `src/` - Source code files
- `toolbox/` - Toolbox packaging files (.mltbx)
- `resources/` - Project resources and configuration files

## Installation

1. Add the toolbox to MATLAB path:
   ```matlab
   addpath(genpath('path/to/ADCToolbox/matlab/src'))
   ```

2. Or install the toolbox package:
   - Double-click `toolbox/ADCToolbox_0v12.mltbx` to install
   - The toolbox will be automatically added to MATLAB path
   - You can also download this toolbox in the Matlab add-ons

## Usage

See individual function help for detailed usage:
```matlab
help functionName
```

## Main Functions

- `sineFit` - Sine wave fitting
- `specPlot` - Spectrum analysis
- `INLsine` - INL/DNL analysis
- `FGCalSine` - Foreground calibration
- And more...
