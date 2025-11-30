# ADCToolbox Examples Usage Guide

## Four Ways to Run Examples

After installing ADCToolbox, you have **4 different ways** to run the example scripts. Choose the method that works best for you!

---

## Method 1: CLI Commands (Recommended - Easiest)

After `pip install adctoolbox`, CLI commands are automatically installed and available system-wide.

```bash
# Just type the command name - works from any directory!
adctoolbox-quickstart                    # Quickstart: basic workflow
adctoolbox-example-sine-fit              # Common: sine wave fitting
adctoolbox-example-spec-plot             # AOUT: spectrum analysis
adctoolbox-example-calibration           # DOUT: foreground calibration
adctoolbox-example-workflow              # Workflows: complete analysis
```

**Advantages:**
- ✅ Shortest commands
- ✅ No need to remember module paths
- ✅ Works from any directory
- ✅ Tab completion (in most shells)

**Best for:** Quick exploration, beginners, everyday use

---

## Method 2: Python Module Path (Explicit)

Run examples as Python modules with full path specification.

```bash
python -m adctoolbox.examples.quickstart.example_00_basic_workflow
python -m adctoolbox.examples.common.example_00_sine_fit
python -m adctoolbox.examples.aout.example_00_spec_plot
python -m adctoolbox.examples.dout.example_00_fg_cal_sine
python -m adctoolbox.examples.workflows.example_00_complete_adc_analysis
```

**Advantages:**
- ✅ Explicit - clear what's being run
- ✅ Works without CLI entry points
- ✅ Portable across systems

**Best for:** Scripts, automation, when you want explicit paths

---

## Method 3: Python Import (Interactive)

Run examples from within a Python shell or script.

```python
# In Python shell or script
from adctoolbox.examples.quickstart import example_00_basic_workflow
# Script executes automatically on import

# Or explicitly call main()
from adctoolbox.examples.common import example_00_sine_fit
example_00_sine_fit.main()
```

**Advantages:**
- ✅ Works in Jupyter notebooks
- ✅ Can be integrated into larger scripts
- ✅ Access to script variables (if needed)

**Best for:** Jupyter notebooks, interactive exploration, integration into other code

---

## Method 4: Direct Script Execution (Development)

Run example files directly from the file system.

```bash
# Navigate to examples directory
cd /path/to/adctoolbox/python/src/adctoolbox/examples

# Run any example directly
python common/example_00_sine_fit.py
python aout/example_00_spec_plot.py
python dout/example_00_fg_cal_sine.py
python quickstart/example_00_basic_workflow.py
python workflows/example_00_complete_adc_analysis.py
```

**Advantages:**
- ✅ Direct file access
- ✅ Easy to edit and test changes
- ✅ No installation needed (for development)

**Best for:** Development, debugging, modifying examples

---

## Recommended Learning Path

**For beginners, we recommend Method 1 (CLI commands):**

1. **Start with quickstart:**
   ```bash
   adctoolbox-quickstart
   ```
   Complete workflow showing all major capabilities

2. **Learn fundamental utilities:**
   ```bash
   adctoolbox-example-sine-fit
   ```
   Sine wave parameter extraction

3. **Explore spectrum analysis:**
   ```bash
   adctoolbox-example-spec-plot
   ```
   FFT-based ADC performance metrics

4. **Try digital calibration:**
   ```bash
   adctoolbox-example-calibration
   ```
   Foreground calibration for SAR ADCs

5. **Advanced comprehensive workflow:**
   ```bash
   adctoolbox-example-workflow
   ```
   Complete multi-angle ADC analysis

---

## Quick Reference Table

| Method | Command Example | Best For |
|--------|----------------|----------|
| **CLI** | `adctoolbox-quickstart` | Quick use, beginners |
| **Module** | `python -m adctoolbox.examples.quickstart.example_00_basic_workflow` | Scripts, automation |
| **Import** | `from adctoolbox.examples.quickstart import example_00_basic_workflow` | Jupyter, integration |
| **Direct** | `python quickstart/example_00_basic_workflow.py` | Development, editing |

---

## Output Files

All examples save output figures to:
```
adctoolbox/examples/output/
```

Each example prints the full absolute path of every saved file:
```
[save]->[D:\path\to\output\figure.png]
```

---

## Reading the Example Code

**Important:** CLI commands execute the scripts but don't show the source code. To **read and learn from the code**:

### Option 1: View on GitHub (No installation needed)
```
https://github.com/Arcadia-1/ADCToolbox/tree/main/python/src/adctoolbox/examples
```

### Option 2: Find installed files
```bash
# Find where adctoolbox is installed
python -c "import adctoolbox; print(adctoolbox.__file__)"

# Navigate to examples directory
cd /path/to/site-packages/adctoolbox/examples

# Open files in your editor
code common/example_00_sine_fit.py
```

### Option 3: Clone the repository (Best for learning)
```bash
git clone https://github.com/Arcadia-1/ADCToolbox.git
cd ADCToolbox/python/src/adctoolbox/examples

# Open in your favorite editor
code .                                    # VS Code
vim common/example_00_sine_fit.py        # Vim
```

### Option 4: Use Python's inspect module
```python
import inspect
from adctoolbox.examples.common import example_00_sine_fit

# Get the source code as a string
source = inspect.getsource(example_00_sine_fit)
print(source)

# Or get the file path
file_path = inspect.getfile(example_00_sine_fit)
print(f"Source file: {file_path}")
```

**Recommendation:** Clone the repository to read, modify, and experiment with examples!

---

## Need Help?

- **Examples README:** See `python/src/adctoolbox/examples/README.md` for detailed descriptions
- **Main README:** See root `README.md` for overall project documentation
- **Issues:** Report problems at https://github.com/Arcadia-1/ADCToolbox/issues
