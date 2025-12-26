"""Add timing instrumentation to signal generation examples."""
from pathlib import Path
import re

examples_dir = Path("d:/ADCToolbox/python/src/adctoolbox/examples/03_generate_signals")

files = [
    "exp_g01_generate_signal_demo.py",
    "exp_g03_sweep_quant_bits.py",
    "exp_g04_sweep_jitter_fin.py",
    "exp_g05_sweep_static_nonlin.py",
    "exp_g06_sweep_dynamic_nonlin.py",
    "exp_g07_sweep_interferences.py",
]

for filename in files:
    filepath = examples_dir / filename
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    content = filepath.read_text(encoding='utf-8')
    original_content = content

    # Step 1: Add import time and t_start at the very beginning (after docstring)
    # Find the docstring end
    docstring_match = re.search(r'^(\"\"\".*?\"\"\"\n\n)', content, re.DOTALL | re.MULTILINE)
    if docstring_match:
        docstring = docstring_match.group(1)
        # Add timing imports right after docstring
        content = content.replace(
            docstring,
            docstring + 'import time\nt_start = time.perf_counter()\n\n',
            1
        )
        print("✓ Added import timing")

    # Step 2: Add t_import after all imports (before first major comment or code)
    # Find first substantive line after imports (usually output_dir or params)
    import_end_patterns = [
        r'(from adctoolbox\.siggen import [^\n]+\n)(\n# )',
        r'(from adctoolbox\.siggen import [^\n]+\n)(\noutput_dir)',
        r'(from adctoolbox\.siggen import [^\n]+\n)(\n# Parameters)',
    ]

    for pattern in import_end_patterns:
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                r'\1\nt_import = time.perf_counter() - t_start\n\2',
                content,
                count=1
            )
            print("✓ Added import time capture")
            break

    # Step 3: Add preparation timing (before main loop or analyze_spectrum calls)
    # Look for patterns like "for" loops, analyze_spectrum calls, or result arrays
    prep_patterns = [
        (r'(t_import = time\.perf_counter\(\) - t_start\n\n)(# Setup|output_dir|# Parameters)', r'\1\nt_prep_start = time.perf_counter()\n\2'),
        (r'(t_import = time\.perf_counter\(\) - t_start\n\n)(N = )', r'\1\nt_prep_start = time.perf_counter()\n\2'),
    ]

    for old_pat, new_pat in prep_patterns:
        if re.search(old_pat, content):
            content = re.sub(old_pat, new_pat, content, count=1)
            print("✓ Added prep timing start")
            break

    # Step 4: Add timing report at the end (before plt.close() or at very end)
    # Find the pattern at the end
    end_patterns = [
        r'(print\(f"\[Save figure\][^\n]+\)\n)(plt\.close\(\)|$)',
        r'(plt\.savefig\([^\n]+\n)(plt\.close\(\)|$)',
    ]

    timing_report = '''
t_total = time.perf_counter() - t_start
print(f"\\n{'='*60}")
print(f"Timing Report:")
print(f"{'='*60}")
print(f"  Import time:      {t_import*1000:7.2f} ms")
print(f"  Total runtime:    {t_total*1000:7.2f} ms")
print(f"{'='*60}\\n")

'''

    for pattern in end_patterns:
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                r'\1' + timing_report + r'\2',
                content,
                count=1
            )
            print("✓ Added timing report")
            break

    # Write back if changed
    if content != original_content:
        filepath.write_text(content, encoding='utf-8')
        print(f"✅ Updated: {filename}")
    else:
        print(f"⚠️  No changes: {filename}")

print(f"\n{'='*60}")
print("Timing instrumentation complete!")
print(f"{'='*60}")
