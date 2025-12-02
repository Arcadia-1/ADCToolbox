#!/usr/bin/env python3
"""Update all test files to use tests/config.py"""

from pathlib import Path
import re

# Map test patterns to config names
PATTERN_MAP = {
    'sinewave_*.csv': 'AOUT',
    'dout_*.csv': 'DOUT',
    'jitter_sweep_*.csv': 'JITTER',
}

def update_test_file(filepath):
    """Update a test file to use config.py"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Add import if not present
    if 'from tests import config' not in content:
        content = content.replace(
            'from tests.unit._runner import run_unit_test_batch',
            'from tests.unit._runner import run_unit_test_batch\nfrom tests import config'
        )

    # Replace hardcoded paths with config references
    for pattern, config_name in PATTERN_MAP.items():
        # Match: input_subpath="...", ... file_pattern="pattern"
        old_pattern = re.compile(
            r'input_subpath\s*=\s*["\']([^"\']+)["\'],\s*'
            r'test_module_name\s*=\s*([^,]+),\s*'
            r'file_pattern\s*=\s*["\']' + re.escape(pattern) + r'["\']'
        )

        replacement = (
            r'input_subpath=config.' + config_name + r"['input_path'], "
            r'test_module_name=\2, '
            r'file_pattern=config.' + config_name + r"['file_pattern']"
        )

        content = old_pattern.sub(replacement, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

# Update all test files
test_dir = Path('D:/ADCToolbox/python/tests/unit')
updated = 0

for test_file in sorted(test_dir.glob('test_*.py')):
    if update_test_file(test_file):
        print(f'Updated: {test_file.name}')
        updated += 1
    else:
        print(f'No change: {test_file.name}')

print(f'\nUpdated {updated} files')
print('\nNow edit python/tests/config.py to change test paths!')
