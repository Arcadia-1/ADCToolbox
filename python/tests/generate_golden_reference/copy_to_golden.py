"""Copy Python outputs from test_output/ to test_reference/"""

import shutil
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent.parent
test_output = project_root / 'test_output'
test_reference = project_root / 'test_reference'

print('Copying Python outputs to test_reference/...')

# Find all Python output files
for dataset_dir in test_output.iterdir():
    if not dataset_dir.is_dir() or dataset_dir.name.startswith('.'):
        continue

    for test_dir in dataset_dir.iterdir():
        if not test_dir.is_dir() or test_dir.name.startswith('.'):
            continue

        # Find Python files
        python_files = list(test_dir.glob('*_python.csv')) + \
                      list(test_dir.glob('*_python.png'))

        if python_files:
            # Create destination
            dst_dir = test_reference / dataset_dir.name / test_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy files
            for src_file in python_files:
                dst_file = dst_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                print(f'  {dataset_dir.name}/{test_dir.name}/{src_file.name}')

print('Done!')
