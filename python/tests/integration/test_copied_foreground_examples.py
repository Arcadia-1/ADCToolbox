"""Exercise the documented copy-to-workspace workflow outside the package."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('number', [10, 11])
def test_copied_calibration_example_and_audit(tmp_path, number):
    examples = tmp_path/'examples'
    environment = {**os.environ, 'MPLBACKEND': 'Agg', 'OPENBLAS_NUM_THREADS': '1'}

    def run(*args):
        result = subprocess.run([sys.executable, *map(str, args)], cwd=tmp_path,
            env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr

    run('-m', 'adctoolbox.cli', examples)
    if number == 10:
        stem, audit = 'exp_cal10_foreground_bitweights', 'audit_foreground_bitweights'
        budget = ['--sar-records', '4', '--sar-validation-records', '2',
                  '--sar-size', '1024', '--sar-iterations', '3', '--sar-plants', 'binary']
    else:
        stem, audit = 'exp_cal11_combined_foreground', 'audit_combined_foreground'
        budget = ['--records', '4', '--validation-records', '2',
                  '--size', '1024', '--iterations', '3', '--architectures', 'binary']
    output = tmp_path/'output'
    folder = examples/'12_foreground'
    run(folder/(stem+'.py'), '--output', output, '--seeds', '11', *budget)
    configuration = json.loads((output/'configuration.json').read_text())
    copied_key = f'examples/12_foreground/{stem}.py'
    assert copied_key in configuration['source_sha256']
    for name, digest in configuration['source_sha256'].items():
        assert hashlib.sha256((output/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    audit_output = tmp_path/'audit.json'
    run(folder/(audit+'.py'), '--directory', output, '--output', audit_output)
    assert json.loads(audit_output.read_text())['status'] == 'passed'
