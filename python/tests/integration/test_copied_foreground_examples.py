"""Exercise official copied foreground scripts against a real wheel install."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest


def _run(command, *, cwd, env):
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                               text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


@pytest.fixture(scope='module')
def installed_examples(tmp_path_factory):
    root = tmp_path_factory.mktemp('installed_foreground')
    project = Path(__file__).resolve().parents[2]
    build = root/'project'
    build.mkdir()
    shutil.copytree(project/'src', build/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info'))
    for name in ['pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in']:
        shutil.copy2(project/name, build/name)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    temporary = root/'tmp'
    temporary.mkdir()
    env.update(TEMP=str(temporary), TMP=str(temporary))
    env.update(MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    wheels = root/'wheels'
    _run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
          '--no-index', '--wheel-dir', str(wheels), str(build)], cwd=root, env=env)
    environment = root/'venv'
    # Reuse NumPy/SciPy/Matplotlib while installing the toolbox itself from the wheel.
    venv.EnvBuilder(with_pip=False, system_site_packages=True).create(environment)
    scripts = environment/('Scripts' if os.name == 'nt' else 'bin')
    python = scripts/('python.exe' if os.name == 'nt' else 'python')
    _run([str(python), '-m', 'pip', 'install', '--no-deps', '--no-index', '--ignore-installed',
          str(next(wheels.glob('adctoolbox-*.whl')))], cwd=root, env=env)
    package = Path(_run([str(python), '-I', '-c',
        'import adctoolbox; print(adctoolbox.__file__)'], cwd=root, env=env).stdout.strip()).parent
    assert package.is_relative_to(environment)
    examples = root/'copied_examples'
    cli = scripts/('adctoolbox-get-examples.exe' if os.name == 'nt' else 'adctoolbox-get-examples')
    _run([str(cli), str(examples)], cwd=root, env=env)
    return root, python, examples/'12_foreground', env


@pytest.mark.parametrize('example,audit,budget,rows', [
    ('exp_cal10_foreground_bitweights', 'audit_foreground_bitweights',
     ['--sar-plants', 'binary', '--sar-records', '4', '--sar-validation-records', '2',
      '--sar-size', '1024', '--sar-iterations', '10'], 119),
    ('exp_cal11_combined_foreground', 'audit_combined_foreground',
     ['--architectures', 'binary', '--records', '4', '--validation-records', '2',
      '--size', '1024', '--iterations', '10'], 162),
])
def test_copied_wheel_example_and_frozen_audit(installed_examples, example, audit, budget, rows):
    root, python, examples, env = installed_examples
    output = root/example
    script = examples/(example+'.py')
    # A harmless local edit distinguishes the executed copy from the wheel source.
    with script.open('a', encoding='utf-8') as stream:
        stream.write('\n# Copied-example provenance regression.\n')
    _run([str(python), str(script),
          *budget, '--seeds', '11', '--output', str(output)], cwd=examples, env=env)
    config = json.loads((output/'configuration.json').read_text())
    hashes = config['source_sha256']
    assert 'calibration/foreground.py' in hashes
    assert f'examples/12_foreground/{example}.py' in hashes
    assert 'examples/12_foreground/_source_snapshot.py' in hashes
    for name, digest in hashes.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert hashlib.sha256((output/'source_snapshot'/path).read_bytes()).hexdigest() == digest
    # Verify snapshots contain the copied sources actually executed, not installed originals.
    for source in examples.glob('*.py'):
        key = 'examples/12_foreground/'+source.name
        if key in hashes:
            assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[key]
    result_path = root/(audit+'.json')
    _run([str(python), str(examples/(audit+'.py')), '--directory', str(output),
          '--output', str(result_path)], cwd=examples, env=env)
    result = json.loads(result_path.read_text())
    assert result['status'] == 'passed'
    assert result['frozen_rows_replayed'] == rows
