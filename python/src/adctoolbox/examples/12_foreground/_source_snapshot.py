"""Save package and copied-example sources using stable package-relative names."""
import hashlib
from pathlib import Path


def snapshot_sources(sources, *, package, example_dir, output):
    package = Path(package).resolve()
    example_dir = Path(example_dir).resolve()
    hashes = {}
    # Include this helper so the snapshot records how source names were assigned.
    for source in sorted({Path(p).resolve() for p in [*sources, __file__]}):
        try:
            relative = source.relative_to(package)
        except ValueError:
            # The official CLI preserves the examples subdirectory layout.
            relative = Path('examples')/example_dir.name/source.relative_to(example_dir)
        content = source.read_bytes()
        name = relative.as_posix()
        digest = hashlib.sha256(content).hexdigest()
        if name in hashes and hashes[name] != digest:
            raise ValueError(f'Conflicting source snapshot: {name}')
        target = Path(output)/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = digest
    return hashes
