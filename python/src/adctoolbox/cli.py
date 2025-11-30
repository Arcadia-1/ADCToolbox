import argparse
import shutil
import sys
from pathlib import Path
import adctoolbox

def copy_examples():
    """
    CLI Entry Point: Copies the 'examples' folder to the current working directory.
    """
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description="Copy ADCToolbox examples to your workspace.")
    parser.add_argument('dest', nargs='?', default='adctoolbox_examples', 
                        help="Destination directory name (default: adctoolbox_examples)")
    args = parser.parse_args()

    # 2. Locate source directory (inside the installed package)
    package_dir = Path(adctoolbox.__file__).parent
    src_dir = package_dir / "examples"

    if not src_dir.exists():
        print(f"[Error] Corrupted installation: Examples not found at {src_dir}")
        sys.exit(1)

    # 3. Locate destination directory (current working dir)
    dest_dir = Path.cwd() / args.dest

    if dest_dir.exists():
        print(f"[Error] Directory '{dest_dir.name}' already exists.")
        print("Please remove it or specify a different name.")
        sys.exit(1)

    # 4. Perform copy
    try:
        shutil.copytree(src_dir, dest_dir, 
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        
        print(f"\n[Success!] Examples copied to: [{dest_dir}]")
        print("-" * 50)
        print("Next steps:")
        print(f"  cd {dest_dir.name}")
        print("  python example_00_basic.py")
        print("-" * 50)
        
    except Exception as e:
        print(f"[Error] Copy failed: [{e}]")
        sys.exit(1)

if __name__ == "__main__":
    copy_examples()