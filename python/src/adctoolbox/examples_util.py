"""
Utility functions for accessing ADCToolbox examples.

Examples are NOT included in the pip package to keep it lightweight.
All examples are available in the GitHub repository.

Repository: https://github.com/Arcadia-1/ADCToolbox
Examples: https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples
"""

import os
import sys
from pathlib import Path

EXAMPLES_URL = "https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples"
REPO_URL = "https://github.com/Arcadia-1/ADCToolbox"


def get_examples_dir():
    """
    Get the path to the examples directory (development mode only).

    This only works if you're running from the source repository.
    If you installed via pip, examples are NOT included in the package.

    Returns:
        Path: Path to examples directory (development mode only)

    Raises:
        FileNotFoundError: If not in development mode
    """
    # Only check source tree (for development)
    source_path = Path(__file__).parent.parent.parent.parent / "examples"

    if source_path.exists() and source_path.is_dir():
        return source_path.resolve()

    # If not found, provide helpful message
    raise FileNotFoundError(
        "Examples are not included in pip installation.\n\n"
        "Examples are available on GitHub:\n"
        f"  {EXAMPLES_URL}\n\n"
        "To access examples:\n"
        "  1. View online: https://github.com/Arcadia-1/ADCToolbox/tree/main/python/examples\n"
        "  2. Clone repository: git clone https://github.com/Arcadia-1/ADCToolbox.git\n"
        "  3. Download examples folder directly from GitHub"
    )


def list_examples():
    """
    List all available examples grouped by category.

    Returns:
        dict: Dictionary of categories and their example files
    """
    try:
        examples_dir = get_examples_dir()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return {}

    categories = {
        'quickstart': 'Quick Start',
        'aout': 'Analog Output Analysis',
        'dout': 'Digital Output Calibration',
        'common': 'Common Utilities',
        'workflows': 'Complete Workflows',
        'data_generation': 'Data Generation'
    }

    examples = {}

    for category, title in categories.items():
        category_path = examples_dir / category
        if category_path.exists():
            py_files = sorted(category_path.glob("*.py"))
            if py_files:
                examples[title] = [
                    {
                        'name': f.stem,
                        'path': str(f),
                        'category': category
                    }
                    for f in py_files
                ]

    return examples


def print_examples():
    """
    Print information about accessing examples.
    """
    print("\n" + "=" * 70)
    print("ADCToolbox Examples")
    print("=" * 70)

    # Try to get examples dir (only works in dev mode)
    try:
        examples_dir = get_examples_dir()
        print(f"\n✓ Running in development mode")
        print(f"  Examples location: {examples_dir}")

        examples = list_examples()
        if examples:
            print("\nAvailable examples:")
            for category, files in examples.items():
                print(f"\n  {category}:")
                for file_info in files:
                    print(f"    • {file_info['name']}.py")
    except FileNotFoundError as e:
        print("\n✗ Examples not included in pip installation")
        print("\n" + str(e))
        return

    print("\n" + "=" * 70)


def copy_examples_to(destination):
    """
    Copy all examples to a destination directory (development mode only).

    This only works if running from source repository.
    If installed via pip, use git clone instead.

    Args:
        destination (str or Path): Destination directory

    Returns:
        Path: Path to copied examples, or None if not available
    """
    import shutil

    try:
        examples_dir = get_examples_dir()
    except FileNotFoundError as e:
        print(str(e))
        return None

    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Copy entire examples directory
    dest_examples = dest / "adctoolbox_examples"

    if dest_examples.exists():
        print(f"Warning: {dest_examples} already exists. Overwriting...")
        shutil.rmtree(dest_examples)

    shutil.copytree(examples_dir, dest_examples)

    print(f"Examples copied to: {dest_examples}")
    print(f"\nTo run examples:")
    print(f"  cd {dest_examples}")
    print(f"  python quickstart/basic_workflow.py")

    return dest_examples


def main():
    """
    Command-line interface for examples utility.

    Usage:
        python -m adctoolbox.examples_util          # Show how to access examples
        python -m adctoolbox.examples_util list     # List examples (dev mode only)
        python -m adctoolbox.examples_util copy .   # Copy examples (dev mode only)
    """
    import sys

    if len(sys.argv) == 1 or sys.argv[1] == 'list':
        print_examples()
    elif sys.argv[1] == 'copy':
        if len(sys.argv) < 3:
            print("Usage: python -m adctoolbox.examples_util copy <destination>")
            print("\nNote: This only works in development mode.")
            print("If installed via pip, clone the repository instead:")
            print(f"  git clone {REPO_URL}")
            sys.exit(1)
        result = copy_examples_to(sys.argv[2])
        if result is None:
            sys.exit(1)
    elif sys.argv[1] == 'dir':
        try:
            print(get_examples_dir())
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    elif sys.argv[1] == 'url':
        print(f"Examples URL: {EXAMPLES_URL}")
        print(f"Repository URL: {REPO_URL}")
    else:
        print("Unknown command. Available commands: list, copy, dir, url")
        print("\nExamples are available on GitHub:")
        print(f"  {EXAMPLES_URL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
