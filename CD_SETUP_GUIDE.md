# CD (Continuous Deployment) Setup Guide

## Overview

Your CD workflow will automatically publish ADCToolbox to PyPI whenever you create a new version tag. This means users can install the latest version with `pip install adctoolbox`.

## Prerequisites

### 1. Create a PyPI Account
1. Go to https://pypi.org/account/register/
2. Verify your email
3. (Optional but recommended) Set up 2FA

### 2. Create a PyPI API Token
1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `GitHub Actions - ADCToolbox`
4. Scope: "Entire account" (or limit to project after first upload)
5. **Copy the token** (you'll only see it once!)

### 3. Add Token to GitHub Secrets
1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI token (starts with `pypi-`)
6. Click "Add secret"

## How the CD Workflow Works

```
┌─────────────────────────────────────────────────────────┐
│  Developer Action: Create and Push Version Tag         │
│  $ git tag v0.2.2                                       │
│  $ git push origin v0.2.2                               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  GitHub Actions CD Workflow Triggers                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Step 1: Build Package                                  │
│  - Builds wheel (.whl)                                  │
│  - Builds source distribution (.tar.gz)                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Step 2: Check Package                                  │
│  - Validates metadata                                   │
│  - Checks for common errors                             │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Step 3: Publish to PyPI                                │
│  - Uploads to https://pypi.org/project/adctoolbox/      │
│  - Uses secure API token                                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Step 4: Create GitHub Release                          │
│  - Creates release with tag                             │
│  - Attaches .whl and .tar.gz files                      │
│  - Auto-generates release notes from commits            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Users can install:                                     │
│  $ pip install adctoolbox                               │
│  $ pip install adctoolbox==0.2.2                        │
│  $ pip install --upgrade adctoolbox                     │
└─────────────────────────────────────────────────────────┘
```

## Release Process

### Step 1: Update Version Number

Edit `python/pyproject.toml`:
```toml
[project]
name = "adctoolbox"
version = "0.2.2"  # ← Update this
```

### Step 2: Update CHANGELOG (optional but recommended)

Create/update `CHANGELOG.md`:
```markdown
# Changelog

## [0.2.2] - 2025-12-06

### Added
- 21 ready-to-run examples (b01-b04, a01-a14, d01-d05)
- CI workflow testing examples

### Fixed
- spec_plot return value in exp_b02_spectrum.py
- inl_dnl_from_sine data clipping

### Changed
- Updated documentation with 3-step install process
```

### Step 3: Commit Changes

```bash
git add python/pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.2.2"
git push origin main
```

### Step 4: Create and Push Tag

```bash
# Create annotated tag (recommended)
git tag -a v0.2.2 -m "Release v0.2.2 - 21 examples complete"

# Or simple tag
git tag v0.2.2

# Push the tag (this triggers CD!)
git push origin v0.2.2
```

### Step 5: Monitor Deployment

1. Go to GitHub → Actions tab
2. Watch the "CD - Publish to PyPI" workflow
3. After ~2 minutes, check:
   - PyPI: https://pypi.org/project/adctoolbox/
   - GitHub Releases: https://github.com/yourusername/ADCToolbox/releases

## Testing Before Publishing

### Test Locally

```bash
cd python

# Build package
python -m build

# Check package
twine check dist/*

# Install locally to test
pip install dist/adctoolbox-0.2.2-py3-none-any.whl

# Test it works
python -c "from adctoolbox import spec_plot; print('Success!')"
adctoolbox-get-examples
```

### Test on TestPyPI (Recommended for First Time)

1. Create account at https://test.pypi.org/
2. Create API token
3. Upload test:
   ```bash
   twine upload --repository testpypi dist/*
   ```
4. Install from test:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ adctoolbox
   ```

## Version Numbering

Use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.3.0): New features, backwards compatible
- **PATCH** (0.2.2): Bug fixes, backwards compatible

Examples:
- `v0.2.1` → `v0.2.2`: Bug fix (spec_plot return values)
- `v0.2.2` → `v0.3.0`: New features (add NTF analyzer)
- `v0.3.0` → `v1.0.0`: Major release (API redesign)

## Troubleshooting

### Error: "File already exists"
- You already published this version to PyPI
- PyPI doesn't allow re-uploading same version
- Solution: Bump version number and create new tag

### Error: "Invalid or non-existent authentication"
- Check `PYPI_API_TOKEN` secret is set correctly
- Verify token hasn't expired
- Ensure token has upload permissions

### Error: "Package name already taken"
- Someone else registered `adctoolbox` on PyPI
- Solution: Choose different name in `pyproject.toml`
- Or claim the name if you own it

### Error: "README rendering failed"
- PyPI couldn't render your README.md
- Solution: Validate with `twine check dist/*`
- Check markdown syntax

## Managing Releases

### Delete a Tag (if you made a mistake)
```bash
# Delete local tag
git tag -d v0.2.2

# Delete remote tag
git push origin :refs/tags/v0.2.2
```

**Note**: Can't delete from PyPI once published! Only option is to "yank" the release.

### Yank a Release (emergency only)
```bash
# On PyPI website, or using twine
pip install twine
twine upload --repository pypi --skip-existing dist/*
```

Then on PyPI website → Manage → Options → "Yank this release"

## First-Time Checklist

Before publishing v0.2.2 for the first time:

- [ ] PyPI account created and verified
- [ ] PyPI API token created
- [ ] GitHub secret `PYPI_API_TOKEN` added
- [ ] Test build works: `python -m build`
- [ ] Test package works: `twine check dist/*`
- [ ] Test install works: `pip install dist/*.whl`
- [ ] Test examples work: `adctoolbox-get-examples`
- [ ] README.md looks good (will be PyPI description)
- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG.md updated (optional)
- [ ] All changes committed to main branch
- [ ] Tag created: `git tag v0.2.2`
- [ ] Tag pushed: `git push origin v0.2.2`

## After Publishing

### Verify Installation
```bash
# Uninstall local version
pip uninstall adctoolbox

# Install from PyPI
pip install adctoolbox

# Test it works
python -c "from adctoolbox import spec_plot, find_bin; print('Success!')"
adctoolbox-get-examples
```

### Update Documentation
Add installation badge to README.md:
```markdown
[![PyPI version](https://badge.fury.io/py/adctoolbox.svg)](https://badge.fury.io/py/adctoolbox)
[![Downloads](https://pepy.tech/badge/adctoolbox)](https://pepy.tech/project/adctoolbox)
```

## Advanced: Pre-release Versions

For beta/alpha releases:
```bash
# Update version in pyproject.toml
version = "0.3.0b1"  # Beta 1
version = "0.3.0rc1" # Release candidate 1

# Tag and publish
git tag v0.3.0b1
git push origin v0.3.0b1

# Users install with
pip install --pre adctoolbox
```

## Summary

Your complete CI/CD pipeline:

1. **CI (on every commit)**: Tests basic examples automatically
2. **CD (on version tag)**: Builds and publishes to PyPI automatically

This is production-grade automation used by major Python projects!

## Quick Reference

```bash
# Release workflow
vim python/pyproject.toml    # Update version
git commit -am "Bump version to X.Y.Z"
git push
git tag vX.Y.Z
git push origin vX.Y.Z        # Triggers CD!

# Wait 2 minutes, then:
pip install --upgrade adctoolbox
```

Done! 🚀
