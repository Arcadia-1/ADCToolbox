# Release v0.4.0 Checklist

**Release Date**: 2025-12-18
**Release Type**: Documentation Release
**Status**: Ready for Publishing

---

## ✅ Completed Steps

### 1. Version Numbers Updated
- [x] `python/src/adctoolbox/__init__.py` → **0.4.0**
- [x] `python/pyproject.toml` → **dynamic versioning from __init__.py**
- [x] `python/docs/source/conf.py` → **0.4.0**

### 2. Changelogs Updated
- [x] `CHANGELOG.md` → Added v0.4.0 section with complete changes
- [x] `python/docs/source/changelog.rst` → Added v0.4.0 section

### 3. Documentation Built
- [x] Sphinx documentation successfully built
- [x] Location: `python/docs/build/html/`
- [x] Build status: **13 warnings (non-critical), build succeeded**

### 4. ReadTheDocs Configuration
- [x] Created `.readthedocs.yaml` in project root
- [x] Configuration points to `python/docs/source/conf.py`
- [x] Python dependencies configured with docs extras

### 5. README Updated
- [x] Added documentation badges
- [x] Added documentation section with links
- [x] Links to ReadTheDocs (will be live after setup)

### 6. Cleanup
- [x] Deleted obsolete `python/src/__init__.py` file

---

## 📋 Next Steps for Publishing

### Step 1: Commit All Changes

```bash
cd d:\ADCToolbox

# Check what's changed
git status

# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Release v0.4.0: Documentation overhaul with Sphinx and ReadTheDocs

- Updated version to 0.4.0 across all files
- Complete documentation overhaul with 15 algorithm guides
- Updated CHANGELOG.md and changelog.rst
- Added ReadTheDocs configuration (.readthedocs.yaml)
- Updated README with documentation links and badges
- Built Sphinx HTML documentation
- Fixed dynamic versioning in pyproject.toml
- Removed obsolete src/__init__.py"
```

### Step 2: Create Git Tag

```bash
# Create annotated tag
git tag -a v0.4.0 -m "Release v0.4.0: Complete Sphinx documentation with algorithm guides"

# Verify tag was created
git tag -l
```

### Step 3: Push to GitHub

```bash
# Push commits
git push origin main

# Push tags
git push origin v0.4.0
```

### Step 4: Set Up ReadTheDocs (One-Time Setup)

1. **Sign up at ReadTheDocs**: https://readthedocs.org/
   - Use your GitHub account to sign in

2. **Import Your Project**:
   - Go to: https://readthedocs.org/dashboard/
   - Click "Import a Project"
   - Select your ADCToolbox repository
   - ReadTheDocs will auto-detect `.readthedocs.yaml`

3. **Configure Build**:
   - Default branch: `main`
   - Documentation type: Sphinx
   - The `.readthedocs.yaml` file handles the rest

4. **Build Documentation**:
   - ReadTheDocs will automatically build on every push
   - First build may take 2-3 minutes
   - Your docs will be live at: `https://adctoolbox.readthedocs.io/`

### Step 5: PyPI Publishing (Your CD Pipeline)

Since you mentioned using CD to publish to PyPI, your CI/CD pipeline should:

1. **Detect the tag** `v0.4.0`
2. **Build the package**:
   ```bash
   cd python
   python -m build
   ```
3. **Publish to PyPI**:
   ```bash
   twine upload dist/*
   ```

**Or manually (if needed)**:

```bash
cd d:\ADCToolbox\python

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build package
python -m build

# Verify package
twine check dist/*

# Upload to PyPI (will prompt for credentials or use API token)
twine upload dist/*
```

### Step 6: Create GitHub Release (Optional but Recommended)

1. Go to: https://github.com/your-username/ADCToolbox/releases
2. Click "Draft a new release"
3. Choose tag: `v0.4.0`
4. Release title: **"ADCToolbox v0.4.0 - Documentation Release"**
5. Description (copy from CHANGELOG.md):

```markdown
**Documentation Release** - Complete Sphinx documentation overhaul with algorithm guides.

### Added
- **Complete Documentation Overhaul**:
  - 15 detailed algorithm documentation pages with Python API
  - Updated installation guide emphasizing `adctoolbox-get-examples`
  - Enhanced quickstart guide with learning path
  - All API reference docs updated to Python snake_case naming

### Changed
- **Documentation Structure**:
  - Installation guide shortened, git clone moved to bottom
  - Quickstart restructured to start with basic examples (exp_b01, exp_b02, then exp_s01)
  - Used actual code from examples instead of synthetic snippets
  - Emphasized "Learning with Examples" throughout documentation

### Removed
- Deleted 13 obsolete MATLAB-named algorithm documentation files
- Removed obsolete `src/__init__.py` file

### Fixed
- Version number synchronization across all files
- Dynamic versioning in `pyproject.toml`
- Documentation links and references updated to v0.4.0

### Documentation
📚 Full documentation now live at: https://adctoolbox.readthedocs.io/
```

6. Attach files (optional):
   - Upload `dist/adctoolbox-0.4.0-py3-none-any.whl`
   - Upload `dist/adctoolbox-0.4.0.tar.gz`

7. Click "Publish release"

### Step 7: Verify Everything Works

After publishing:

```bash
# Wait a few minutes for PyPI to propagate

# Test installation in a fresh environment
pip install --upgrade adctoolbox

# Verify version
python -c "import adctoolbox; print(adctoolbox.__version__)"
# Should print: 0.4.0

# Test getting examples
cd /tmp/test_adctoolbox
adctoolbox-get-examples

# Test running an example
cd adctoolbox_examples/02_spectrum
python exp_s01_analyze_spectrum_simplest.py

# Check documentation
# Visit: https://adctoolbox.readthedocs.io/
```

---

## 📊 Summary of Changes in v0.4.0

### Files Modified (11 files)
1. `python/src/adctoolbox/__init__.py` - Version and docstring
2. `python/pyproject.toml` - Fixed dynamic versioning
3. `python/docs/source/conf.py` - Version to 0.4.0
4. `CHANGELOG.md` - Added v0.4.0 entry
5. `python/docs/source/changelog.rst` - Added v0.4.0 entry
6. `README.md` - Added documentation section and badges
7. `.readthedocs.yaml` - Created ReadTheDocs config
8. `project_log.md` - Updated with v0.3.0 documentation session

### Files Deleted (1 file)
1. `python/src/__init__.py` - Obsolete namespace marker

### Documentation Built
- 29 source files processed
- 35 HTML pages generated
- Location: `python/docs/build/html/`
- Build time: ~30 seconds
- Warnings: 13 (non-critical, missing cross-references)

### Version Consistency Check
All version references now point to **0.4.0**:
- ✅ `__version__` in `__init__.py`
- ✅ Dynamic version in `pyproject.toml`
- ✅ Sphinx `conf.py` version and release
- ✅ CHANGELOG.md latest version
- ✅ changelog.rst latest version

---

## 🔍 Important Notes

### About ReadTheDocs
- **Free for open source**: ReadTheDocs is free for public repositories
- **Automatic builds**: Builds automatically on every push to main
- **Version support**: Can host multiple versions (stable, latest, v0.3.0, v0.4.0, etc.)
- **Search support**: Built-in search across all documentation
- **PDF/EPUB**: Can generate downloadable PDF and EPUB formats

### About PyPI Publishing
- **Version must be unique**: Once you publish 0.4.0, you cannot re-upload it
- **Test first**: Always test on TestPyPI before publishing to production PyPI
- **API tokens**: Recommended over username/password for security
- **Wheel and sdist**: `python -m build` creates both .whl and .tar.gz

### Documentation Links
After ReadTheDocs is set up, your documentation will be available at:
- Latest: https://adctoolbox.readthedocs.io/en/latest/
- v0.4.0: https://adctoolbox.readthedocs.io/en/v0.4.0/
- Stable: https://adctoolbox.readthedocs.io/en/stable/

---

## ✅ Ready to Publish!

All preparation is complete. You can now:
1. Commit the changes
2. Create the tag
3. Push to GitHub
4. Let your CD pipeline publish to PyPI
5. Set up ReadTheDocs (one-time)
6. Create GitHub release (optional)

**Estimated time**: 15-20 minutes for the entire process.

---

## 🆘 Troubleshooting

### If ReadTheDocs build fails:
- Check build logs at: https://readthedocs.org/projects/adctoolbox/builds/
- Most common issues:
  - Missing dependencies: Add to `pyproject.toml` under `[project.optional-dependencies] docs`
  - Import errors: Ensure all imports in docs work
  - Configuration errors: Validate `.readthedocs.yaml` syntax

### If PyPI upload fails:
- Verify package builds: `python -m build && twine check dist/*`
- Ensure version is unique (not already published)
- Check PyPI credentials/token
- Try TestPyPI first: `twine upload --repository testpypi dist/*`

### If documentation links are broken:
- Wait 5-10 minutes after ReadTheDocs setup
- Badges may show "unknown" until first build completes
- Update badge URLs if project name differs

---

**Last Updated**: 2025-12-18
**Prepared By**: Claude Code
**Release Manager**: [Your Name]
