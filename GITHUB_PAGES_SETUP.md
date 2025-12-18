# GitHub Pages Setup for ADCToolbox Documentation

## ✅ What's Been Configured

Your Sphinx documentation is now set up to deploy automatically to GitHub Pages using GitHub Actions.

### Files Created/Modified:

1. **`.github/workflows/docs.yml`** - GitHub Actions workflow
   - Triggers on: push to main, version tags, manual dispatch
   - Builds Sphinx HTML documentation
   - Deploys to GitHub Pages automatically

2. **`README.md`** - Updated with GitHub Pages links
   - Documentation URL: https://arcadia-1.github.io/ADCToolbox/
   - All internal links point to GitHub Pages

3. **Sphinx docs built** in `python/docs/build/html/`
   - 29 source files, 35 HTML pages
   - Ready for deployment

---

## 🚀 How to Deploy (One-Time Setup)

### Step 1: Enable GitHub Pages in Repository Settings

1. Go to your repository settings:
   ```
   https://github.com/Arcadia-1/ADCToolbox/settings/pages
   ```

2. Under **"Build and deployment"**:
   - **Source**: Select **"GitHub Actions"**
   - (This is the modern method, no need for gh-pages branch)

3. Click **Save**

That's it! GitHub Pages is now enabled.

### Step 2: Push Your Changes

```bash
cd d:\ADCToolbox

# Stage all changes
git add .

# Commit
git commit -m "Release v0.4.0: Documentation with GitHub Pages deployment

- Updated version to 0.4.0
- Added GitHub Actions workflow for docs deployment
- Updated README with GitHub Pages links
- Complete Sphinx documentation overhaul"

# Create tag
git tag -a v0.4.0 -m "Release v0.4.0: Complete Sphinx documentation"

# Push everything
git push origin main
git push origin v0.4.0
```

### Step 3: Wait for Deployment

1. **Check the Actions tab**:
   - Go to: https://github.com/Arcadia-1/ADCToolbox/actions
   - You'll see the "Build and Deploy Documentation" workflow running
   - Takes about 2-3 minutes

2. **Once complete**, your docs will be live at:
   - **https://arcadia-1.github.io/ADCToolbox/**

3. **Verify deployment**:
   - Visit the docs URL
   - Check that all pages load correctly
   - Test navigation and search

---

## 🔄 How It Works

### Automatic Deployments

The GitHub Actions workflow automatically:
1. **Triggers** on every push to `main` branch or when you create a version tag
2. **Builds** the Sphinx documentation from `python/docs/source/`
3. **Deploys** the HTML output to GitHub Pages
4. **Serves** your docs at https://arcadia-1.github.io/ADCToolbox/

### Workflow Details

```yaml
on:
  push:
    branches: [main]    # Deploys on every push to main
    tags: ['v*']        # Deploys on version tags (v0.4.0, v0.5.0, etc.)
  workflow_dispatch:    # Allows manual trigger from Actions tab
```

### Build Process

```bash
# 1. Install Python and dependencies
pip install -e .[docs]

# 2. Build Sphinx docs
cd python/docs
sphinx-build -b html source build/html

# 3. Create .nojekyll file (allows _static, _modules folders)
touch build/html/.nojekyll

# 4. Deploy to GitHub Pages
# (GitHub Actions handles this automatically)
```

---

## 📝 Making Updates to Documentation

### Regular Documentation Updates

1. **Edit your docs** in `python/docs/source/`:
   - RST files: `index.rst`, `installation.rst`, etc.
   - Markdown files: `algorithms/*.md`
   - Python docstrings in source code

2. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update documentation"
   git push origin main
   ```

3. **Automatic deployment**:
   - GitHub Actions builds and deploys automatically
   - Wait 2-3 minutes
   - Changes live at https://arcadia-1.github.io/ADCToolbox/

### Version-Specific Documentation

For each release:

1. **Update version** in:
   - `python/src/adctoolbox/__init__.py`
   - `python/docs/source/conf.py`

2. **Create tag and push**:
   ```bash
   git tag -a v0.5.0 -m "Release v0.5.0"
   git push origin v0.5.0
   ```

3. **Docs auto-deploy** with the new version number

---

## 🛠️ Manual Build (Local Testing)

Before pushing, you can test the docs locally:

```bash
cd d:\ADCToolbox\python\docs

# Build HTML
python -m sphinx -b html source build/html

# Open in browser
start build/html/index.html  # Windows
# or
open build/html/index.html   # macOS
```

---

## 🔧 Troubleshooting

### If deployment fails:

1. **Check GitHub Actions logs**:
   - https://github.com/Arcadia-1/ADCToolbox/actions
   - Click on the failed workflow
   - Review error messages

2. **Common issues**:

   **Missing dependencies**:
   ```yaml
   # Add to pyproject.toml under [project.optional-dependencies]
   docs = [
       "sphinx>=7.0.0",
       "sphinx-rtd-theme>=2.0.0",
       "myst-parser>=2.0.0",
   ]
   ```

   **Import errors in docs**:
   - Ensure all imports in RST files are correct
   - Check that modules exist in `adctoolbox/`

   **Sphinx build warnings**:
   - Review warnings in Actions log
   - Fix broken cross-references
   - Update missing documentation

3. **Re-run workflow**:
   - Go to Actions tab
   - Click on failed workflow
   - Click "Re-run jobs"

### If docs don't appear:

1. **Check Pages settings**:
   - Ensure "Source" is set to "GitHub Actions"
   - Not "Deploy from branch"

2. **Check deployment status**:
   - Go to: https://github.com/Arcadia-1/ADCToolbox/deployments
   - Should show "github-pages" as active

3. **Wait for DNS**:
   - First deployment may take 5-10 minutes
   - Subsequent deployments are faster (1-2 minutes)

---

## 📊 Documentation Structure

Your documentation will be organized as:

```
https://arcadia-1.github.io/ADCToolbox/
├── index.html                    # Home page
├── installation.html             # Installation guide
├── quickstart.html               # Quick start guide
├── algorithms/
│   ├── index.html                # Algorithm overview
│   ├── fit_sine_4param.html      # Sine fitting
│   ├── analyze_spectrum.html     # Spectrum analysis
│   └── ... (13 more)
├── api/
│   ├── index.html                # API reference home
│   ├── fundamentals.html         # Fundamentals module
│   ├── spectrum.html             # Spectrum module
│   ├── aout.html                 # Analog output module
│   └── dout.html                 # Digital output module
├── changelog.html                # Version history
├── _static/                      # CSS, JS, images
└── _modules/                     # Source code links
```

---

## 🎯 Next Steps

1. **Push to GitHub** (follow Step 2 above)
2. **Enable GitHub Pages** (follow Step 1 above)
3. **Wait for first deployment** (~3 minutes)
4. **Verify docs are live** at https://arcadia-1.github.io/ADCToolbox/
5. **Share the link** in your README badges and PyPI description

---

## ✨ Benefits of GitHub Pages + Actions

- ✅ **Free hosting** for public repositories
- ✅ **Automatic deployments** on every push
- ✅ **Version control** for documentation
- ✅ **No external dependencies** (no ReadTheDocs account needed)
- ✅ **Fast CDN** (GitHub's global infrastructure)
- ✅ **Custom domains** supported (optional)
- ✅ **Integrates with your existing CI/CD**

---

**Setup Created**: 2025-12-18
**Documentation URL**: https://arcadia-1.github.io/ADCToolbox/
**Repository**: https://github.com/Arcadia-1/ADCToolbox
