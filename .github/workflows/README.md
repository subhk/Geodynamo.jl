# GitHub Actions Workflows

This directory contains automated workflows for Geodynamo.jl that handle continuous integration, automated releases, and dependency management.

## 🔄 Workflows Overview

### 1. **CI.yml** - Continuous Integration
**Triggers:** Push to main/master, Pull Requests, Manual dispatch, Tags

**What it does:**
- ✅ **Cross-platform testing** - Tests on Ubuntu, macOS, Windows
- ✅ **Multiple Julia versions** - Tests Julia 1.8, 1.10, 1.11, and nightly
- ✅ **Architecture coverage** - Tests x64 and ARM64 (Apple Silicon)
- ✅ **MPI testing** - Runs smoke tests with OpenMPI
- ✅ **Code coverage** - Uploads coverage reports to Codecov
- ✅ **Documentation** - Builds and deploys docs automatically
- ✅ **Code quality** - Checks formatting with JuliaFormatter and quality with Aqua
- ✅ **Performance benchmarks** - Runs benchmarks on tagged releases
- ✅ **Integration tests** - Tests with latest compatible dependencies

### 2. **TagBot.yml** - Automated Release Management  
**Triggers:** JuliaRegistrator comments, Manual dispatch

**What it does:**
- 🏷️ **Automatic tagging** - Creates Git tags when package is registered
- 📦 **GitHub releases** - Creates release pages with changelogs
- 📝 **Release notes** - Auto-generates release notes from commits and PRs
- 🔗 **Links to registry** - Provides installation instructions

### 3. **CompatHelper.yml** - Dependency Management
**Triggers:** Daily at 5:00 AM UTC, Manual dispatch

**What it does:**
- 🔄 **Dependency updates** - Automatically checks for new compatible versions
- 📋 **Pull requests** - Creates PRs with dependency updates
- 🔍 **Security checks** - Scans for known vulnerabilities
- ⚡ **Smart updates** - Only updates when safe and beneficial

---

## 🛠️ Setup Instructions

### 1. Required Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

#### **CODECOV_TOKEN** (Optional but recommended)
For code coverage reporting:
1. Go to [codecov.io](https://codecov.io/) and sign in with GitHub
2. Add your repository
3. Copy the upload token
4. Add as `CODECOV_TOKEN` secret

#### **DOCUMENTER_KEY** (Optional, for documentation)
For automatic documentation deployment:
1. Generate SSH key pair: `ssh-keygen -t rsa -b 4096 -C "documenter" -f documenter`
2. Add the public key (`documenter.pub`) as a repository deploy key with write access
3. Add the private key (`documenter`) as `DOCUMENTER_KEY` secret

### 2. Enable Workflows

The workflows are automatically enabled when you push them to your repository. You can:
- **View status** in the "Actions" tab of your GitHub repository
- **Manual trigger** using the "Run workflow" button
- **Configure notifications** in your GitHub settings

### 3. Branch Protection (Recommended)

Set up branch protection for your main branch:
1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - ✅ Require a pull request before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Status checks: `test`, `quality`, `docs`

---

## 📊 Monitoring and Usage

### CI Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/yourusername/Geodynamo.jl/workflows/CI/badge.svg)](https://github.com/yourusername/Geodynamo.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/yourusername/Geodynamo.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/Geodynamo.jl)
[![TagBot](https://github.com/yourusername/Geodynamo.jl/workflows/TagBot/badge.svg)](https://github.com/yourusername/Geodynamo.jl/actions/workflows/TagBot.yml)
[![CompatHelper](https://github.com/yourusername/Geodynamo.jl/workflows/CompatHelper/badge.svg)](https://github.com/yourusername/Geodynamo.jl/actions/workflows/CompatHelper.yml)
```

### Workflow Triggers

| Event | CI | TagBot | CompatHelper |
|-------|-------|---------|--------------|
| **Push to main** | ✅ Runs full test suite | ❌ | ❌ |
| **Pull Request** | ✅ Tests changes | ❌ | ❌ |
| **New tag** | ✅ + benchmarks | ✅ Creates release | ❌ |
| **JuliaRegistrator comment** | ❌ | ✅ Creates tag/release | ❌ |
| **Daily 5 AM UTC** | ❌ | ❌ | ✅ Checks dependencies |
| **Manual dispatch** | ✅ Can run anytime | ✅ Can run anytime | ✅ Can run anytime |

---

## 🚀 Release Process

### Automatic Release (Recommended)

1. **Update version** in `Project.toml`
2. **Commit and push** to main branch  
3. **Comment on commit** `@JuliaRegistrator register`
4. **Wait for registration** - JuliaRegistrator will submit to General registry
5. **TagBot creates release** - Automatically creates GitHub release when registered

### Manual Release

If you need to create a release manually:

1. **Create tag**: `git tag v1.2.3 && git push origin v1.2.3`
2. **Run TagBot**: Go to Actions → TagBot → "Run workflow"
3. **Wait for release**: TagBot will create the GitHub release

### Release Notes

TagBot automatically generates release notes including:
- 📋 **Commit summary** since last release
- 🔗 **Pull request links** and contributors  
- 📦 **Installation instructions**
- 🔄 **Changelog link** comparing versions

---

## 🔧 Customization

### CI Matrix Configuration

Edit `.github/workflows/CI.yml` to customize:

```yaml
strategy:
  matrix:
    version: ['1.8', '1.10', '1.11']  # Julia versions
    os: [ubuntu-latest, macos-latest] # Operating systems
    arch: [x64]                       # Architectures
```

### CompatHelper Schedule

Change the dependency check frequency in `.github/workflows/CompatHelper.yml`:

```yaml
schedule:
  - cron: "0 5 * * *"  # Daily at 5 AM UTC
  # - cron: "0 5 * * 1"  # Weekly on Monday  
  # - cron: "0 5 1 * *"  # Monthly on 1st
```

### TagBot Configuration

Customize release creation in `.github/workflows/TagBot.yml`:

```yaml
with:
  token: ${{ secrets.GITHUB_TOKEN }}
  changelog_ignore_labels: |
    documentation
    style
  changelog_include_all_prs: false
```

---

## 🐛 Troubleshooting

### Common Issues

#### ❌ "Tests failing on Windows"
- MPI may not work on Windows runners
- Consider excluding Windows from MPI tests
- Use Windows-specific MPI setup (MS-MPI)

#### ❌ "Documentation build fails"
- Check `DOCUMENTER_KEY` secret is set correctly
- Verify docs/Project.toml has correct dependencies
- Ensure docs/make.jl doesn't have errors

#### ❌ "TagBot not creating releases"  
- Verify JuliaRegistrator successfully registered the package
- Check that the comment format was correct: `@JuliaRegistrator register`
- Ensure repository permissions allow TagBot to create releases

#### ❌ "CompatHelper PRs not appearing"
- Check CompatHelper has permission to create PRs
- Verify there are actually dependency updates available
- Look for errors in the CompatHelper workflow logs

### Performance Tips

1. **Use caching** - Workflows cache Julia artifacts for faster builds
2. **Parallel jobs** - Multiple OS/version combinations run in parallel  
3. **Conditional runs** - Some jobs only run on specific events
4. **Artifact cleanup** - Benchmark results are kept for 90 days only

### Getting Help

- **GitHub Actions docs**: https://docs.github.com/en/actions
- **Julia Actions**: https://github.com/julia-actions
- **TagBot**: https://github.com/JuliaRegistries/TagBot
- **CompatHelper**: https://github.com/JuliaRegistries/CompatHelper.jl

---

## 📈 Benefits

These workflows provide:

✅ **Automated testing** - Catch bugs before they reach users  
✅ **Cross-platform compatibility** - Ensure code works everywhere  
✅ **Dependency management** - Stay up-to-date with ecosystem  
✅ **Professional releases** - Automated, consistent release process  
✅ **Code quality** - Maintain high standards automatically  
✅ **Documentation** - Always up-to-date docs  
✅ **Performance tracking** - Monitor performance over time  
✅ **Security monitoring** - Check for vulnerable dependencies  

Your package now has enterprise-grade automation! 🎉