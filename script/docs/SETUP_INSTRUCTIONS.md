# GitHub Pages Setup Instructions

This guide will help you set up a GitHub Pages website for Geodynamo.jl using the documentation files created in this directory.

## 📁 File Overview

The documentation website includes:

```
script/docs/
├── _config.yml           # Jekyll configuration for GitHub Pages
├── index.md             # Homepage (markdown version)  
├── getting-started.md   # Complete beginner tutorial
├── api-reference.md     # Comprehensive API documentation
├── visualization.md     # Plotting and visualization guide
├── examples.md          # Working code examples
└── README.md           # This setup guide
```

## 🚀 Setup Steps

### Step 1: Copy Files to Main Docs Directory

```bash
# Navigate to your repository root
cd /path/to/Geodynamo.jl

# Copy documentation files to the main docs directory
cp script/docs/* docs/

# If you don't have write permissions, use sudo:
sudo cp script/docs/* docs/
sudo chown -R $(whoami):$(id -gn) docs/
```

### Step 2: Enable GitHub Pages

1. **Go to your repository on GitHub**
   - Visit: https://github.com/your-username/Geodynamo.jl

2. **Navigate to Settings**  
   - Click the "Settings" tab at the top of the repository

3. **Find Pages Section**
   - Scroll down to "Pages" in the left sidebar

4. **Configure Source**
   - Set "Source" to "Deploy from a branch"
   - Select "main" branch  
   - Select "/docs" folder
   - Click "Save"

5. **Wait for Deployment**
   - GitHub will build and deploy your site
   - This usually takes 2-5 minutes
   - You'll see a green checkmark when it's ready

### Step 3: Access Your Website

Your documentation website will be available at:
```
https://your-username.github.io/Geodynamo.jl/
```

## 🎨 Customization Options

### Jekyll Theme

The site uses the `minima` theme by default. You can change it in `_config.yml`:

```yaml
# Options include: minima, cayman, slate, architect, etc.
theme: minima
```

### Site Information

Update these fields in `_config.yml`:

```yaml
title: "Your Package Name"
description: "Your package description"
url: "https://your-username.github.io"
baseurl: "/Your-Repo-Name"
author:
  name: "Your Name"
  email: "your.email@example.com"
```

### Navigation Menu

The header navigation is defined in `_config.yml`:

```yaml
header_pages:
  - getting-started.md
  - api-reference.md
  - visualization.md
  - examples.md
```

### Colors and Styling

Create a `docs/assets/style.css` file for custom styling:

```css
/* Custom styles */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.feature-card {
    border: 1px solid #e1e4e8;
    border-radius: 6px;
    padding: 1rem;
    margin: 1rem 0;
}
```

## 📝 Content Updates

### Adding New Pages

1. Create a new `.md` file in `docs/`
2. Add YAML front matter:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   ---
   ```
3. Add the page to navigation in `_config.yml`

### Updating Existing Content

Simply edit the markdown files in `docs/`. Changes will be automatically deployed when you commit to the main branch.

## 🔧 Advanced Features

### Custom Domain

If you have a custom domain:

1. Add a `CNAME` file to `docs/`:
   ```
   your-domain.com
   ```

2. Configure DNS with your domain provider
3. Update `_config.yml`:
   ```yaml
   url: "https://your-domain.com"
   baseurl: ""
   ```

### Analytics

Add Google Analytics by updating `_config.yml`:

```yaml
google_analytics: UA-XXXXXXXXX-X
```

### Search Functionality

For advanced search, consider using:
- **Algolia DocSearch** (for popular open source projects)
- **lunr.js** (client-side search)
- **Simple Jekyll Search** (basic search)

## 🐛 Troubleshooting

### Site Not Building

1. **Check GitHub Actions tab** for build errors
2. **Verify YAML syntax** in `_config.yml` using an online validator
3. **Check file paths** - all links should be relative
4. **Review Jekyll docs** at https://jekyllrb.com/docs/

### Broken Links

- Use relative links: `[API](api-reference.html)` not `[API](/api-reference.html)`
- Check that all referenced files exist
- Ensure proper file extensions (`.md` files become `.html`)

### Styling Issues

- **Test locally** using Jekyll serve:
  ```bash
  cd docs
  bundle exec jekyll serve
  ```
- **Check browser console** for CSS/JS errors
- **Validate HTML** using online validators

## 📚 Resources

- **[GitHub Pages Documentation](https://docs.github.com/en/pages)**
- **[Jekyll Documentation](https://jekyllrb.com/docs/)**
- **[Markdown Guide](https://www.markdownguide.org/)**
- **[YAML Syntax](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html)**

## 🎉 Success!

Once set up, your documentation website will:

- ✅ **Automatically update** when you push changes
- ✅ **Be mobile-responsive** with clean styling  
- ✅ **Include syntax highlighting** for code examples
- ✅ **Provide easy navigation** between sections
- ✅ **Be discoverable** by search engines
- ✅ **Load fast** with GitHub's CDN

## 📞 Need Help?

If you encounter issues:

1. **Check the GitHub Actions logs** for build errors
2. **Compare with working examples** like other Julia packages
3. **Ask on GitHub Discussions** for community support
4. **File an issue** if you find bugs in the documentation

---

**Your professional documentation website is ready to go! 🚀**