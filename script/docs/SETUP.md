# Documentation Website Setup Guide

This guide will help you set up a professional documentation website for Geodynamo.jl using GitHub Pages and Jekyll.

## 📁 What's Included

The documentation website includes:

```
docs/
├── _config.yml              # Jekyll site configuration
├── index.md                 # Homepage
├── getting-started.md       # Complete tutorial
├── api-reference.md         # API documentation
├── visualization.md         # Plotting guide
├── examples.md             # Code examples
├── assets/
│   └── css/
│       └── style.scss      # Custom styling
├── _layouts/
│   ├── default.html        # Base layout
│   └── page.html           # Page layout
├── Gemfile                 # Ruby dependencies
└── SETUP.md               # This setup guide
```

## 🚀 Quick Setup (GitHub Pages)

### Method 1: Copy Files to Main docs/

1. **Copy documentation files:**
   ```bash
   cd /path/to/Geodynamo.jl
   cp -r script/docs/* docs/
   ```

2. **Enable GitHub Pages:**
   - Go to your GitHub repository settings
   - Navigate to "Pages" section
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/docs" folder
   - Save settings

3. **Wait for deployment** (usually 2-5 minutes)

4. **Access your site:**
   ```
   https://your-username.github.io/Geodynamo.jl/
   ```

### Method 2: Use docs/ as Root

If you want to use docs/ as the website root:

1. **Set GitHub Pages source** to "main" branch and "/" (root folder)
2. **Move content** from script/docs/ to repository root
3. **Update _config.yml** baseurl to `""`

## 🔧 Local Development Setup

### Prerequisites

- **Ruby 3.0+** ([Install Ruby](https://www.ruby-lang.org/en/downloads/))
- **Bundler** (`gem install bundler`)

### Setup Steps

1. **Navigate to docs directory:**
   ```bash
   cd docs/  # or wherever you copied the files
   ```

2. **Install dependencies:**
   ```bash
   bundle install
   ```

3. **Start local server:**
   ```bash
   bundle exec jekyll serve
   ```

4. **Open in browser:**
   ```
   http://localhost:4000/Geodynamo.jl/
   ```

### Development Commands

```bash
# Start server with live reload
bundle exec jekyll serve --livereload

# Build site for production
bundle exec jekyll build

# Serve with draft posts
bundle exec jekyll serve --drafts

# Serve on different port
bundle exec jekyll serve --port 4001
```

## ⚙️ Customization

### Site Configuration

Edit `_config.yml` to customize:

```yaml
# Site information
title: "Your Package Name"
description: "Your package description"
baseurl: "/Your-Repo-Name"
url: "https://your-username.github.io"

# Author information
author:
  name: "Your Name"
  email: "your.email@example.com"
  github: "your-username"

# Custom variables
geodynamo:
  version: "1.0.0"
  julia_version: "1.8+"
```

### Navigation Menu

The navigation is automatically generated from pages with `nav_order`:

```yaml
---
layout: page
title: "Page Title"
nav_order: 3
---
```

### Custom Styling

Edit `assets/css/style.scss` to customize appearance:

```scss
// Custom colors
$primary-color: #667eea;
$secondary-color: #764ba2;

// Add your custom styles here
.my-custom-class {
  color: $primary-color;
}
```

### Adding New Pages

1. **Create new markdown file** in docs/
2. **Add YAML front matter:**
   ```yaml
   ---
   layout: page
   title: "New Page"
   nav_order: 6
   description: "Page description"
   ---
   ```
3. **Write your content** in Markdown

## 🎨 Theme Features

### Built-in Components

**Feature Cards:**
```html
<div class="feature-grid">
  <div class="feature-card">
    <h3>Feature Title</h3>
    <p>Feature description</p>
  </div>
</div>
```

**Info Boxes:**
```html
<div class="info-box">
  <h4>💡 Information</h4>
  <p>Important information for users</p>
</div>
```

**Code with Copy Button:**
All code blocks automatically get copy buttons on hover.

**External Link Indicators:**
External links automatically get an indicator icon.

**Table of Contents:**
Pages with 3+ headings automatically get a table of contents.

### Responsive Design

The theme is fully responsive and works on:
- ✅ Desktop computers
- ✅ Tablets  
- ✅ Mobile phones
- ✅ High-DPI displays

## 📊 Analytics & SEO

### Google Analytics

Add to `_config.yml`:

```yaml
google_analytics: UA-XXXXXXXXX-X
```

### SEO Optimization

The site includes:
- ✅ **Meta tags** for search engines
- ✅ **Open Graph** tags for social sharing
- ✅ **Structured data** for rich snippets
- ✅ **Sitemap** generation
- ✅ **RSS feed** for updates

### Social Sharing

Update `_config.yml` for better social sharing:

```yaml
social:
  name: "Package Name"
  links:
    - "https://github.com/your-username/your-repo"
    - "https://twitter.com/your-handle"
```

## 🐛 Troubleshooting

### Common Issues

**❌ "Jekyll not found"**
```bash
gem install jekyll bundler
bundle install
```

**❌ "Permission denied"**
```bash
sudo chown -R $(whoami) docs/
```

**❌ "Site not updating"**
- Check GitHub Actions for build errors
- Verify file names and paths
- Clear browser cache

**❌ "Ruby version error"**
```bash
rbenv install 3.0.0
rbenv global 3.0.0
```

### Build Errors

Check Jekyll build logs:
```bash
bundle exec jekyll build --verbose
```

Common fixes:
- Check YAML front matter syntax
- Verify file paths in links
- Ensure all referenced files exist

## 🚀 Advanced Features

### Custom Domain

1. **Add CNAME file** to docs/:
   ```
   your-domain.com
   ```

2. **Configure DNS** with your domain provider

3. **Update _config.yml:**
   ```yaml
   url: "https://your-domain.com"
   baseurl: ""
   ```

### Automatic Deployment

The site deploys automatically when you push to main branch. You can also set up:

- **Staging environments** for testing
- **Pull request previews** with Netlify
- **Custom build actions** with GitHub Actions

### Content Management

For non-technical users:
- **Forestry.io** - GUI editor for Jekyll
- **Netlify CMS** - Git-based content management
- **Prose.io** - Simple GitHub-based editor

## 📚 Resources

- **[Jekyll Documentation](https://jekyllrb.com/docs/)**
- **[GitHub Pages Documentation](https://docs.github.com/en/pages)**
- **[Liquid Template Language](https://shopify.github.io/liquid/)**
- **[Markdown Guide](https://www.markdownguide.org/)**

## 🎉 Success!

Your documentation website includes:

✅ **Professional design** with modern styling  
✅ **Mobile responsive** layout  
✅ **Interactive features** like copy buttons and smooth scrolling  
✅ **SEO optimized** for search engines  
✅ **Social media ready** with Open Graph tags  
✅ **Automated deployment** via GitHub Pages  
✅ **Fast loading** with optimized assets  
✅ **Accessible** design following web standards  

## 💬 Need Help?

- **GitHub Issues** - Report bugs or request features
- **GitHub Discussions** - Ask questions and get help
- **Jekyll Community** - General Jekyll support

---

**Your professional documentation website is ready to showcase your Julia package! 🚀**