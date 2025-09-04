# Documentation Files for GitHub Pages

This directory contains the documentation files that can be used to create a GitHub Pages website for Geodynamo.jl.

## Setup Instructions

1. **Copy files to main docs directory:**
   ```bash
   cp -r script/docs/* docs/
   ```

2. **Enable GitHub Pages:**
   - Go to your GitHub repository settings
   - Navigate to "Pages" section
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/docs" folder
   - Save settings

3. **Configure Jekyll (optional):**
   - The `_config.yml` file configures the Jekyll theme
   - GitHub Pages will automatically build the site using Jekyll

## File Structure

```
docs/
├── _config.yml              # Jekyll configuration
├── index.md                 # Main homepage (replaces index.html)
├── API_REFERENCE.md         # Complete API documentation
├── VISUALIZATION.md         # Plotting and analysis guide
├── EXAMPLES.md             # Code examples gallery
├── _layouts/               # Custom layouts
├── assets/                 # CSS, images, and other assets
└── script/                # JavaScript for interactive features
```

## Features

- **Responsive design** that works on desktop and mobile
- **Syntax highlighting** for Julia code
- **Interactive examples** with collapsible sections
- **Search functionality** (when using supported themes)
- **Navigation menu** with all major sections
- **Clean, professional styling**

## Customization

You can customize the appearance by:
- Editing `_config.yml` for site-wide settings
- Modifying CSS in `assets/style.css`
- Adding custom layouts in `_layouts/`
- Updating navigation in `_config.yml`

The site will be available at: `https://yourusername.github.io/Geodynamo.jl/`