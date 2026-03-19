# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A static site generator that scrapes OpenShift Pipelines upgrade PRs from `redhat-appstudio/infra-deployments` and produces an HTML dashboard showing nightly release status. The dashboard is deployed to GitHub Pages via a scheduled GitHub Actions workflow (every 6 hours).

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Generate the dashboard (outputs to data/)
uv run python generate.py --limit 30 --output data/ -v

# Requires GITHUB_TOKEN or GH_TOKEN env var for API rate limits
GITHUB_TOKEN=<token> uv run python generate.py -v
```

There are no tests in this project.

## Architecture

The project has two main components:

1. **`generate.py`** - Single-file Python script that:
   - Searches GitHub for pipeline upgrade PRs using multiple search queries (`SEARCH_QUERIES`)
   - Scrapes both merged PRs (past releases) and open PRs (upcoming releases)
   - Extracts version, environment (production/dev/staging), index images, clusters, and release notes from PR titles, bodies, and file diffs
   - Renders a Jinja2 template to produce static HTML + JSON output

2. **`templates/index.html`** - Jinja2 template for the dashboard. Self-contained single-page HTML with inline CSS and JavaScript. Dark-themed UI with hero cards for latest versions, expandable release cards with component-level release notes.

## Key Patterns

- **Data source**: All data comes from GitHub Search API and PR API on `redhat-appstudio/infra-deployments`
- **No external HTTP library**: Uses stdlib `urllib.request` for all API calls
- **Version extraction**: Regex pattern `(\d+\.\d+\.\d+-\d+)` from PR titles
- **Environment detection**: Keyword matching in PR titles (production/prod, dev, staging)
- **Release notes parsing**: Splits PR body on `## ` headers into component sections, then `### ` into subsections (features, bug fixes, etc.)
- **Markdown conversion**: Custom `md_to_html()` handles bullet lists, bold, code, and links (not a full markdown parser)
- **Cluster extraction**: Parsed from file paths matching `components/pipeline-service/*/` in PR diffs

## Tech Stack

- Python 3.13, managed with `uv`
- Only dependency: `jinja2` (pulls in `markupsafe`)
- GitHub Actions for CI/CD, deployed to GitHub Pages
