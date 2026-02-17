#!/usr/bin/env python3
"""
Scrape merged pipeline upgrade PRs from redhat-appstudio/infra-deployments,
extract version info and release notes, and generate a static HTML dashboard.

Usage:
    python generate.py                     # uses GITHUB_TOKEN env var
    python generate.py --limit 20          # scrape last 20 PRs
    python generate.py --output data/      # custom output directory
"""

import argparse
import json
import os
import re
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

logger = logging.getLogger(__name__)


def md_to_html(text: str) -> str:
    """Convert simplified markdown (bullet lists, bold, code, links) to HTML."""
    if not text:
        return ""

    lines = text.split("\n")
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        is_bullet = stripped.startswith("* ") or stripped.startswith("- ")

        if is_bullet:
            content = stripped[2:]
            content = _inline_md(content)
            if not in_list:
                html_lines.append('<ul class="md-list">')
                in_list = True
            html_lines.append(f"<li>{content}</li>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            content = _inline_md(stripped)
            html_lines.append(f"<p>{content}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    """Convert inline markdown: **bold**, `code`, [link](url)."""
    import html as html_mod
    text = html_mod.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    return text

INFRA_REPO = "redhat-appstudio/infra-deployments"
SEARCH_QUERIES = [
    "upgrade production pipelines build",
    "upgrade pipelines nightly",
    "Upgrade pipelines nightly",
    "upgrade pipelines dev",
    "upgrade pipelines Konflux nightly",
    "upgrade pipelines deployment",
]
INDEX_IMAGE_PATTERN = re.compile(
    r"quay\.io/openshift-pipeline/pipelines-index-[\w.]+@sha256:[a-f0-9]+"
)
VERSION_PATTERN = re.compile(r"(\d+\.\d+\.\d+-\d+)")
COMPONENT_SECTIONS = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def github_api(endpoint: str, token: str | None = None, params: dict | None = None) -> dict | list:
    """Make a GitHub API request."""
    url = f"https://api.github.com/{endpoint.lstrip('/')}"
    if params:
        url = f"{url}?{urlencode(params)}"

    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.load(resp)
    except HTTPError as e:
        logger.error(f"GitHub API error {e.code} for {url}")
        if e.code == 403:
            logger.error("Rate limited. Set GITHUB_TOKEN for higher limits.")
        raise


def search_upgrade_prs(token: str | None, limit: int = 30) -> list[dict]:
    """Search for merged pipeline upgrade PRs in infra-deployments."""
    all_prs = {}

    for query in SEARCH_QUERIES:
        search_q = f'repo:{INFRA_REPO} is:pr is:merged "{query}"'
        logger.info(f"Searching: {query}")
        try:
            results = github_api(
                "search/issues",
                token=token,
                params={"q": search_q, "sort": "updated", "order": "desc", "per_page": limit},
            )
            for item in results.get("items", []):
                all_prs[item["number"]] = item
        except HTTPError:
            logger.warning(f"Search failed for query: {query}")

    sorted_prs = sorted(all_prs.values(), key=lambda p: p.get("closed_at", ""), reverse=True)
    return sorted_prs[:limit]


def get_pr_details(pr_number: int, token: str | None) -> dict:
    """Get full PR details including diff info."""
    pr = github_api(f"repos/{INFRA_REPO}/pulls/{pr_number}", token=token)
    return pr


def get_pr_files(pr_number: int, token: str | None) -> list[dict]:
    """Get changed files for a PR."""
    files = github_api(f"repos/{INFRA_REPO}/pulls/{pr_number}/files", token=token)
    return files


def extract_version(title: str) -> str:
    """Extract pipeline version from PR title."""
    match = VERSION_PATTERN.search(title)
    return match.group(1) if match else ""


def extract_environment(title: str) -> str:
    """Determine deployment environment from PR title."""
    title_lower = title.lower()
    if "production" in title_lower or "prod" in title_lower:
        return "production"
    if "dev" in title_lower and "stag" in title_lower:
        return "dev/staging"
    if "staging" in title_lower or "stage" in title_lower:
        return "staging"
    if "dev" in title_lower:
        return "development"
    return "unknown"


def extract_index_images(files: list[dict]) -> tuple[str, str]:
    """Extract old and new index images from PR file diffs."""
    old_image = ""
    new_image = ""

    for f in files:
        patch = f.get("patch", "")
        if not patch:
            continue

        for line in patch.split("\n"):
            images = INDEX_IMAGE_PATTERN.findall(line)
            for img in images:
                if line.startswith("-") and not old_image:
                    old_image = img
                elif line.startswith("+") and not new_image:
                    new_image = img

        if old_image and new_image:
            break

    return old_image, new_image


def extract_clusters(files: list[dict]) -> list[str]:
    """Extract cluster names from changed file paths."""
    clusters = set()
    for f in files:
        path = f.get("filename", "")
        parts = path.split("/")
        if len(parts) >= 4 and parts[0] == "components" and parts[1] == "pipeline-service":
            cluster = parts[3]
            if cluster != "base":
                clusters.add(cluster)
    return sorted(clusters)


def extract_promoted_from(body: str) -> int | None:
    """Extract the PR number this was promoted from."""
    match = re.search(r"Promotion of .+/pull/(\d+)", body or "")
    return int(match.group(1)) if match else None


def parse_release_notes(body: str) -> dict:
    """Parse the PR body to extract structured release notes."""
    if not body:
        return {"raw": "", "components": []}

    notes_start = None
    for marker in ["Release Notes generated by Gemini:", "Release Notes:", "## "]:
        idx = body.find(marker)
        if idx != -1:
            if marker == "## ":
                notes_start = idx
            else:
                notes_start = idx + len(marker)
            break

    if notes_start is None:
        return {"raw": "", "components": []}

    notes_text = body[notes_start:].strip()

    components = []
    sections = re.split(r"(?=^## )", notes_text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        header_match = re.match(r"^##\s+(.+)$", section, re.MULTILINE)
        if not header_match:
            continue

        component_name = header_match.group(1).strip()
        component_body = section[header_match.end():].strip()

        subsections = {}
        current_sub = None
        current_lines = []

        for line in component_body.split("\n"):
            sub_match = re.match(r"^###\s+(.+)$", line)
            if sub_match:
                if current_sub and current_lines:
                    subsections[current_sub] = "\n".join(current_lines).strip()
                current_sub = sub_match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_sub and current_lines:
            subsections[current_sub] = "\n".join(current_lines).strip()

        if component_name and (subsections or component_body):
            subsections_html = {k: md_to_html(v) for k, v in subsections.items()}
            components.append({
                "name": component_name,
                "subsections": subsections,
                "subsections_html": subsections_html,
                "raw": component_body,
                "raw_html": md_to_html(component_body),
            })

    return {
        "raw": notes_text,
        "components": components,
    }


def scrape_releases(token: str | None, limit: int = 30) -> list[dict]:
    """Scrape and structure release data from PRs."""
    logger.info(f"Searching for up to {limit} upgrade PRs...")
    prs = search_upgrade_prs(token, limit=limit)
    logger.info(f"Found {len(prs)} PRs to process")

    releases = []

    for pr_item in prs:
        pr_number = pr_item["number"]
        title = pr_item.get("title", "")
        logger.info(f"Processing PR #{pr_number}: {title}")

        try:
            pr = get_pr_details(pr_number, token)
            files = get_pr_files(pr_number, token)
        except HTTPError as e:
            logger.warning(f"Skipping PR #{pr_number}: API error {e.code}")
            continue

        version = extract_version(title)
        environment = extract_environment(title)
        old_image, new_image = extract_index_images(files)
        clusters = extract_clusters(files)
        promoted_from = extract_promoted_from(pr.get("body", ""))
        release_notes = parse_release_notes(pr.get("body", ""))

        release = {
            "pr_number": pr_number,
            "pr_url": pr.get("html_url", ""),
            "title": title,
            "version": version,
            "environment": environment,
            "merged_at": pr.get("merged_at", ""),
            "created_at": pr.get("created_at", ""),
            "author": pr.get("user", {}).get("login", ""),
            "old_index_image": old_image,
            "new_index_image": new_image,
            "clusters": clusters,
            "promoted_from_pr": promoted_from,
            "release_notes": release_notes,
            "component_count": len(release_notes.get("components", [])),
        }

        releases.append(release)

    releases.sort(key=lambda r: r.get("merged_at", ""), reverse=True)
    return releases


def generate_dashboard(releases: list[dict], output_dir: Path):
    """Generate the static HTML dashboard from releases data."""
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
    template = env.get_template("index.html")

    latest_prod = next(
        (r for r in releases if r["environment"] == "production" and r["version"]),
        None,
    )
    latest_dev = next(
        (r for r in releases if r["environment"] in ("dev/staging", "development", "staging") and r["version"]),
        None,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    html = template.render(
        releases=releases,
        latest_prod=latest_prod,
        latest_dev=latest_dev,
        last_updated=now,
        total_releases=len(releases),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "index.html").write_text(html)
    logger.info(f"Written {output_dir / 'index.html'}")

    data = {
        "last_updated": now,
        "releases": releases,
    }
    (output_dir / "releases.json").write_text(json.dumps(data, indent=2))
    logger.info(f"Written {output_dir / 'releases.json'}")


def main():
    parser = argparse.ArgumentParser(description="Generate OpenShift Pipelines Nightly Release Dashboard")
    parser.add_argument("--limit", type=int, default=30, help="Max number of PRs to scrape (default: 30)")
    parser.add_argument("--output", type=str, default="data", help="Output directory (default: data/)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        logger.warning("No GITHUB_TOKEN set. API rate limits will be very low (60 req/hr).")

    output_dir = Path(args.output)
    releases = scrape_releases(token, limit=args.limit)

    if not releases:
        logger.error("No releases found. Check your search queries and token.")
        sys.exit(1)

    logger.info(f"\nCollected {len(releases)} releases")
    generate_dashboard(releases, output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
