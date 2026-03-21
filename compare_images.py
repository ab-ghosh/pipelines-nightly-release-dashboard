#!/usr/bin/env python3
"""
Compare old and new index images for each release to extract:
1. Per-component commit diffs
2. Associated pull requests
3. JIRA tickets from commit messages

Requires: podman, GITHUB_TOKEN
Optional: JIRA_TOKEN (for fetching JIRA ticket details)

Usage:
    python compare_images.py --input data/releases.json --output data/comparisons.json --limit 5
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

# Mapping from container image names to upstream git repos
# Only tracking PAC, Pipelines, and Operator
IMAGE_REPO_TO_GIT_REPO = {
    # tektoncd/pipeline
    "pipelines-controller-rhel9": "tektoncd/pipeline",
    "pipelines-entrypoint-rhel9": "tektoncd/pipeline",
    "pipelines-events-rhel9": "tektoncd/pipeline",
    "pipelines-nop-rhel9": "tektoncd/pipeline",
    "pipelines-resolvers-rhel9": "tektoncd/pipeline",
    "pipelines-sidecarlogresults-rhel9": "tektoncd/pipeline",
    "pipelines-webhook-rhel9": "tektoncd/pipeline",
    "pipelines-workingdirinit-rhel9": "tektoncd/pipeline",
    # tektoncd/operator
    "pipelines-operator-bundle": "tektoncd/operator",
    "pipelines-operator-proxy-rhel9": "tektoncd/operator",
    "pipelines-operator-webhook-rhel9": "tektoncd/operator",
    "pipelines-rhel9-operator": "tektoncd/operator",
    # openshift-pipelines/pipelines-as-code
    "pipelines-pipelines-as-code-cli-rhel9": "openshift-pipelines/pipelines-as-code",
    "pipelines-pipelines-as-code-controller-rhel9": "openshift-pipelines/pipelines-as-code",
    "pipelines-pipelines-as-code-watcher-rhel9": "openshift-pipelines/pipelines-as-code",
    "pipelines-pipelines-as-code-webhook-rhel9": "openshift-pipelines/pipelines-as-code",
}

# JIRA ticket pattern — only match known project prefixes
JIRA_PATTERN = re.compile(r"\b(SRVKP-\d+)\b")

# Commit messages matching any of these patterns are filtered out
SKIP_COMMIT_PATTERNS = [
    re.compile(r"(?i)\bCVE-\d{4}-\d+\b"),              # CVE fixes
    re.compile(r"(?i)^Bump\s+"),                         # dependency bumps
    re.compile(r"(?i)^Update\s+dependency\b"),            # dependency updates
    re.compile(r"(?i)\bdependabot\b"),                    # dependabot commits
    re.compile(r"(?i)\brenovate\b"),                      # renovate bot
    re.compile(r"(?i)^Merge (pull request|branch)\b"),    # merge commits
    re.compile(r"(?i)^chore\(deps\)"),                    # chore(deps): ...
    re.compile(r"(?i)^vendor\b"),                         # vendor updates
]


def is_noise_commit(message: str) -> bool:
    """Return True if the commit is a dependency bump, CVE fix, or bot noise."""
    first_line = message.split("\n")[0] if message else ""
    return any(p.search(first_line) for p in SKIP_COMMIT_PATTERNS)
DEFAULT_JIRA_URL = "https://redhat.atlassian.net"


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------

def github_api(endpoint, token=None, accept=None):
    url = f"https://api.github.com/{endpoint.lstrip('/')}"
    headers = {"Accept": accept or "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request(url, headers=headers)
    try:
        with urlopen(req) as resp:
            return json.load(resp)
    except HTTPError as e:
        logger.warning(f"GitHub API error {e.code} for {url}")
        return None


def get_compare(repo, old_sha, new_sha, token):
    """Get commits between two SHAs via GitHub Compare API."""
    data = github_api(f"repos/{repo}/compare/{old_sha}...{new_sha}", token=token)
    if not data:
        return []
    return [
        {
            "sha": c.get("sha", ""),
            "message": c.get("commit", {}).get("message", ""),
            "author": c.get("commit", {}).get("author", {}).get("name", ""),
            "date": c.get("commit", {}).get("author", {}).get("date", ""),
            "url": c.get("html_url", ""),
        }
        for c in data.get("commits", [])
    ]


def get_prs_for_commits(repo, commits, token):
    """Find merged PRs associated with a set of commits.

    Uses the compare API's commit data — most merge-based workflows include
    the PR number in the merge commit message.  Falls back to the GitHub
    Search API to find PRs by SHA in bulk (much faster than per-commit lookups).
    """
    from urllib.parse import quote

    pr_pattern = re.compile(r"#(\d+)\b")
    seen = set()
    prs = []

    # 1) Fast pass: extract PR numbers from commit messages (merge commits)
    pr_numbers_from_msgs = set()
    for c in commits:
        msg = c.get("message", "").split("\n")[0]
        for m in pr_pattern.findall(msg):
            pr_numbers_from_msgs.add(int(m))

    # Fetch details for PR numbers found in messages
    for num in sorted(pr_numbers_from_msgs):
        data = github_api(f"repos/{repo}/pulls/{num}", token=token)
        if data and data.get("number"):
            seen.add(data["number"])
            prs.append({
                "number": data["number"],
                "title": data.get("title", ""),
                "url": data.get("html_url", ""),
                "state": data.get("state", ""),
                "merged_at": data.get("merged_at", ""),
            })

    # 2) Search API pass: find PRs by SHA for commits not covered above
    #    Search in batches (max ~5 SHAs per query to stay within URL limits)
    uncovered_shas = [c["sha"] for c in commits]
    for i in range(0, len(uncovered_shas), 5):
        batch = uncovered_shas[i:i+5]
        sha_query = " ".join(batch[:5])
        search_q = quote(f"repo:{repo} is:pr is:merged {sha_query}")
        data = github_api(
            f"search/issues?q={search_q}&per_page=30",
            token=token,
        )
        if data:
            for item in data.get("items", []):
                num = item.get("number")
                if num and num not in seen:
                    seen.add(num)
                    prs.append({
                        "number": num,
                        "title": item.get("title", ""),
                        "url": item.get("html_url", ""),
                        "state": "closed",
                        "merged_at": item.get("pull_request", {}).get("merged_at", ""),
                    })

    return prs


# ---------------------------------------------------------------------------
# JIRA helpers
# ---------------------------------------------------------------------------

def extract_jira_keys(commits):
    """Extract unique JIRA ticket keys from commit messages."""
    keys = set()
    for c in commits:
        keys.update(JIRA_PATTERN.findall(c.get("message", "")))
    return sorted(keys)


def fetch_jira_details(keys, jira_token, jira_email, jira_url=DEFAULT_JIRA_URL):
    """Fetch JIRA ticket details for a list of keys.

    For Jira Cloud: uses Basic Auth (email:api_token).
    Set JIRA_EMAIL and JIRA_TOKEN env vars.
    """
    if not keys or not jira_token:
        return []

    tickets = []
    # Build auth header: Basic Auth for Jira Cloud, Bearer for Server
    if jira_email:
        # Jira Cloud: Basic Auth with email:api_token
        creds = base64.b64encode(f"{jira_email}:{jira_token}".encode()).decode()
        auth_header = f"Basic {creds}"
    else:
        # Jira Server/DC: Bearer token
        auth_header = f"Bearer {jira_token}"

    # Batch in groups of 50 to avoid URL length limits
    for i in range(0, len(keys), 50):
        batch = keys[i : i + 50]
        jql = f"key in ({','.join(batch)})"
        url = f"{jira_url}/rest/api/3/search/jql"
        payload = json.dumps({
            "jql": jql,
            "fields": ["summary", "status", "assignee", "priority", "issuetype"],
            "maxResults": 50,
        }).encode()
        headers = {
            "Authorization": auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        req = Request(url, data=payload, headers=headers, method="POST")
        try:
            with urlopen(req) as resp:
                data = json.load(resp)
                for issue in data.get("issues", []):
                    fields = issue.get("fields", {})
                    tickets.append(
                        {
                            "key": issue.get("key", ""),
                            "summary": fields.get("summary", ""),
                            "status": fields.get("status", {}).get("name", ""),
                            "priority": fields.get("priority", {}).get("name", ""),
                            "type": fields.get("issuetype", {}).get("name", ""),
                            "assignee": (fields.get("assignee") or {}).get(
                                "displayName", "Unassigned"
                            ),
                            "url": f"{jira_url}/browse/{issue.get('key', '')}",
                        }
                    )
        except HTTPError as e:
            logger.warning(f"JIRA API error {e.code} for batch starting at {batch[0]}")
        except Exception as e:
            logger.warning(f"JIRA API error: {e}")

    return tickets


# ---------------------------------------------------------------------------
# Podman / catalog / image inspection
# ---------------------------------------------------------------------------

def podman_run(args):
    """Run a podman command and return stdout."""
    cmd = ["podman"] + args
    logger.debug(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.debug(f"podman error: {result.stderr.strip()}")
        return None
    return result.stdout.strip()


def extract_catalog_bundles(index_image):
    """Pull an index image and extract bundle data from its catalog.json."""
    # Create container from index image
    container_id = podman_run(["create", "-q", index_image])
    if not container_id:
        logger.error(f"Failed to create container for {index_image}")
        return None, None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_file = os.path.join(tmpdir, "catalog.json")
            result = subprocess.run(
                [
                    "podman",
                    "cp",
                    f"{container_id}:/configs/openshift-pipelines-operator-rh/catalog.json",
                    catalog_file,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"Failed to extract catalog.json: {result.stderr.strip()}")
                return None, container_id

            with open(catalog_file) as f:
                content = f.read()

            # catalog.json is newline-delimited JSON objects
            objects = re.findall(r"\n(\{.*?\n\})", content, re.DOTALL)
            entries = []
            for obj_str in objects:
                try:
                    entries.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    continue

            return entries, container_id
    except Exception as e:
        logger.error(f"Error processing catalog: {e}")
        return None, container_id


def get_bundle_images(entries, channel=None):
    """Extract relatedImages from the latest bundle in the catalog."""
    bundles = [e for e in entries if e.get("schema") == "olm.bundle"]
    if not bundles:
        return {}

    if channel:
        # Find bundle name from channel
        channels = [
            e for e in entries if e.get("schema") == "olm.channel" and e.get("name") == channel
        ]
        if channels:
            channel_entries = channels[0].get("entries", [])
            if channel_entries:
                bundle_name = channel_entries[-1].get("name", "")
                matching = [b for b in bundles if b.get("name") == bundle_name]
                if matching:
                    bundles = matching

    # Use the last bundle (typically latest)
    bundle = bundles[-1]
    images = {}
    for img_entry in bundle.get("relatedImages", []):
        image_ref = img_entry.get("image", "")
        if not image_ref:
            continue
        image_repo = image_ref.split("/")[-1].split("@")[0]
        git_repo = IMAGE_REPO_TO_GIT_REPO.get(image_repo)
        if git_repo:
            images[image_repo] = {"image_ref": image_ref, "git_repo": git_repo}

    return images


def get_upstream_commit(image_ref):
    """Extract upstream commit SHA from an image's labels or /kodata/HEAD."""
    # Try labels first (faster)
    inspect_out = podman_run(["inspect", "--format", "json", image_ref])
    if not inspect_out:
        # Need to pull first
        podman_run(["pull", "-q", image_ref])
        inspect_out = podman_run(["inspect", "--format", "json", image_ref])

    if inspect_out:
        try:
            inspected = json.loads(inspect_out)
            if inspected:
                labels = inspected[0].get("Config", {}).get("Labels", {})
                upstream_ref = labels.get("upstream-vcs-ref")
                if upstream_ref:
                    return upstream_ref
        except (json.JSONDecodeError, IndexError):
            pass

    # Fallback: extract /kodata/HEAD
    container_id = podman_run(["create", "-q", image_ref])
    if not container_id:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            head_file = os.path.join(tmpdir, "HEAD")
            result = subprocess.run(
                ["podman", "cp", f"{container_id}:/kodata/HEAD", head_file],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and os.path.exists(head_file):
                with open(head_file) as f:
                    return f.read().strip()
    finally:
        podman_run(["rm", container_id])

    return None


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_index_images(old_image, new_image, github_token, jira_token, jira_email, jira_url):
    """Compare two index images and return per-repo change data."""
    logger.info(f"Comparing:\n  old: {old_image}\n  new: {new_image}")

    # Extract catalog entries from both images
    old_entries, old_cid = extract_catalog_bundles(old_image)
    new_entries, new_cid = extract_catalog_bundles(new_image)

    # Cleanup catalog containers
    for cid in [old_cid, new_cid]:
        if cid:
            podman_run(["rm", cid])

    if not old_entries or not new_entries:
        logger.error("Failed to extract catalog from one or both images")
        return {}

    old_images = get_bundle_images(old_entries)
    new_images = get_bundle_images(new_entries)

    # Find repos present in both and group by git repo
    repo_images = {}  # git_repo -> {"old_refs": [...], "new_refs": [...]}
    for img_repo, info in new_images.items():
        git_repo = info["git_repo"]
        if not git_repo:
            continue
        if git_repo not in repo_images:
            repo_images[git_repo] = {"old_refs": [], "new_refs": []}
        repo_images[git_repo]["new_refs"].append(info["image_ref"])

    for img_repo, info in old_images.items():
        git_repo = info["git_repo"]
        if git_repo and git_repo in repo_images:
            repo_images[git_repo]["old_refs"].append(info["image_ref"])

    # For each repo, get upstream commits and compare
    changes = {}
    created_containers = []

    for git_repo, refs in repo_images.items():
        if not refs["old_refs"] or not refs["new_refs"]:
            logger.info(f"Skipping {git_repo}: missing old or new image")
            continue

        # Use first image from each to get upstream commit
        logger.info(f"Processing {git_repo}...")
        old_ref = refs["old_refs"][0]
        new_ref = refs["new_refs"][0]

        # Pull images
        podman_run(["pull", "-q", old_ref])
        podman_run(["pull", "-q", new_ref])

        old_commit = get_upstream_commit(old_ref)
        new_commit = get_upstream_commit(new_ref)

        if not old_commit or not new_commit:
            logger.warning(f"  Could not extract commits for {git_repo}")
            continue

        if old_commit == new_commit:
            logger.info(f"  No change in {git_repo} (same commit {old_commit[:8]})")
            continue

        logger.info(f"  {git_repo}: {old_commit[:8]} -> {new_commit[:8]}")

        # Get commits between old and new, filter out noise
        all_commits = get_compare(git_repo, old_commit, new_commit, github_token)
        commits = [c for c in all_commits if not is_noise_commit(c.get("message", ""))]
        skipped = len(all_commits) - len(commits)
        logger.info(f"  Found {len(all_commits)} commits, kept {len(commits)} (filtered {skipped} noise)")

        if not commits:
            logger.info(f"  All commits were noise, skipping {git_repo}")
            continue

        # Get PRs for commits (bulk lookup — much faster)
        pull_requests = get_prs_for_commits(git_repo, commits, github_token)

        logger.info(f"  Found {len(pull_requests)} PRs")

        # Extract JIRA tickets from commit messages AND PR titles
        jira_keys = extract_jira_keys(commits)
        for pr in pull_requests:
            jira_keys.extend(JIRA_PATTERN.findall(pr.get("title", "")))
        jira_keys = sorted(set(jira_keys))
        jira_tickets = []
        if jira_keys:
            logger.info(f"  Found JIRA references: {', '.join(jira_keys)}")
            jira_tickets = fetch_jira_details(jira_keys, jira_token, jira_email, jira_url)

        # Tag each commit with its JIRA keys for inline display
        jira_by_key = {t["key"]: t for t in jira_tickets}
        for c in commits:
            commit_keys = JIRA_PATTERN.findall(c.get("message", ""))
            c["jira_keys"] = [k for k in commit_keys if k in jira_by_key]

        changes[git_repo] = {
            "old_commit": old_commit,
            "new_commit": new_commit,
            "compare_url": f"https://github.com/{git_repo}/compare/{old_commit[:12]}...{new_commit[:12]}",
            "commits": commits,
            "commit_count": len(commits),
            "pull_requests": pull_requests,
            "jira_tickets": jira_tickets,
        }

    return changes


def load_cached(output_path):
    """Load previously cached comparison results."""
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare index images and extract commit/PR/JIRA data")
    parser.add_argument("--input", type=str, default="data/releases.json", help="Path to releases.json")
    parser.add_argument("--output", type=str, default="data/comparisons.json", help="Output path")
    parser.add_argument("--limit", type=int, default=5, help="Max releases to process")
    parser.add_argument("--jira-url", type=str, default=DEFAULT_JIRA_URL, help="JIRA base URL")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached results")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    jira_token = os.environ.get("JIRA_TOKEN")
    jira_email = os.environ.get("JIRA_EMAIL")

    if not github_token:
        logger.warning("No GITHUB_TOKEN set. API rate limits will be very low.")
    if not jira_token:
        logger.warning("No JIRA_TOKEN set. JIRA ticket details will not be fetched.")
    if jira_token and not jira_email:
        logger.warning("No JIRA_EMAIL set. Using Bearer auth (Jira Server). Set JIRA_EMAIL for Jira Cloud.")

    # Load releases
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    releases = data.get("releases", [])
    upcoming = data.get("upcoming", [])
    all_releases = upcoming + releases  # process upcoming first

    # Load cache
    cached = {} if args.no_cache else load_cached(args.output)

    processed = 0
    for release in all_releases:
        if processed >= args.limit:
            break

        pr_number = str(release.get("pr_number", ""))
        old_image = release.get("old_index_image", "")
        new_image = release.get("new_index_image", "")

        if not old_image or not new_image:
            logger.info(f"PR #{pr_number}: skipping (missing index images)")
            continue

        if pr_number in cached:
            # Re-process if JIRA tokens are set but cached entry has no JIRA data
            repos = cached[pr_number].get("repos", {})
            has_jira = any(
                repo_data.get("jira_tickets") for repo_data in repos.values()
            )
            if has_jira or not jira_token:
                logger.info(f"PR #{pr_number}: using cached comparison")
                continue
            logger.info(f"PR #{pr_number}: re-processing to fetch JIRA data")

        logger.info(f"PR #{pr_number}: comparing images...")
        try:
            changes = compare_index_images(
                old_image, new_image, github_token, jira_token, jira_email, args.jira_url
            )
            cached[pr_number] = {
                "pr_number": int(pr_number),
                "old_image": old_image,
                "new_image": new_image,
                "repos": changes,
            }
            processed += 1
        except Exception as e:
            logger.error(f"PR #{pr_number}: comparison failed: {e}")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cached, f, indent=2)
    logger.info(f"Written {args.output} ({len(cached)} comparisons)")


if __name__ == "__main__":
    main()
