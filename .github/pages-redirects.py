#!/usr/bin/env python3
"""Retire arcadia-1.github.io/ADCToolbox in favour of adctoolbox.tokenzhang.com/doc/.

The manual used to be published here, and links to it live on in old PyPI pages, papers and bookmarks. So instead of
the manual, publish one small page for every page it has, forwarding to the same page on the site with its query and
#anchor; a 404 page that does the same for any other path; and the Sphinx inventory, so intersphinx keeps resolving.

    python .github/pages-redirects.py python/docs/build/html python/docs/build/redirects
"""
import html
import json
import shutil
import sys
from pathlib import Path

SITE = "https://adctoolbox.tokenzhang.com/doc/"
PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>The ADCToolbox manual has moved</title>
<link rel="canonical" href="{href}">
<meta http-equiv="refresh" content="0; url={href}">
<script>location.replace({js} + location.search + location.hash)</script>
<p>The ADCToolbox manual now lives at <a href="{href}">{href}</a>.</p>
</html>
"""
# GitHub Pages serves this for any path it has no file for; the path after /ADCToolbox/ is the manual's own
MISSING = PAGE.replace("location.replace({js}", "location.replace({js} + location.pathname.replace(/^\\/ADCToolbox\\/?/, '')")


def landing(rel: str) -> str:
    """Where a page of the old manual is now, spelled the way the site serves it without a redirect of its own."""
    if rel == "index.html" or rel.endswith("/index.html"):
        return SITE + rel[: -len("index.html")]
    return SITE + rel.removesuffix(".html")


def main(built: Path, out: Path) -> None:
    shutil.rmtree(out, ignore_errors=True)
    pages = sorted(built.rglob("*.html"))
    for page in pages:
        rel = page.relative_to(built).as_posix()
        url = landing(rel)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(PAGE.format(href=html.escape(url), js=json.dumps(url)), encoding="utf-8")
    (out / "404.html").write_text(MISSING.format(href=html.escape(SITE), js=json.dumps(SITE)), encoding="utf-8")
    shutil.copy(built / "objects.inv", out / "objects.inv")
    (out / ".nojekyll").touch()
    print(f"{len(pages)} pages of the manual now forward to {SITE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
