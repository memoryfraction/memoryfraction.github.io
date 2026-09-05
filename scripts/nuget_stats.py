#!/usr/bin/env python3
"""NuGet download stats: daily snapshot + auto-rendered chart.

Why this exists
---------------
NuGet's public data source only exposes *lifetime cumulative* download counts
(per package and per version). It provides NO official time series at all
(no daily / weekly / monthly buckets; the legacy v3 stats endpoints are gone).

So the best we can do is:
  1. snapshot the cumulative values once a day into stats/history.jsonl
  2. derive "downloads in period" as the delta between consecutive snapshots
  3. render a chart (per-version bars + cumulative trend once history builds up)

The snapshot file is the single source of truth for all charts.

Storage (all committed back to the repo, all plain text / PNG):
  stats/history.jsonl          one JSON record per UTC day (deduped by date)
  charts/nuget-downloads.png   the chart embedded in README.md

Usage:
  python scripts/nuget_stats.py            # snapshot + render (daily cron job)
  python scripts/nuget_stats.py --render   # re-render chart from existing history
"""

from __future__ import annotations

import argparse
import textwrap
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

PACKAGES = [
    "HealthData.Interop.Fhir",
    "Quant.Infra.Net",
    "LightningLocationSystemDataAnalyzer-LLDSA",
]

# stable color per package (tab10 order)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
SHORT_NAMES = {
    "LightningLocationSystemDataAnalyzer-LLDSA": "LLDSA (lightning)",
}

SEARCH_URL = "https://azuresearch-usnc.nuget.org/query"
REQUEST_TIMEOUT = 30

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "stats" / "history.jsonl"
CHART_PATH = REPO_ROOT / "charts" / "nuget-downloads.png"


# --------------------------------------------------------------------------
# data fetch
# --------------------------------------------------------------------------

def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nuget-stats-snapshot/1.0"})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_package(package_id: str) -> dict:
    """Fetch lifetime-cumulative download stats for one package.

    NOTE: the search endpoint does NOT honor the `packageIds` parameter in all
    deployments -- `q=+{id}` is the reliable exact-match form.
    """
    params = urllib.parse.urlencode({"q": "+" + package_id, "take": 1})
    data = _http_get_json(SEARCH_URL + "?" + params)
    hits = data.get("data") or []
    if not hits or hits[0].get("id", "").lower() != package_id.lower():
        raise RuntimeError("search did not return exact match for " + repr(package_id))
    hit = hits[0]
    versions = {}
    for v in hit.get("versions", []):
        versions[v["version"]] = int(v.get("downloads", 0))
    return {
        "total": int(hit.get("totalDownloads", 0)),
        "latestVersion": hit.get("version"),
        "versions": versions,
    }


# --------------------------------------------------------------------------
# history store
# --------------------------------------------------------------------------

def load_history(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print("  ! skipping corrupt history line: " + line[:80], file=sys.stderr)
    records.sort(key=lambda r: r.get("date", ""))
    return records


def save_history(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")


def snapshot() -> dict:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    packages = {}
    errors = {}
    for pkg in PACKAGES:
        try:
            packages[pkg] = fetch_package(pkg)
            p = packages[pkg]
            print("  ok  " + pkg + ": total=" + format(p["total"], ",") + " versions=" + str(len(p["versions"])))
        except Exception as exc:
            errors[pkg] = str(exc)
            print("  ERR " + pkg + ": " + str(exc), file=sys.stderr)
    if not packages:
        raise SystemExit("all package fetches failed; refusing to write an empty snapshot")
    return {
        "date": day,
        "fetchedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": SEARCH_URL,
        "note": ("lifetime cumulative downloads (NuGet provides no time series; "
                 "day-over-day deltas = downloads in that period)"),
        "packages": packages,
        "errors": errors or None,
    }


# --------------------------------------------------------------------------
# chart rendering
# --------------------------------------------------------------------------

def _semver_key(v: str):
    parts = []
    for chunk in v.split("."):
        m = re.match(r"(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    return parts


def _fmt_int(x, _pos=None):
    return format(int(x), ",")


def _color(pkg: str, pkgs: list) -> str:
    idx = pkgs.index(pkg) if pkg in pkgs else 0
    return PALETTE[idx % len(PALETTE)]


def render_chart(history: list, out_path: Path) -> None:
    """Render the cumulative-downloads trend as a single-panel chart.

    Why only one panel:
      - The NuGet search API returns a *lifetime-cumulative* count per
        package and per version, but it does NOT return a reliable
        per-version cumulative for the LATEST version of a package
        (it comes back 0 / empty). A per-package or per-version bar
        would therefore show fake zeros (e.g. Fhir 1.3.4 = 0, QIN
        1.5.3 = 0) even though those versions are in active use.
      - NuGet also has no official time-series API, so the only
        trustworthy series we can build is the daily-snapshot delta
        (cumulative at each UTC day, from stats/history.jsonl).
      - Hence: a single cumulative-trend line chart, one series per
        package. (NuGet has no per-version time series.)
    """
    if not history:
        raise SystemExit("no history records; nothing to render")

    latest = history[-1]
    pkgs = [p for p in PACKAGES if p in latest.get("packages", {})]
    if not pkgs:
        raise SystemExit("no packages found in latest snapshot")
    if len(history) < 2:
        raise SystemExit("need at least 2 snapshots to render a trend")

    fig = plt.figure(figsize=(11.0, 4.6))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.62, wspace=0.38,
                           left=0.07, right=0.985, top=0.80, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0])
    dates = [h["date"] for h in history]
    for pkg in pkgs:
        series = [h["packages"].get(pkg, {}).get("total") for h in history]
        if any(v is not None for v in series):
            ax.plot(dates, series, marker="o", markersize=4, linewidth=1.8,
                    color=_color(pkg, pkgs), label=pkg)
    ax.set_title("Cumulative downloads over time (daily snapshots; NuGet has no official series)",
                 fontsize=11)
    ax.set_xlabel("UTC date (snapshot day)", fontsize=9)
    ax.set_ylabel("Lifetime-cumulative downloads", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("NuGet downloads — memoryfraction", fontsize=14, weight="bold", y=0.98)
    fig.text(0.985, 0.01,
             "source: azuresearch-usnc.nuget.org (lifetime cumulative) · updated "
             + latest["date"] + " (UTC)",
             ha="right", fontsize=7.5, color="#777")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print("  chart -> " + str(out_path))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Snapshot NuGet download stats + render chart.")
    ap.add_argument("--render", action="store_true",
                    help="only re-render the chart from existing history")
    args = ap.parse_args()

    history = load_history(HISTORY_PATH)

    if not args.render:
        print("snapshotting NuGet download stats ...")
        rec = snapshot()
        history = [h for h in history if h.get("date") != rec["date"]] + [rec]
        history.sort(key=lambda r: r.get("date", ""))
        save_history(HISTORY_PATH, history)
        print("  history -> " + str(HISTORY_PATH) + " (" + str(len(history)) + " day(s))")
        if rec.get("errors"):
            print("  note: " + str(len(rec["errors"])) + " package(s) failed this run", file=sys.stderr)

    render_chart(history, CHART_PATH)
    print("done.")


if __name__ == "__main__":
    main()





