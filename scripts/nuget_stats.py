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

# Order fixes each package's color, so append new packages at the end.
# HealthData.Interop.Fhir became a meta-package in v1.4.2; the three
# HealthData.Interop.* components below were split out of it (first
# published 2026-09, v1.0.1) and have no history before that.
PACKAGES = [
    "HealthData.Interop.Fhir",
    "Quant.Infra.Net",
    "LightningLocationSystemDataAnalyzer-LLDSA",
    "HealthData.Interop.Abstractions",
    "HealthData.Interop.Logging.Extensions",
    "HealthData.Interop.Logging.Serilog",
]

# stable color per package (tab10 order); keep len(PALETTE) >= len(PACKAGES)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
SHORT_NAMES = {
    "LightningLocationSystemDataAnalyzer-LLDSA": "LLDSA (lightning)",
}

SEARCH_URL = "https://azuresearch-usnc.nuget.org/query"
REGISTRATION_URL = "https://api.nuget.org/v3/registration5-semver1/"
REQUEST_TIMEOUT = 30

# A version's downloads within this many days of publishing count as a release
# burst (mirrors / scanners fetch every new version). Keep in sync with
# BURST_DAYS in stats/index.html.
BURST_DAYS = 4

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "stats" / "history.jsonl"
CHART_PATH = REPO_ROOT / "charts" / "nuget-downloads.png"
ORGANIC_CHART_PATH = REPO_ROOT / "charts" / "nuget-organic.png"


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
    deployments. `q=+{id}` is not exact either: for HealthData.Interop.Abstractions
    it ranks the HealthData.Interop.Fhir meta-package first, because that
    package's description names its components. `q=packageid:{id}` is the
    exact-id filter.
    """
    params = urllib.parse.urlencode({"q": "packageid:" + package_id, "take": 1})
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


def fetch_publish_dates(package_id: str) -> dict:
    """{version: "YYYY-MM-DD"} from the registration API (unlisted versions skipped)."""
    index = _http_get_json(REGISTRATION_URL + package_id.lower() + "/index.json")
    dates = {}
    for page in index.get("items", []):
        items = page.get("items")
        if items is None:  # large packages page their versions out
            items = _http_get_json(page["@id"]).get("items", [])
        for it in items:
            ce = it.get("catalogEntry", {})
            published = ce.get("published", "")
            # unlisted versions report a 1900-01-01 publish date
            if ce.get("version") and published[:4] > "1900":
                dates[ce["version"]] = published[:10]
    return dates


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

    fig = plt.figure(figsize=(11.0, 5.4))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 1, figure=fig, hspace=0.62, wspace=0.38,
                           left=0.07, right=0.985, top=0.83, bottom=0.27)

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
    # dates are categorical; label at most ~8 of them (always the latest)
    step = max(1, -(-len(dates) // 8))
    ticks = list(range(0, len(dates), step))
    if ticks[-1] != len(dates) - 1:
        if len(dates) - 1 - ticks[-1] < step / 2:
            ticks.pop()
        ticks.append(len(dates) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks])
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.grid(linewidth=0.4, alpha=0.5)
    # below the plot: 6+ series inside the axes would cover the lines
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=3, frameon=False)
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


def _days_between(a: str, b: str) -> int:
    return (datetime.strptime(b, "%Y-%m-%d") - datetime.strptime(a, "%Y-%m-%d")).days


def classify_daily(history: list, pkg: str, pub: dict) -> dict:
    """Split each daily delta into organic / release burst / old-version sweep.

    Mirrors classifyDaily() in stats/index.html:
      burst   = downloads to a version within BURST_DAYS of its publish date
      sweep   = downloads to superseded versions; when a sweep touches every old
                version at once, the same floor is removed from the latest too
      organic = what remains on the latest version published by that day
    Returns {date: {"organic": n, "burst": n, "sweep": n}}.
    """
    out = {}
    prev = None
    for h in history:
        p = h.get("packages", {}).get(pkg)
        if not p or not p.get("versions"):
            continue
        if prev is not None:
            date = h["date"]
            released = [v for v in p["versions"] if v in pub and pub[v] <= date]
            latest = max(released, key=lambda v: (pub[v], _semver_key(v))) if released else None
            row = {"organic": 0, "burst": 0, "sweep": 0}
            latest_delta, old = 0, []
            for v, count in p["versions"].items():
                d = max(0, count - prev["versions"].get(v, 0))
                age = _days_between(pub[v], date) if v in pub else None
                if age is not None and 0 <= age <= BURST_DAYS:
                    row["burst"] += d
                elif v == latest:
                    latest_delta = d
                else:
                    row["sweep"] += d
                    old.append(d)
            floor = min(old) if len(old) >= 2 and all(x > 0 for x in old) else 0
            row["organic"] = max(0, latest_delta - floor)
            row["sweep"] += latest_delta - row["organic"]
            rest = max(0, p["total"] - prev["total"]) - sum(row.values())
            if rest > 0:
                row["sweep"] += rest
            out[date] = row
        prev = p
    return out


def render_organic_chart(history: list, out_path: Path) -> None:
    """Cumulative *organic* downloads per package since tracking began.

    Needs version publish dates from the registration API; if they can't be
    fetched the chart is skipped (the total-downloads chart still renders).
    """
    latest = history[-1]
    pkgs = [p for p in PACKAGES if p in latest.get("packages", {})]
    dates = [h["date"] for h in history]

    series = {}
    for pkg in pkgs:
        try:
            pub = fetch_publish_dates(pkg)
        except Exception as exc:
            print("  ! organic chart skipped: publish dates for " + pkg + ": " + str(exc), file=sys.stderr)
            return
        daily = classify_daily(history, pkg, pub)
        if not daily:
            continue
        running, points = 0, []
        for d in dates:
            if d in daily:
                running += daily[d]["organic"]
                points.append(running)
            else:
                # before the package's first delta there is nothing to accumulate
                points.append(running if points and points[-1] is not None else None)
        series[pkg] = points

    fig = plt.figure(figsize=(11.0, 5.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.07, 0.27, 0.915, 0.56])
    for pkg, points in series.items():
        ax.plot(dates, points, marker="o", markersize=4, linewidth=1.8,
                color=_color(pkg, pkgs), label=pkg + " (" + format(points[-1] or 0, ",") + ")")
    ax.set_title("Cumulative organic downloads since " + dates[0]
                 + " (release bursts & mirror sweeps removed; heuristic)", fontsize=11)
    ax.set_xlabel("UTC date (snapshot day)", fontsize=9)
    ax.set_ylabel("Organic downloads (cumulative)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    step = max(1, -(-len(dates) // 8))
    ticks = list(range(0, len(dates), step))
    if ticks[-1] != len(dates) - 1:
        if len(dates) - 1 - ticks[-1] < step / 2:
            ticks.pop()
        ticks.append(len(dates) - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([dates[i] for i in ticks])
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.grid(linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=3, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle("NuGet organic downloads — memoryfraction", fontsize=14, weight="bold", y=0.98)
    fig.text(0.985, 0.01,
             "organic = downloads to the latest version > " + str(BURST_DAYS)
             + " days after release, minus mirror-sweep floor · updated " + latest["date"] + " (UTC)",
             ha="right", fontsize=7.5, color="#777")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print("  organic chart -> " + str(out_path))


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
    render_organic_chart(history, ORGANIC_CHART_PATH)
    print("done.")


if __name__ == "__main__":
    main()





