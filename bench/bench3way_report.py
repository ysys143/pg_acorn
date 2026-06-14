"""Extract the 3-way comparison table from results_3way_pg.json +
results_3way_qdrant.json: per engine/config/selectivity, the matched-recall
(~0.95) operating point (recall, median/p95/min latency) and peak throughput
(QPS, concurrency-at-peak).  Prints a markdown table for REPORT_3way.md.

  uv run --with numpy python3 bench/bench3way_report.py
"""
import json
import os

HERE = os.path.dirname(__file__)
SELS = [1, 2, 5, 10, 20]
TARGET = 0.94   # matched-recall floor


def pick(cells):
    """Lowest-ef cell reaching recall >= TARGET; else the highest-recall cell."""
    ok = [c for c in cells if c["recall"] >= TARGET]
    if ok:
        return min(ok, key=lambda c: (c["ef"] if c["ef"] else 0))
    return max(cells, key=lambda c: c["recall"])


def peak_qps(by_conc):
    best = max(by_conc.items(), key=lambda kv: kv[1]["qps"])
    return best[1]["qps"], int(best[0])


def main():
    pg = json.load(open(os.path.join(HERE, "results_3way_pg.json")))
    qd = json.load(open(os.path.join(HERE, "results_3way_qdrant.json")))

    # latency/recall per (engine, sel)
    rows = []
    for name, persel in pg.get("latency", {}).items():
        for sel in SELS:
            cells = persel.get(str(sel))
            if not cells:
                continue
            c = pick(cells)
            rows.append((name, sel, c))
    for sel in SELS:
        cells = qd.get("latency", {}).get(str(sel))
        if cells:
            rows.append(("qdrant", sel, pick(cells)))

    print("\n## Matched-recall (~0.95) — recall / median / p95 / min ms, by engine x sel\n")
    print("| engine | sel | ef | recall | med ms | p95 ms | min ms |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, sel, c in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"| {name} | {sel}% | {c['ef']} | {c['recall']:.3f} | "
              f"{c['med_ms']} | {c['p95_ms']} | {c['min_ms']} |")

    print("\n## Peak throughput (QPS) at the matched-recall ef\n")
    print("| engine | sel | ef | peak QPS | @conc | QPS by conc |")
    print("|---|---:|---:|---:|---:|---|")
    pg_tp = pg.get("throughput", {})
    for name, d in pg_tp.items():
        q, cc = peak_qps(d["by_conc"])
        series = " ".join(f"c{k}={v['qps']:.0f}" for k, v in d["by_conc"].items())
        print(f"| {name} | {d['sel']}% | {d.get('ef')} | {q:.0f} | {cc} | {series} |")
    for sel, d in qd.get("throughput", {}).items():
        q, cc = peak_qps(d["by_conc"])
        series = " ".join(f"c{k}={v['qps']:.0f}" for k, v in d["by_conc"].items())
        print(f"| qdrant | {sel}% | {d.get('ef')} | {q:.0f} | {cc} | {series} |")


if __name__ == "__main__":
    main()
