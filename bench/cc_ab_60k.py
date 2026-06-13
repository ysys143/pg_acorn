#!/usr/bin/env python3
"""Before/after cache-latency A/B at the money cells, 60K fixture.

Reuses cc_debug_60k's fixture builder + sweep but trims EFS to the money
cells (1600 destabilizes the docker VM) and tags results so a host-side
binary swap (parent-commit .so = before, HEAD .so = after) can be compared.

    python3 -u /workspace/bench/cc_ab_60k.py build              # once
    python3 -u /workspace/bench/cc_ab_60k.py sweep <label>      # per binary

Writes /tmp/cc_ab_<label>.json with {(mode,sel,ef): [min,med,bufs,rows]}.
"""
import json
import sys

import cc_debug_60k as base

# Money cells only; cosine <=> like the base harness (indexes are cosine).
base.EFS = [100, 200, 400, 800]


def sweep(label):
    results = {}
    for mode in ("inline", "cache"):
        base.run_mode(mode, results)
    # serialize (tuple keys -> "mode|sel|ef")
    out = {f"{m}|{s}|{e}": list(v) for (m, s, e), v in results.items()}
    with open(f"/tmp/cc_ab_{label}.json", "w") as f:
        json.dump(out, f)
    print(f"\n[ab] wrote /tmp/cc_ab_{label}.json", flush=True)
    for sel, _ in base.SELS:
        for ef in base.EFS:
            vi = results.get(("inline", sel, ef))
            vc = results.get(("cache", sel, ef))
            if vi and vc:
                print(f"sel={sel}% ef={ef}: inline={vi[0]:.2f}ms "
                      f"cache={vc[0]:.2f}ms cache/inline={vc[0] / vi[0]:.2f}x",
                      flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        base.build()
    elif cmd == "sweep":
        sweep(sys.argv[2])
    else:
        raise SystemExit("usage: cc_ab_60k.py build | sweep <label>")
