"""
Upscale every image of a category (or a list of generation ids) with Real-ESRGAN, the same way the Studio's
Upscale bar does, and record the result in generations.db.

    python upscale_batch.py --category Nature                  # 4K (3840 px long side)
    python upscale_batch.py --category Nature --targets 4k 8k  # both: <stem>_3840px.png and <stem>_7680px.png
    python upscale_batch.py --ids 31 32 33 --targets 2k

Files land next to the originals (outputs/<category>/<stem>_<px>px.png). The database row keeps the largest
target as its upscale (the columns upscaled_path, upscaled_width ... hold one result); every other target is
listed in extra upscale files next to it. Already existing target files are skipped, so the command can be
re-run. Runs in its own process with the x4plus network (about 1.5 GB of VRAM), so the web app can stay up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

import db
from upscaler import TARGETS, Upscaler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", help="upscale every image generation of this category")
    ap.add_argument("--ids", type=int, nargs="*", help="generation ids instead of a category")
    ap.add_argument("--targets", nargs="+", default=["4k"], help="2k, 4k, 8k or a pixel count for the long side")
    ap.add_argument("--force", action="store_true", help="redo targets whose file already exists")
    args = ap.parse_args()

    targets = []
    for t in args.targets:
        targets.append(TARGETS[t.lower()] if t.lower() in TARGETS else int(t))
    targets = sorted(set(targets))

    if args.ids:
        rows = [r for i in args.ids if (r := db.get(i))]
    elif args.category:
        rows = db.fetch_filtered(limit=100000, category=args.category, kind="image")
        rows.sort(key=lambda r: r["id"])
    else:
        sys.exit("give --category <name> or --ids <id ...>")
    rows = [r for r in rows if Path(r["image_path"]).exists()]
    if not rows:
        sys.exit("nothing to upscale")

    up = Upscaler()
    print(f"{up.name}: {len(rows)} images x {len(targets)} target(s) {targets}")
    t_all = time.time()
    done = skipped = 0
    for n, rec in enumerate(rows, 1):
        src = Path(rec["image_path"])
        img = None
        extras = []
        for px in targets:
            dest = src.with_name(f"{src.stem}_{px}px.png")
            if dest.exists() and not args.force:
                skipped += 1
                fields = None
                out_size = Image.open(dest).size
                secs, passes = None, None
            else:
                img = img or Image.open(src).convert("RGB")
                out, info = up.upscale_to(img, px)
                out.save(dest)
                out_size, secs, passes = out.size, info["seconds"], info["passes"]
                done += 1
            fields = {
                "upscale_target": px, "upscaled_width": out_size[0], "upscaled_height": out_size[1], "upscaled_path": str(dest),
                "upscaled_size_kb": round(dest.stat().st_size / 1024, 1), "upscale_seconds": secs, "upscale_passes": passes,
                "upscaler": up.name,
            }
            extras.append(fields)
            print(f"[{n:>3}/{len(rows)}] #{rec['id']} {src.name} -> {out_size[0]}x{out_size[1]}"
                  + (f" in {secs:.1f}s ({passes} pass{'es' if passes != 1 else ''})" if secs is not None else " (already there)"), flush=True)
        # the largest target is the row's upscale; the others are kept in extra_json["upscales"]
        largest = extras[-1]
        row_fields = {k: v for k, v in largest.items() if v is not None}
        if len(extras) > 1:
            try:
                extra = json.loads(rec.get("extra_json") or "{}")
            except json.JSONDecodeError:
                extra = {}
            extra["upscales"] = [{"target_px": e["upscale_target"], "width": e["upscaled_width"], "height": e["upscaled_height"],
                                  "path": e["upscaled_path"], "size_kb": e["upscaled_size_kb"]} for e in extras]
            row_fields["extra_json"] = json.dumps(extra)
        db.update(rec["id"], row_fields)
    print(f"\n{done} upscales written, {skipped} skipped, {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
