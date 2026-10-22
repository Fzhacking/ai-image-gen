"""
SQLite store for every generated image: prompt, settings, timings, and GPU telemetry.

Database file: generations.db next to this script.

    python db.py            # print the last 20 generations
    python db.py 12         # print full details of generation #12
    python db.py stats      # totals and averages
"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "generations.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT    NOT NULL,
    image_path          TEXT    NOT NULL,
    file_size_kb        REAL,

    user_prompt         TEXT    NOT NULL,
    final_prompt        TEXT    NOT NULL,
    negative_prompt     TEXT,
    parser_notes        TEXT,

    model_id            TEXT,
    width               INTEGER,
    height              INTEGER,
    steps               INTEGER,
    guidance            REAL,
    seed                INTEGER,
    style               TEXT,
    batch_index         INTEGER,
    batch_size          INTEGER,

    inference_seconds   REAL,
    save_seconds        REAL,
    total_seconds       REAL,
    steps_per_second    REAL,

    gpu_name            TEXT,
    vram_total_gb       REAL,
    vram_peak_gb        REAL,
    vram_reserved_gb    REAL,
    compute_capability  TEXT,
    driver_version      TEXT,
    cuda_version        TEXT,
    torch_version       TEXT,
    diffusers_version   TEXT,

    temp_before_c       REAL,
    temp_after_c        REAL,
    power_before_w      REAL,
    power_after_w       REAL,
    power_limit_w       REAL,
    gpu_util_after_pct  REAL,
    sm_clock_mhz        REAL,
    mem_clock_mhz       REAL,
    fan_after_pct       REAL,
    pstate              TEXT,

    upscale_target      INTEGER,
    upscaled_width      INTEGER,
    upscaled_height     INTEGER,
    upscaled_path       TEXT,
    upscaled_size_kb    REAL,
    upscale_seconds     REAL,
    upscale_passes      INTEGER,
    upscaler            TEXT,

    extra_json          TEXT
);
CREATE INDEX IF NOT EXISTS idx_generations_created ON generations(created_at);
"""


NEW_COLUMNS = {
    "upscale_target": "INTEGER",
    "upscaled_width": "INTEGER",
    "upscaled_height": "INTEGER",
    "upscaled_path": "TEXT",
    "upscaled_size_kb": "REAL",
    "upscale_seconds": "REAL",
    "upscale_passes": "INTEGER",
    "upscaler": "TEXT",
}


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    existing = {r[1] for r in con.execute("PRAGMA table_info(generations)")}
    for name, kind in NEW_COLUMNS.items():
        if name not in existing:
            con.execute(f"ALTER TABLE generations ADD COLUMN {name} {kind}")
    return con


def insert(record: dict[str, Any]) -> int:
    """Insert one generation. Unknown keys go into extra_json. Returns the new row id."""
    with connect() as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(generations)")} - {"id"}
        known = {k: v for k, v in record.items() if k in cols}
        extra = {k: v for k, v in record.items() if k not in cols}
        if extra:
            known["extra_json"] = json.dumps(extra, default=str)
        keys = list(known)
        sql = f"INSERT INTO generations ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})"
        cur = con.execute(sql, [known[k] for k in keys])
        return cur.lastrowid


def update(row_id: int, fields: dict[str, Any]) -> None:
    """Set the given known columns on one generation (used when an image is upscaled later)."""
    with connect() as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(generations)")} - {"id"}
        known = {k: v for k, v in fields.items() if k in cols}
        if not known:
            return
        assignments = ", ".join(f"{k} = ?" for k in known)
        con.execute(f"UPDATE generations SET {assignments} WHERE id = ?", [*known.values(), row_id])


def fetch(limit: int = 50, search: str = "") -> list[dict[str, Any]]:
    with connect() as con:
        if search:
            rows = con.execute(
                "SELECT * FROM generations WHERE user_prompt LIKE ? OR final_prompt LIKE ? "
                "ORDER BY id DESC LIMIT ?", (f"%{search}%", f"%{search}%", limit),
            )
        else:
            rows = con.execute("SELECT * FROM generations ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]


def get(row_id: int) -> dict[str, Any] | None:
    with connect() as con:
        r = con.execute("SELECT * FROM generations WHERE id = ?", (row_id,)).fetchone()
        return dict(r) if r else None


def stats() -> dict[str, Any]:
    with connect() as con:
        r = con.execute(
            "SELECT COUNT(*) AS images, "
            "ROUND(SUM(total_seconds), 1) AS total_seconds, "
            "ROUND(AVG(inference_seconds), 2) AS avg_inference_seconds, "
            "ROUND(MIN(inference_seconds), 2) AS fastest_seconds, "
            "ROUND(MAX(inference_seconds), 2) AS slowest_seconds, "
            "ROUND(AVG(steps_per_second), 2) AS avg_steps_per_second, "
            "ROUND(AVG(vram_peak_gb), 2) AS avg_vram_peak_gb, "
            "ROUND(MAX(vram_peak_gb), 2) AS max_vram_peak_gb, "
            "ROUND(MAX(temp_after_c), 0) AS max_temp_c, "
            "ROUND(MAX(power_after_w), 0) AS max_power_w, "
            "SUM(CASE WHEN upscaled_path IS NOT NULL THEN 1 ELSE 0 END) AS upscaled_images, "
            "ROUND(SUM(upscale_seconds), 1) AS upscale_seconds, "
            "MIN(created_at) AS first, MAX(created_at) AS last "
            "FROM generations"
        ).fetchone()
        return dict(r)


def _print_row(r: dict[str, Any]) -> None:
    for k, v in r.items():
        if v is not None:
            print(f"  {k:<20} {v}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "stats":
        _print_row(stats())
    elif arg.isdigit():
        row = get(int(arg))
        _print_row(row) if row else print(f"no generation #{arg}")
    else:
        for r in fetch(20):
            size = (f"{r['width']}x{r['height']}->{r['upscaled_width']}x{r['upscaled_height']}"
                    if r.get('upscaled_width') else f"{r['width']}x{r['height']}")
            print(f"#{r['id']:<4} {r['created_at'][:19]}  {size}  "
                  f"{r['steps']:>2} steps  {r['inference_seconds']:>5.1f}s  "
                  f"{r['vram_peak_gb']:>4.1f} GB  seed {r['seed']}  | {r['user_prompt'][:60]}")
