"""Writing result rows with provenance (CSV for spreadsheets, JSON with the full provenance block)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ._engine import provenance


def save_results(rows: list[dict], output_dir: Path, name: str, extra_provenance: dict | None = None) -> dict:
    """Write `<name>.csv`, `<name>.json` and `<name>_provenance.json` into `output_dir`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(extra_provenance)
    try:
        from cardio_organoid.morphology.features import PILOT_DISCLAIMER
        prov["morphology_disclaimer"] = PILOT_DISCLAIMER
    except ImportError:
        pass
    cols = list(dict.fromkeys(k for r in rows for k in r))
    csv_path = output_dir / f"{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    json_path = output_dir / f"{name}.json"
    json_path.write_text(json.dumps({"provenance": prov, "rows": rows}, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    prov_path = output_dir / f"{name}_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "provenance": str(prov_path),
            "engine_unmodified": prov["engine_verification"]["engine_unmodified"],
            "config_matches_validated_runs": prov["config_matches_validated_runs"]}
