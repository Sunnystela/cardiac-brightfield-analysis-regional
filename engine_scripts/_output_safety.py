"""Shared canonical-output-directory safety guard (round 27).

Round 26 found that re-running scripts/beat_episode_reconstruction.py
with its DEFAULT (canonical, un-versioned) --output-dir -- at a moment
when the manual reference files already held newer region definitions
-- silently overwrote a canonical results/ snapshot that an earlier
manual-QC round had intentionally frozen (see
results/beat_event_episodes/manual_validation_log.json's
round_26_entries for the full incident and recovery). The versioned
(_v2/_v3/...) output-directory convention already used by several
scripts is not, by itself, a safeguard -- it only helps if every run
remembers to pass a non-default --output-dir.

Going forward, no diagnostic/reconstruction script may write to a
registered CANONICAL directory by default:

  - Omitting --output-dir always writes to a fresh, timestamped SCRATCH
    directory under results/_scratch_runs/ -- never the canonical path.
  - Writing to a versioned directory (e.g. results/beat_event_episodes_v2)
    is always allowed freely (that is exactly what the versioning
    convention is for).
  - Writing directly to the canonical directory requires BOTH pointing
    --output-dir at that exact path AND passing --allow-canonical-write
    explicitly -- an unambiguous, deliberate action, meant to happen
    only after reviewing scratch/versioned output first.

This module only implements the guard/CLI-argument plumbing; it does not
decide what counts as "validated" -- that judgment (reviewing the
scratch/versioned output, comparing against manual QC) remains a human
step before ever passing --allow-canonical-write.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_ROOT = REPO_ROOT / "results" / "_scratch_runs"


def default_scratch_dir(label: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return SCRATCH_ROOT / f"{label}_{ts}"


def add_output_dir_args(parser: argparse.ArgumentParser, canonical_dir: Path, label: str) -> None:
    """Adds --output-dir and --allow-canonical-write to `parser`. Call
    resolve_output_dir(args, canonical_dir, label) after parsing to get
    the actual directory to use and have the guard enforced."""
    try:
        canonical_display = canonical_dir.relative_to(REPO_ROOT)
    except ValueError:
        canonical_display = canonical_dir
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            f"Where to write results. Default (omit this flag): a fresh timestamped scratch "
            f"directory under results/_scratch_runs/ -- NEVER the canonical directory. Pass a "
            f"versioned directory (e.g. {canonical_display}_v2) for a validated re-evaluation "
            f"you intend to keep. Pass the canonical path itself ({canonical_display}) ONLY "
            f"together with --allow-canonical-write, after reviewing scratch/versioned output."
        ),
    )
    parser.add_argument(
        "--allow-canonical-write", action="store_true",
        help=(
            "Required, in addition to --output-dir pointing at the canonical directory, to let "
            "this run write directly into it. Omit for every exploratory/diagnostic run -- this "
            "is the guard that round 26's accidental overwrite (see "
            "results/beat_event_episodes/manual_validation_log.json) did not have."
        ),
    )


def resolve_output_dir(args: argparse.Namespace, canonical_dir: Path, label: str) -> Path:
    canonical_resolved = canonical_dir.resolve()
    if args.output_dir is None:
        out = default_scratch_dir(label)
        print(f"[output-safety] --output-dir not given -- writing to a SCRATCH directory: {out}")
        print("[output-safety] review this output; if you intend to keep it, rerun with an "
              "explicit versioned --output-dir, or the canonical path + --allow-canonical-write.")
        return out

    requested_resolved = args.output_dir.resolve()
    if requested_resolved == canonical_resolved:
        if not args.allow_canonical_write:
            raise SystemExit(
                f"[output-safety] REFUSING to write directly to the canonical directory "
                f"({canonical_resolved}) without --allow-canonical-write.\n"
                f"This exact mistake (default-output-dir accidentally overwriting a frozen "
                f"canonical snapshot) happened in round 26 -- see "
                f"results/beat_event_episodes/manual_validation_log.json's round_26_entries.\n"
                f"Write to a scratch or versioned directory first, validate the output, then "
                f"rerun with:\n"
                f"    --output-dir {canonical_resolved} --allow-canonical-write\n"
                f"once you deliberately intend to promote it -- and regenerate "
                f"results/CANONICAL_MANIFEST.json (scripts/generate_canonical_manifest.py) "
                f"afterward."
            )
        print(f"[output-safety] --allow-canonical-write set -- writing directly to the canonical "
              f"directory: {canonical_resolved}")
        print("[output-safety] remember to rerun scripts/generate_canonical_manifest.py after this.")
    return args.output_dir
