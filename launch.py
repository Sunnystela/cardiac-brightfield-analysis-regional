"""Cardiac bright-field standalone analysis -- command-line entry point (no GUI).

    python launch.py                      interactive text menu (what Visual Studio's Start/F5 runs)
    python launch.py info                 engine provenance, config check, package versions
    python launch.py make-record ...      write a segmentation-interface record for a reviewed mask
    python launch.py preview ...          ROI / mask overlay PNG on the reference frame
    python launch.py morphology ...       morphology (IMAGE or VIDEO reference frame)
    python launch.py legacy-rhythm ...    legacy BPM / IBI / IBI-CV (VIDEO; Farneback on CPU)
    python launch.py pixel-first ...      pixel-first motion descriptors (VIDEO; not a BPM)
    python launch.py integrated ...       all of the above, from validated inputs only
    python launch.py validate ...         compare against stored results of the original repository
    python launch.py regional-rhythm ...  per-group (multi-rhythm) BPMs from a legacy_rhythm folder [added]
    python launch.py regional-gui         inspect groups / merge them by hand [added]

All paths come from the command line or the menu; nothing is hard-coded. Run `python launch.py <command> -h`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import adapters  # noqa: E402  (bootstraps the verbatim engine copy before any cardio_organoid import)

ANALYSES = ("morphology", "legacy-rhythm", "pixel-first", "integrated")


def _select_record(args):
    records = adapters.load_records(args.record)
    if len(records) == 1 and args.well is None:
        return records[0]
    if args.well is None or args.day is None:
        raise SystemExit(f"{args.record} holds {len(records)} records: pass --well and --day (and --modality if needed)")
    return adapters.find_record(records, args.well, args.day, args.modality)


def _mask_root(args) -> Path:
    return Path(args.mask_root) if args.mask_root else Path(args.record).resolve().parent


def run_analysis(kind: str, args) -> dict:
    rec = _select_record(args)
    meta = {"plate": args.plate, "organoid_id": args.organoid_id}
    src, root, out = Path(args.input), _mask_root(args), Path(args.output_dir)
    if kind == "morphology":
        row = adapters.analyze_morphology(rec, root, src, meta)
    elif kind == "legacy-rhythm":
        row = adapters.analyze_legacy_rhythm(src, rec, root, out, meta, quiet=not args.verbose)
    elif kind == "pixel-first":
        row = adapters.analyze_pixel_first(src, rec, root, out, meta, render_figures=not args.no_figures)
    else:
        row = adapters.analyze_integrated(src, rec, root, out, meta, run_legacy=not args.skip_legacy,
                                          run_pixel_first=not args.skip_pixel_first, render_figures=not args.no_figures,
                                          run_regional=not args.skip_regional)
    name = f"{kind.replace('-', '_')}_{adapters.observation_id(rec)}"
    saved = adapters.save_results([row], out, name, {"analysis": kind, "record_file": str(args.record),
                                                      "mask_root": str(root), "input": str(src)})
    status_keys = [k for k in row if k.endswith("_status") or k in ("bpm", "median_ibi_s", "ibi_cv", "pf_mag_rate_per_min",
                                                                    "n_rhythm_groups", "regional_bpms")]
    print(json.dumps({k: row[k] for k in status_keys}, indent=2, default=str))
    print(json.dumps(saved, indent=2))
    return row


def cmd_info(_args) -> None:
    print(json.dumps(adapters.provenance(), indent=2, default=str))


def cmd_make_record(args) -> None:
    mask = Path(args.mask)
    if mask.suffix.lower() == ".png":
        mask = adapters.mask_png_to_npz(mask, mask.with_suffix(".npz"))
    rec = adapters.make_record(batch=args.batch, condition=args.condition, well=args.well, day=args.day,
                               modality=args.modality, mask_path=str(mask.resolve()),
                               segmentation_qc_status=args.status, mask_provenance=args.provenance,
                               qc_reason=args.reason, selected_candidate=args.candidate)
    print(adapters.write_records([rec], Path(args.out)))


def cmd_preview(args) -> None:
    rec = _select_record(args)
    mask = adapters.require_mask(rec, _mask_root(args), "preview")
    print(json.dumps(adapters.render_mask_preview(Path(args.input), rec.modality, mask, Path(args.out_png)), indent=2))


def cmd_regional_rhythm(args) -> None:
    from adapters.regional_rhythm import analyze_regional_rhythm, load_merge_file, overlay_inputs, parse_merges
    merges = parse_merges(args.merge) + (load_merge_file(args.merge_file) if args.merge_file else [])
    frame = mask = None
    if args.input and args.record:
        frame, mask = overlay_inputs(Path(args.input), _select_record(args), _mask_root(args))
    row = analyze_regional_rhythm(Path(args.from_run), Path(args.output_dir) if args.output_dir else None,
                                  merges or None, frame_bgr=frame, mask=mask, render_figures=not args.no_figures)
    print(json.dumps(row, indent=2, default=str))


def cmd_regional_gui(args) -> None:
    from regional_gui.app import main as gui_main
    gui_main(args.from_run, args.input, args.record, args.mask_root)


def cmd_validate(args) -> None:
    sys.argv = ["run_validation.py", "--original-repo", str(args.original_repo)] + (["--skip-rederive"] if args.skip_rederive else [])
    sys.path.insert(0, str(PROJECT_ROOT / "validation"))
    import run_validation
    run_validation.main()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command")
    sub.add_parser("info", help="engine provenance and environment").set_defaults(func=cmd_info)

    mr = sub.add_parser("make-record", help="write a segmentation-interface record (reviewer states status/provenance)")
    for k in ("batch", "condition", "well", "modality", "mask", "status", "provenance", "reason", "out"):
        mr.add_argument(f"--{k}", required=True)
    mr.add_argument("--day", type=int, required=True)
    mr.add_argument("--candidate", type=int, required=True, help="SAM2 candidate index 0/1/2 the mask came from")
    mr.set_defaults(func=cmd_make_record)

    def common(p, need_out=True):
        p.add_argument("--input", required=True, help="video (.avi) or image (.tif/.tiff)")
        p.add_argument("--record", required=True, help="segmentation_interface_records.json (one or many records)")
        p.add_argument("--well")
        p.add_argument("--day", type=int)
        p.add_argument("--modality", choices=["IMAGE", "VIDEO"])
        p.add_argument("--mask-root", help="base folder for relative mask_path (default: the record file's folder)")
        if need_out:
            p.add_argument("--output-dir", required=True)
            p.add_argument("--plate")
            p.add_argument("--organoid-id")

    pv = sub.add_parser("preview", help="ROI / mask overlay PNG")
    common(pv, need_out=False)
    pv.add_argument("--out-png", required=True)
    pv.set_defaults(func=cmd_preview)

    for kind in ANALYSES:
        p = sub.add_parser(kind)
        common(p)
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--no-figures", action="store_true", help="pixel-first: skip PNG/MP4 rendering (numbers unchanged)")
        p.add_argument("--skip-legacy", action="store_true")
        p.add_argument("--skip-pixel-first", action="store_true")
        p.add_argument("--skip-regional", action="store_true", help="integrated: skip the regional (multi-rhythm) BPM step")
        p.set_defaults(func=lambda a, k=kind: run_analysis(k, a))

    rr = sub.add_parser("regional-rhythm", help="per-group BPMs from an existing legacy_rhythm folder (no optical flow)")
    rr.add_argument("--from-run", required=True, help="<output>/<observation>/legacy_rhythm (or its parent folder)")
    rr.add_argument("--output-dir", help="default: <observation>/regional_rhythm next to legacy_rhythm")
    rr.add_argument("--merge", help='manual merge of AUTO group labels, e.g. "G1+G3,G2+U1"')
    rr.add_argument("--merge-file", help="regional_merge.json saved by the regional GUI")
    rr.add_argument("--input", help="video, only for drawing the group map on the real frame")
    rr.add_argument("--record", help="segmentation_interface_records.json, only for the map overlay")
    rr.add_argument("--well")
    rr.add_argument("--day", type=int)
    rr.add_argument("--modality", choices=["IMAGE", "VIDEO"])
    rr.add_argument("--mask-root")
    rr.add_argument("--no-figures", action="store_true")
    rr.set_defaults(func=cmd_regional_rhythm)

    rg = sub.add_parser("regional-gui", help="window to inspect regional groups and merge them by hand")
    rg.add_argument("--from-run", help="legacy_rhythm folder to open at start")
    rg.add_argument("--input", help="video (for the map overlay)")
    rg.add_argument("--record", help="segmentation_interface_records.json (for the map overlay)")
    rg.add_argument("--mask-root")
    rg.set_defaults(func=cmd_regional_gui)

    va = sub.add_parser("validate", help="compare with stored results of the original repository (read-only)")
    va.add_argument("--original-repo", required=True, type=Path)
    va.add_argument("--skip-rederive", action="store_true")
    va.set_defaults(func=cmd_validate)
    return ap


def interactive() -> None:
    print("Cardiac bright-field standalone analysis (no GUI)\n")
    menu = ["info", *ANALYSES, "preview", "quit"]
    for i, m in enumerate(menu, 1):
        print(f"  {i}. {m}")
    choice = menu[int(input("\nselect number: ").strip()) - 1]
    if choice == "quit":
        return
    if choice == "info":
        return cmd_info(None)
    ask = lambda q, default=None: (input(f"{q}{f' [{default}]' if default else ''}: ").strip().strip('"') or default)
    argv = [choice, "--input", ask("input video/image path"), "--record", ask("segmentation record JSON path")]
    for flag, q in (("--well", "well (blank if the record file holds one record)"), ("--day", "day number (blank if one record)"),
                    ("--mask-root", "mask root folder (blank = record file folder)")):
        v = ask(q)
        if v:
            argv += [flag, v]
    if choice == "preview":
        argv += ["--out-png", ask("output PNG path")]
    else:
        argv += ["--output-dir", ask("output folder")]
    args = build_parser().parse_args(argv)
    args.func(args)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return interactive()
    args = build_parser().parse_args(argv)
    if not getattr(args, "func", None):
        build_parser().print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
