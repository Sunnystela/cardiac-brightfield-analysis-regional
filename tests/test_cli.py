"""launch.py end to end on a synthetic IMAGE (fast; no video, no optical flow)."""
import json

import numpy as np

import launch


def test_make_record_then_morphology_and_preview(work_dir):
    import cv2
    yy, xx = np.mgrid[:400, :500]
    mask = (yy - 200) ** 2 + (xx - 250) ** 2 <= 120 ** 2
    cv2.imwrite(str(work_dir / "img.tif"), np.where(mask, 180, 90).astype(np.uint8))
    cv2.imwrite(str(work_dir / "mask.png"), mask.astype(np.uint8) * 255)
    rec = work_dir / "record.json"
    launch.main(["make-record", "--batch", "SYN", "--condition", "Test", "--well", "a1", "--day", "3", "--modality", "IMAGE",
                 "--mask", str(work_dir / "mask.png"), "--status", "ACCEPT", "--provenance", "human_confirmed",
                 "--reason", "cli_test", "--candidate", "0", "--out", str(rec)])
    out = work_dir / "out"
    launch.main(["morphology", "--input", str(work_dir / "img.tif"), "--record", str(rec), "--output-dir", str(out),
                 "--plate", "P1"])
    res = json.loads((out / "morphology_SYN_Test_a1_D3_IMAGE.json").read_text())
    row = res["rows"][0]
    assert row["morphology_status"] == "MEASURED" and row["area_px"] == int(mask.sum())
    assert row["plate"] == "P1" and row["organoid_id"] == "SYN_Test_a1" and row["day"] == 3
    assert row["calibration_status"] == "NOT_CALIBRATED_no_metadata_sidecar"
    prov = res["provenance"]
    assert prov["engine_verification"]["engine_unmodified"] and prov["config_matches_validated_runs"]
    assert (out / "morphology_SYN_Test_a1_D3_IMAGE.csv").exists()
    launch.main(["preview", "--input", str(work_dir / "img.tif"), "--record", str(rec), "--out-png", str(work_dir / "p.png")])
    assert (work_dir / "p.png").stat().st_size > 0
