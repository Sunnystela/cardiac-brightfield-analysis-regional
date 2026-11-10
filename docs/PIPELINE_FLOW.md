# PIPELINE_FLOW — 입력에서 최종 출력까지 호출 순서

## 0. GUI가 호출할 공개 API (`adapters/__init__.py`)

```
영상/이미지 선택 ─► 세그멘테이션 기록 선택 ─► ROI/Mask 확인 ─► 분석 ─► 결과 저장
                    load_records()            render_mask_preview()   analyze_*()   save_results()
                    find_record()             require_mask()
                    make_record()  (검토자가 상태를 명시)
```

| 함수 | 입력 | 출력 |
|---|---|---|
| `load_records(json)` / `find_record(recs, well, day[, modality])` | `segmentation_interface_records.json` (엔진 형식 v1.0) | `SegmentationRecord` |
| `make_record(...)` / `write_records(...)` | 검토자가 정한 status, provenance, reason, candidate (기본값 없음) | 기록 JSON. 엔진의 `validate_record`로 검사 |
| `mask_png_to_npz(png, npz)` | 이진 PNG | 엔진 mask 형식 (`npz['mask']`, bool) |
| `require_mask(rec, mask_root, purpose)` | 기록 | mask. **ACCEPT + 사람 provenance + 파일 존재**가 아니면 `InterfaceFailClosed` |
| `render_mask_preview(src, modality, mask, png)` | 원본, mask | 기준 프레임 위 mask 외곽선(녹색)과 functional ROI(노랑) PNG |
| `analyze_morphology(rec, mask_root, src)` | IMAGE/VIDEO | morphology 1행 |
| `analyze_legacy_rhythm(video, rec, mask_root, out)` | VIDEO | legacy BPM/IBI 1행과 motion cache |
| `reaggregate_legacy_rhythm(video, rec, motion_dir, round36_json)` | 저장된 motion cache | legacy 1행 (영상 디코딩 없음) |
| `analyze_pixel_first(video, rec, mask_root, out, render_figures=True)` | VIDEO | pixel-first 1행과 산출물 |
| `analyze_integrated(src, rec, mask_root, out)` | IMAGE/VIDEO | 위 세 분석을 합친 1행 |
| `save_results(rows, out, name)` | 행 목록 | `<name>.csv`, `<name>.json`(provenance 포함), `<name>_provenance.json` |
| `provenance()` / `verify_engine()` | — | 버전, config sha256, 엔진 무결성 |

모든 analyze 함수는 **QC 사유로는 예외를 던지지 않는다.** 대신 상태 열(`*_status`)과 사유 열(`*_reason`)이 있는
행을 돌려준다. 실패한 경우 출력 폴더에 traceback JSON도 남긴다.

## 1. 공통 관문 (모든 분석)

```
record ─► validate_record (엔진)            invalid → FAILED_CLOSED + 사유
       ─► status == ACCEPT ?                 아니면 → MISSING_<status> / NOT_RUN + 사유 (측정 안 함)
       ─► provenance ∈ {human_confirmed, human_corrected}   (automatic ACCEPT 금지)
       ─► mask 파일 존재 (require_downstream_mask)
       ─► mask.shape == 기준 프레임 shape (VIDEO: 중간 프레임 frame_count//2, IMAGE: 이미지)
```

## 2. Morphology

```
analyze_morphology
 └ require_mask ─► morph_metrics (engine_scripts/p36_conv_first_pass_screening.py:102)
                    └ compute_morphology_features(mask).pixel (src/.../morphology/features.py:201)
                        area · perimeter · equivalent_diameter · circularity* · solidity* · ellipse · radial_cv
                        (* src/.../segmentation/sam2_segmenter.py:270,288)
 └ IMAGE: calibration_for_image(<image>_metadata.xml) ─► parse_pixel_calibration_um ─► um_per_pixel (X만 사용)
   VIDEO: NOT_CALIBRATED_no_metadata_source
 └ area_um2 = area_px·um², equivalent_diameter_um = eqd_px·um
```

## 3. 기존 BPM / IBI (VIDEO)

```
analyze_legacy_rhythm
 └ 관문 + 중간 프레임 shape 확인
 └ analyze_video_functional (engine_scripts/longitudinal_modality_aware_dry_run.py:128), fail_closed_mode
     ├ run_motion_pipeline (src/.../pipeline/analyze.py:169)
     │    fps = CAP_PROP_FPS (영상별 실제 값) · mask 주입 (SAM2 미실행)
     │    ROI = mask + 400 px · patch 격자 cell = round(0.12·eqd)
     │    쌍 (f, f+3)마다: Farneback → whole-ROI median drift 제거 → centroid 기준 radial 성분 → patch nanmean
     │    patch 주기성 surrogate 검정 (max_lag 100 샘플 → 영상이 최소 ~101 쌍 ≈ 303 프레임 이상이어야 함)
     │    ⇒ motion_pipeline/<obs>/patch_waveforms_radial.npz, motion_pipeline_qc.json
     ├ round-36 process_video ⇒ round36/<obs>_round36_raw_per_patch.json (patch 상태 A/B/C/D, 신뢰도)
     └ post_motion_functional (:168)   ← reaggregate_legacy_rhythm은 여기만 다시 실행
          C 상태가 아닌 patch → 극성별 이벤트 → patch 자체 주기 후보
          → 조화(harmonic) 보정 spanning tree → 기본 주기 P (신뢰도 가중 중앙값)
          → cycle-aware consensus (참여율 ≥ 0.8) ⇒ median_ibi_s, ibi_cv, consensus_beat_count
          → evidence_bundle ⇒ candidate_bpm (evidence 1)
          → classify_confidence_state ⇒ ESTIMABLE / LOW_CONFIDENCE / NOT_ESTIMABLE
 └ functional_record_from_result (원본 함수 그대로) → bpm 관문 (ESTIMABLE/LOW_CONFIDENCE일 때만)
 ⇒ <out>/<obs>/legacy_rhythm/{result.json, functional_result_full.json, motion_pipeline/, round36/}
```

## 4. Pixel-first (VIDEO, 탐색적 기술 지표)

```
analyze_pixel_first
 └ 관문 + 중간 프레임 shape 확인
 └ <out>/<obs>/pixel_first/segmentation_interface/segmentation_interface_records.json (절대 mask 경로 기록)
 └ rebind_paths(pixel_first_motion_batch, REPO_ROOT=<out>/<obs>/pixel_first, MANIFEST=…/segmentation_interface)
     ├ compute_primitives (:146)  Farneback(ROI) · drift 제거 · validity 지도 · int16 flow chunk · 배경 block
     │    ⇒ observation/{flow_chunks/, validity.npz, series.npz, primitives.json}
     └ derive (:314)  strict-inside / outer-support / background · 활성 block 군집 · 주기 · timing
          ⇒ observation/{summary.json, derived_timeseries.npz, derived_maps.npz, (figures: PNG, MP4)}
 └ row_from_summary  (build_master pf_* 블록과 동일) ⇒ pf_* 열
```
`pf_mag_rate_per_min`은 어떤 BPM 열에도 들어가지 않는다.

## 5. 통합 분석

```
analyze_integrated
 └ 관문 (1절) — 통과 못 하면 integrated_status = NOT_RUN_INPUT_NOT_VALIDATED / NOT_RUN_INPUT_PROBLEM, 측정 없음
 └ analyze_morphology
 └ VIDEO: analyze_legacy_rhythm  → first-pass 열 (bpm, median_ibi_s, ibi_cv, functional_qc_status …)
                                 + pixel-first master 규약 열 (legacy_bpm_reported, legacy_bpm/ibi_cv/median_ibi_s: ESTIMABLE만)
          analyze_pixel_first   → pf_* 열
   IMAGE: functional_execution_status = NOT_MEASURED_IMAGE, pixel_first_status = NOT_APPLICABLE_IMAGE
```
채널마다 자기 상태 열을 유지한다. 한 채널이 실패해도 다른 채널의 값과 실패 사유가 함께 남는다.

## 6. 저장

`save_results` → `<name>.csv`(엑셀용), `<name>.json`(provenance와 행), `<name>_provenance.json`.
provenance 항목: standalone 버전, 엔진 manifest sha256, 원본 git HEAD/dirty, **엔진 파일 무결성 재검사 결과**,
config sha256과 검증 run config 일치 여부, Python/플랫폼, 패키지 버전, 시각, morphology PILOT disclaimer.
식별 열(`observation_id, batch, condition, plate, well, organoid_id, day, modality, source_file, mask_path,
segmentation_qc_status, qc_reason, mask_provenance`)은 모든 행에 들어간다.

## 7. CLI 대응 (`launch.py`)

| 명령 | 호출 |
|---|---|
| `info` | `provenance()` |
| `make-record` | `mask_png_to_npz`(PNG일 때) → `make_record` → `write_records` |
| `preview` | `require_mask` → `render_mask_preview` |
| `morphology` / `legacy-rhythm` / `pixel-first` / `integrated` | 해당 `analyze_*` → `save_results` |
| `validate` | `validation/run_validation.py` |
| (인자 없음) | 대화형 메뉴 → 위 명령 |
