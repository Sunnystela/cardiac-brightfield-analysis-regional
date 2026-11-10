# VALIDATION_REPORT — 원본 결과와 새 프로젝트의 비교

실행 2026-11-03. 명령:
`python validation/run_validation.py --original-repo <원본 저장소>` (`python launch.py validate …`와 같음)

- 원본 값, 신규 값, 차이, 일치 여부(행 단위): `validation/results/VALIDATION_RESULTS.csv` (3,867행)
- 요약: `validation/results/validation_summary.json`, 실행 환경: `validation/results/validation_provenance.json`
- 일치 기준은 **완전 일치**(float `==`, 허용오차 0)이다. 차이 열은 참고용이다.

## 1. 결과 요약 — 불일치 0건

| 검사 | 대상 | 원본 비교 대상 | 비교 수 | 일치 |
|---|---|---|---|---|
| M morphology | a1 D1 (IMAGE, 보정 있음), a1 D20, f3 D20, b2 D21(다중 성분), e1 D17(구멍), a2 D3 (NO_ACCEPTABLE_CANDIDATE 관문 사례) | `morphology/morphology_results.json` | 84 | **84** |
| L legacy BPM/IBI | 7개 관측: a1 D20, f3 D20, RERUN·LOW_CONFIDENCE(a1 D14), RERUN·NOT_ESTIMABLE(a2 D19), REUSED·ESTIMABLE(a1 D13), REUSED·NOT_ESTIMABLE(c2 D21), REUSED·LOW_CONFIDENCE(a1 D18) | `functional/functional_results.json` | 112 | **112** |
| LC legacy 규약 | pixel-first master 130행 전부 | `MASTER_9WELL_130.csv`의 legacy_* 4열 | 520 | **520** |
| PM pixel-first 열 매핑 | summary.json이 있는 83행 전부 | `MASTER_9WELL_130.csv` pf_* 37열 | 3,071 | **3,071** |
| PD pixel-first derive 재실행 | a1 D20 (REUSED pilot primitives), h3 D20 (COMPUTED primitives) | 저장된 `summary.json`(관측당 leaf 103개), `derived_timeseries.npz`(배열 14개), `derived_maps.npz`(배열 6개), master 행(37열) | 80 | **80** |

M과 L의 비교 열:
- M: morphology_status, calibration_status, um_per_pixel, area_px, perimeter_px, equivalent_diameter_px, circularity,
  solidity, eccentricity, aspect_ratio, radial_cv, perimeter_over_sqrt_area, area_um2, equivalent_diameter_um
- L: functional_qc_status, candidate_bpm, bpm, fundamental_bpm, median_ibi_s, ibi_cv, consensus_beat_count,
  reliable_active_fraction, observable_fraction, n_total_patches, n_reliable_patches, n_period_families,
  period_family_bpms_descriptive, actual_fps, frame_step, triggered_failure_modes

### 대표 값 (원본 = 신규)
| 관측 | 지표 | 값 |
|---|---|---|
| a1 D1 IMAGE | area_px / circularity / solidity / um_per_pixel / area_um2 | 567379 / 0.845120 / 0.990925 / 0.22 / 27461.14 |
| a1 D20 VIDEO | area_px / circularity / solidity | 553818 / 0.532210 / 0.859197 |
| f3 D20 VIDEO | area_px / circularity / solidity | 579489 / 0.478681 / 0.819893 |
| a1 D20 | functional_qc / bpm / median_ibi_s / ibi_cv | ESTIMABLE / 54.6873 / 1.09715 / 0.08682 |
| f3 D20 | functional_qc / bpm / median_ibi_s / ibi_cv | ESTIMABLE / **66.8533** / **1.69525** (= 35.4/min) / 0.32844 |
| a1 D14 | functional_qc / bpm | LOW_CONFIDENCE / 46.2738 (보고됨) |
| a2 D19 | functional_qc / bpm / median_ibi_s | NOT_ESTIMABLE / None / 1.36861 (원본처럼 IBI는 채워짐) |
| a1 D20 | pf_mag_rate_per_min / pf_strict_speed_p95 / pf_measurable_fraction | 54.687 / 30.136 px/s / 0.2881 |
| h3 D20 | pf_mag_rate_per_min / pf_strict_speed_p95 / pf_measurable_fraction | 35.386 / 19.540 px/s / 0.2254 |

f3 D20의 `bpm` ≠ `60/median_ibi_s`는 **오류가 아니라 원래 정의**다. 두 값은 서로 다른 추정기에서 나온다
(`SOURCE_CODE_MAP.md` §3.2). 보정하지 않았다.

## 2. 검증하지 않은 것 (중요)
- **실제 영상에서 영상 → Farneback → patch 파형 단계(`run_motion_pipeline`, round-36, `compute_primitives`)를 다시
  돌리지 않았다.** MD 보호 조건(optical flow 재계산 금지)에 따른 것이다. legacy 검증은 저장된 motion cache에서
  `post_motion_functional`만 다시 실행했고, pixel-first 검증은 저장된 int16 flow에서 `derive`만 다시 실행했다.
  이 단계의 코드는 run snapshot과 바이트 단위로 같다는 것까지만 확인했다.
- SAM2 세그멘테이션은 검증하지 않았다(이번 공개 API에 포함하지 않음).
- 생리학적 정확성(어떤 BPM이 실제 박동과 맞는지)은 이번 범위가 아니다.
- Windows에서는 실행하지 않았다.

## 3. 합성 영상 end-to-end 시험 (생물학적 검증 아님)
합성 영상: 1080×1920, 330 프레임, 30 fps, 반지름 300 px 원판이 1 Hz로 ±6% 반경 방향 맥동하는 텍스처
(`tests/synthetic.py`). 전체 영상 경로를 adapter로 실행했다(`tests/test_synthetic_video.py`, CPU 단일 스레드).

| 항목 | 결과 |
|---|---|
| legacy | RUN, ESTIMABLE, **bpm 60.0, median_ibi_s 1.000 s**, fundamental_bpm 60.0, 신뢰 patch 48/70, actual_fps 30.0, frame_step 3 |
| legacy ibi_cv | **0.287**. 대표 시각 간격이 가운데 구간은 정확히 1.0 s이고, 처음 세 cycle(0.6, 0.7, 1.3 s)과 마지막(0.4 s)만 벗어났다. 완전히 규칙적인 입력에서도 기록 가장자리 cycle 때문에 IBI CV가 커질 수 있다는 관찰이다. 원인은 엔진의 cycle 배정(양/음 이벤트를 합친 대표 시각)으로 추정하며, 확인하거나 수정하지 않았다 |
| pixel-first | COMPUTED, 영역 1개, **pf_mag_period_s 0.5 s → 120/min** (박동 60/min의 2배). |u|가 수축과 이완 모두에서 커지기 때문이다. BPM 대체 금지 규칙의 근거 |
| pixel-first 재현성 | 같은 입력으로 두 번 실행한 summary.json이 완전히 같음 |
| 짧은 영상(180 프레임 = 59 쌍) | legacy가 FAILED로 기록되었다(`periodicity_max_lag` 100 > 59 쌍). failed.json이 남고 bpm=None이다. 조용히 버려지지 않음 |
| 640×640 프레임 | functional ROI(mask+400 px)가 프레임 전체를 덮어 배경 기준 cell이 없어 FAILED로 기록되었다(엔진 동작, 실제 1080×1920 영상과 무관) |

## 4. 테스트 결과 (`pytest tests`)
| 묶음 | 결과 |
|---|---|
| 기본(빠름): 엔진 무결성, 격리, mask 관문, BPM 관문, morphology 수치, pf 매핑, CLI | **26 passed** (1.2 s) |
| `CARDIAC_ORIGINAL_REPO` 원본 대조(경량) | **1 passed** (3.1 s) |
| `CARDIAC_RUN_SLOW=1` 합성 영상 3개 | **3 passed** (195 s) |

## 5. 안전 기록
- 원본 저장소: 읽기 전용. 검증 시작 시 standalone 안에 marker 파일을 만들고, 끝난 뒤 원본 `src scripts configs tests
  docs results checkpoints`에서 marker보다 새 파일을 찾았다. **0개.** `git status` dirty 경로 수도 153개로 전후 같다.
- 자원: CPU 전용(`CUDA_VISIBLE_DEVICES=""`), `nice 19`, `ionice idle`, 단일 스레드. 전체 검증은 wall 75 s,
  최대 RSS 1.4 GB. 합성 영상 테스트는 최대 RSS 1.4 GB.
- 이 sandbox에서는 MD 프로세스가 보이지 않았다(다른 환경에서 실행 중인 것으로 보임). 종료, 재시작, 접근은 하지 않았다.
