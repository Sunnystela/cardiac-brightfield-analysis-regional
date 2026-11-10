# SOURCE_CODE_MAP — 기능별 계산 코드, 실제 실행 증거, 버전

작성 2026-11-03. 원본 저장소 `cardiac-organoid-agent`는 읽기 전용으로만 사용했다(파일 수정·이동·삭제, git 쓰기 없음).
이 프로젝트의 계산 엔진은 원본 파일을 **바이트 단위로 그대로** 복사한 것이다. 줄 번호는 원본과 같다.

| 이 프로젝트 | 원본 저장소 |
|---|---|
| `src/cardio_organoid/**` | `src/cardio_organoid/**` |
| `engine_scripts/*.py` | `scripts/*.py` |
| `configs/default.yaml` | `configs/default.yaml` |

복사 목록, 파일별 sha256, run snapshot별 일치 여부는 `configs/ENGINE_MANIFEST.json`에 있다(`tools/sync_engine.py`로 재생성 가능).

---------------------------------------------------------------------------------------------------

## 1. 실제 실행 버전 식별

### 1.1 방법 — HEAD만으로 판단하지 않음
원본 HEAD는 `dca4b015511d…`이고 working tree에 dirty 경로가 153개 있다. 모든 run manifest에도 `git_dirty_code: true`가
기록되어 있으므로 **commit hash는 계산 코드의 근거로 쓰지 않았다.** 대신 아래 사슬로 확인했다.

1. 각 run의 `run_manifest.json`에 기록된 `code_snapshot_sha256`과 실제 `code_snapshot.tar.gz`의 sha256을 비교했다.
   8개 모두 일치하여 snapshot이 손상되지 않았다.
2. 각 tarball 안의 파일 sha256을 원본 현재 파일과 비교했다.
3. 원본 현재 파일의 sha256을 standalone 복사본과 비교했다. 80/80 동일하다.

### 1.2 Run manifest (모두 config sha256 `aab9e85b92ff…`, git `dca4b01` + dirty)

| 단계 | run manifest (원본 `results/_scratch_runs/…`) | 시각 (UTC) | code snapshot sha256[:12] |
|---|---|---|---|
| 세그멘테이션 manifest v2 (사람 검토) | `p36_conv_segmentation_manifest_v2_human_reviewed_20260928T155748Z/manifest` | 2026-09-28 15:57:51 | f08c9bb73985 |
| first-pass morphology | `p36_conv_first_pass_screening_20260928T155956Z/manifests/morphology` | 2026-09-28 15:59:56 | 35bec54fe262 |
| first-pass functional_plan | `…/manifests/functional_plan` | 2026-09-28 16:00:05 | 31bd52564697 |
| first-pass functional_run | `…/manifests/functional_run` | 2026-09-28 16:00:25 | 8f025b6eefa7 |
| first-pass functional_aggregate | `…/manifests/functional_aggregate` | 2026-09-28 18:11:30 | 66e138e295dc |
| pixel-first plan | `pixel_first_screening_P36_Conv_9well_20260929T062953Z/manifests/plan` | 2026-09-29 06:30:01 | 69e40beaa809 |
| pixel-first primitives | `…/manifests/primitives` | 2026-09-29 07:10:46 | ae209e501ec3 |
| pixel-first derive | `…/manifests/derive` | 2026-09-29 07:34:39 | cbdc2bc1af30 |
| **first-pass integrate** (`P36_CONV_MASTER_OBSERVATION_TABLE.csv`) | **없음** | (파일 mtime 2026-09-28 20:13:41) | — |
| **pixel-first aggregate** (`MASTER_9WELL_130.csv`) | **없음** | (파일 mtime 2026-09-29 09:38:13) | — |

### 1.3 결과
- **계산 파일은 모두 모든 snapshot과 현재 파일, standalone 복사본이 동일하다.** 대상은 `src/cardio_organoid/**`
  전체와, 아래 3개를 제외한 `engine_scripts/` 전부다. `configs/default.yaml`도 8개 run 모두와 동일하다.
- snapshot과 다른 파일은 3개뿐이다. 세 경우 모두 현재 버전을 복사했다.

| 파일 | 다른 snapshot | 차이 | 현재 버전을 쓴 근거 |
|---|---|---|---|
| `p36_conv_first_pass_outputs.py` | fp_functional_aggregate (e86cb523) | 그림(plotting) 부분만 | 현재 파일이 그 후 생성된 pf_plan/primitives/derive snapshot 3개와 동일하다. master 표 mtime(20:13:41)이 파일 mtime(20:13:40) 직후다. |
| `pixel_first_motion_aggregate.py` | pf_primitives, pf_derive (4132b7c2) | 현재 버전에 `pf_mag_period_autocorr` 열과 mkdir 추가 | `MASTER_9WELL_130.csv`에 이 열이 있다. 즉 master는 현재 버전이 썼다. |
| `pixel_first_motion_batch.py` | pf_plan (1e1ccf52) | 표시용 mask(`bgref & ~lowconf`)만 | 실제 계산 단계(primitives, derive) snapshot과 현재 파일이 동일하다. |

- **추정 표시:** manifest가 없는 두 집계 단계(integrate, pixel-first aggregate)를 실행한 코드 버전은 기록된 사실이 아니라
  mtime과 열 존재 여부로 **추정**한 것이다. 숫자 열은 두 버전 차이의 영향을 받지 않고, 이번 검증에서 master 값이
  모두 재현되었다(`VALIDATION_REPORT.md`).
- **Python 패키지 버전은 run manifest에 기록되어 있지 않다.** 원본 `.venv`의 설치 시각은 모두 실행 이전이다
  (2026-09-03, 09-04, 09-27). 그 이후 설치된 것이 없으므로 현재 `.venv`가 실행 당시 환경이라고 **추정**한다.

---------------------------------------------------------------------------------------------------

## 2. Notion 목록과 실제 저장소 대조

| Notion 목록 | 실제 위치 | 존재 | 실행 경로 사용 여부 | 이 프로젝트에서 |
|---|---|---|---|---|
| `pipeline/segmentation_interface.py` | `src/cardio_organoid/pipeline/` | ✔ | ✔ 모든 하위 분석의 mask 관문(`require_downstream_mask`) | 그대로 복사, adapter가 이 관문만 사용 |
| `morphology/features.py` | `src/cardio_organoid/morphology/` | ✔ | ✔ morphology | 그대로 복사 |
| `sam2_segmenter.py` | `src/cardio_organoid/segmentation/` | ✔ | △ `circularity`/`solidity` 함수와 모듈 수준 `import torch`만 사용. SAM2 추론은 이번 run들에서 실행되지 않았다(mask는 manifest v2에서 읽음. seg_v2 manifest의 `sam2_checkpoint`도 None) | 그대로 복사, SAM2 추론은 노출하지 않음 |
| `period_evidence_confidence.py` | `src/cardio_organoid/functional/` | ✔ | ✔ `evidence_bundle` → BPM 원천, `classify_confidence_state` → 판정 | 그대로 복사 |
| `fundamental_period_consensus.py` | `src/cardio_organoid/functional/` | ✔ | ✔ 기본 주기 P | 그대로 복사 |
| `cycle_aware_consensus.py` | `src/cardio_organoid/functional/` | ✔ | ✔ `median_ibi_s`, `ibi_cv`, `consensus_beat_count` | 그대로 복사 |
| `cycle_phase_assignment.py` | `src/cardio_organoid/functional/` | ✔ | ✔ `choose_anchor_time_s` | 그대로 복사 |
| `reliable_hotspot_candidates.py` | `src/cardio_organoid/motion/` | ✔ | ✔ round-36 상태 A/B/C/D(`classify_state`) | 그대로 복사 |
| `pixel_first.py` | `src/cardio_organoid/motion/` | ✔ | ✔ pixel-first 보조 함수 | 그대로 복사 |
| `scripts/pixel_first_motion_batch.py` | 〃 | ✔ | ✔ primitives와 derive | `engine_scripts/` |
| `scripts/pixel_first_motion_aggregate.py` | 〃 | ✔ | ✔ master 작성(manifest 없음) | `engine_scripts/` |
| `scripts/p36_conv_first_pass_screening.py` | 〃 | ✔ | ✔ morphology와 legacy 기능 분석 조율 | `engine_scripts/` |
| `scripts/p36_conv_first_pass_outputs.py` | 〃 | ✔ | ✔ master 작성(manifest 없음) | `engine_scripts/` |
| `qc/data_curation.py` | `src/cardio_organoid/qc/` | ✔ | ✘ 2026-10-12 작성(위 run들보다 나중). `data_curation_master_qc_reconciliation.py`만 사용하며, 파일 IO가 없는 QC 상태 정의 모듈이다 | 참고용으로 그대로 복사, **계산에 연결하지 않음** |
| `functional/bpm.py` | `src/cardio_organoid/functional/` | ✔ (**0 byte**) | ✘ 어떤 run도 import하지 않음 | **복사하지 않음** |

`docs/CURRENT_STATUS.md`(원본)의 "regional BPM/IBI/CV — `functional/bpm.py`" 설명은 낡은 기록이다. 실제 BPM 계산
경로는 3절에 정리했다. 이 프로젝트에는 `bpm.py`를 연결하지 않았다(`tests/test_engine_integrity.py`로 확인).

추가로 실행 경로에 필요하여 함께 복사한 모듈이 있다. 정적 import 폐포 기준으로 `cardio_organoid` 모듈 55개와
스크립트 24개다. 다수는 진단 스크립트 사이의 모듈 수준 import 때문에 끌려온다. 엔진을 고치지 않는다는 원칙 때문에
잘라내지 않았다. 전체 목록은 `configs/ENGINE_MANIFEST.json`에 있다.

---------------------------------------------------------------------------------------------------

## 3. 기능 → 계산 함수 (경로는 이 프로젝트 기준, 줄 번호는 원본과 동일)

### 3.1 Morphology — `adapters.analyze_morphology`
`morph_metrics` (`engine_scripts/p36_conv_first_pass_screening.py:102`)
→ `compute_morphology_features(mask).pixel` (`src/cardio_organoid/morphology/features.py:201`)

| 지표 | 함수 | 정의 요약 |
|---|---|---|
| area_px | `features.py:93 area` | `mask.sum()`: 픽셀 수. 모든 성분 포함, 구멍 제외 |
| perimeter_px | `features.py:98 perimeter` | 가장 큰 외곽 contour의 `cv2.arcLength` |
| equivalent_diameter_px | `features.py:112` | `2*sqrt(area_px/pi)` |
| circularity | `segmentation/sam2_segmenter.py:288` | `4π·contourArea(L)/arcLength(L)²`. **픽셀 면적이 아니라 contour 다각형 면적** |
| solidity | `sam2_segmenter.py:270` | `contourArea(L)/contourArea(convexHull(L))` |
| eccentricity, aspect_ratio | `features.py:117` | 가장 큰 contour에 `cv2.fitEllipse` |
| radial_cv | `features.py:143, 186` | 360개 광선의 가장 바깥 경계 반지름으로 계산한 `std/mean` |
| perimeter_over_sqrt_area | `p36_conv_first_pass_screening.py:108` | `perimeter_px/sqrt(area_px)` |
| um_per_pixel, area_um2, equivalent_diameter_um | `p39_conv_a3_longitudinal_morphology_pilot.py:153` → `io/longitudinal_metadata_xml.py:112` | IMAGE는 `<image>_metadata.xml`의 X 값(m→µm)만 사용. VIDEO는 보정 없음 |

Adapter의 차이는 한 가지다. 원본 `calibration_for_image_day`는 sidecar 경로를 원본 저장소 기준 상대경로로 저장하므로,
저장소 밖 파일에서는 예외가 난다. `adapters/morphology.py:calibration_for_image`는 같은 분기, 같은 상태 문자열,
같은 파서를 쓰고 경로만 받은 그대로 기록한다. 원본 함수와의 일치는
`tests/test_morphology_sanity.py::test_calibration_mirror_matches_engine_function`로 검증한다.

### 3.2 기존 BPM / IBI / IBI-CV — `adapters.analyze_legacy_rhythm`, `adapters.reaggregate_legacy_rhythm`
| 단계 | 함수 |
|---|---|
| 영상 → patch별 radial 파형 | `analyze_video_functional` (`engine_scripts/longitudinal_modality_aware_dry_run.py:128`) → `run_motion_pipeline` (`src/cardio_organoid/pipeline/analyze.py:169`): 파일에 기록된 FPS(`CAP_PROP_FPS`), frame_step 3, Farneback(`motion/raft_flow.py:104`, 파라미터 `:38`), `remove_global_drift`(`motion/radial_flow.py:33`), `radial_motion_from_centroid`(`:69`) |
| patch 관측 가능성 상태 | round-36 `process_video` (`engine_scripts/motion_observability_diagnostic.py:239`) → `classify_state` (`motion/reliable_hotspot_candidates.py:68`). 상태는 **영상별 median split**이다 |
| 이벤트, 기본 주기, 합의 | `post_motion_functional` (`longitudinal_modality_aware_dry_run.py:168`): `per_patch_events_and_features` (`patch_phase_consensus_diagnostic.py:132`), `patch_own_period_estimate` (`fundamental_period_consensus.py:85`), `resolve_harmonic_scales` (`:112`) + `weighted_median` (`:182`), `cycle_aware_consensus_for_group` (`cycle_aware_consensus.py:112`) |
| **bpm** | `evidence_bundle` (`functional/period_evidence_confidence.py:193`)의 evidence 1: 양/음 극성별 population median-IBI rate의 가중 중앙값. 가중치가 같으므로 **둘 중 낮은 값**이 된다 |
| **median_ibi_s, ibi_cv, consensus_beat_count** | `cycle_aware_consensus_metrics` (`cycle_aware_consensus.py:83`): 참여율 ≥0.8(config)인 cycle의 대표 시각 간격 |
| 판정 | `classify_confidence_state` (`period_evidence_confidence.py:285`). 임계값은 `DEFAULT_PROVISIONAL_THRESHOLDS` (`:80`)로, **configs가 아니라 모듈 상수**다 |
| 행 매핑 | `functional_record_from_result` (`p36_conv_first_pass_screening.py:194`, adapter가 그대로 호출). BPM 관문(`:284`): ESTIMABLE/LOW_CONFIDENCE일 때만 `bpm`을 채운다 |

**BPM과 IBI는 서로 다른 경로에서 계산된다.** `bpm`은 evidence 1이고, `median_ibi_s`와 `ibi_cv`는 cycle-aware consensus다.
그래서 `bpm = 60/median_ibi_s`는 항상 성립하지 않는다(원본 master에서 223행 중 192행만 일치). 예로 f3 D20은
bpm 66.85, median_ibi_s 1.695 s(35.4/min)이다. 원본 그대로 유지했다. 원본 master에서 `median_ibi_s`와 `ibi_cv`는
NOT_ESTIMABLE 행에도 채워져 있고, 이것도 유지했다. Pixel-first master 규약(`legacy_*`, ESTIMABLE만)은 통합 분석에서 별도 열로 함께 낸다.

### 3.3 Pixel-first — `adapters.analyze_pixel_first`
| 단계 | 함수 |
|---|---|
| primitives | `compute_primitives` (`engine_scripts/pixel_first_motion_batch.py:146`): Farneback, drift 제거, validity(saturated/low_texture/fb_inconsistent), int16 flow chunk, 배경 block 시계열 |
| derive | `derive` (`:314`): strict-inside / outer-support / background 요약, 활성 block 군집화(`AgglomerativeClustering`), 자기상관 주기(`motion/pixel_first.py:55`), peak(`:45`), timing(`:73`) |
| 그림, 동영상 | `figures` (`:521`). adapter 옵션 `render_figures=False`로 건너뛸 수 있다. summary.json을 쓴 다음에 호출되며 반환값이 없으므로 숫자에 영향이 없다 |
| 열 매핑 | `build_master` (`engine_scripts/pixel_first_motion_aggregate.py:72`)의 pf_* 블록을 `adapters/pixel_first.py:row_from_summary`가 그대로 옮겼다. 저장된 83행 전부와 일치한다 |
| 상수 | `engine_scripts/pixel_first_motion_diagnostic.py:57-67` (CROP_PAD 120, ADJ_BAND 20, BG_MIN_DIST 200, BLOCK 4, MIN_REGION_BLOCKS 25, LOW_TEXTURE_GRAD 2.0, FB_MAX 1.0, FLOW_Q 100) |

`pf_mag_rate_per_min`은 |u| 자기상관의 반복률이며 **BPM이 아니다.** 1 Hz 합성 박동에서 legacy BPM은 60.0이었고
pixel-first는 120/min이었다. |u|는 수축과 이완 두 번 모두 커지기 때문이다. 실제 a1 D20에서는 두 값이 같았다(54.69).
즉 관계는 데이터에 따라 달라지므로 BPM 대체값으로 쓰지 않는다.

---------------------------------------------------------------------------------------------------

## 4. Adapter가 엔진에 하는 일 (엔진 파일은 수정하지 않음)
1. `sys.path` 앞쪽에 `engine_scripts/`, `src/`를 넣는다. 원본 스크립트와 같은 순서다.
2. **격리 검사:** 엔진 모듈이 이 프로젝트 밖에서 import되면 중단한다. 원본 `.venv`에는 원본 `src/`를 가리키는
   editable 설치(`__editable__.cardio_organoid-0.1.0.pth`)가 있어서, 이 검사가 없으면 원본 코드가 몰래 쓰일 수 있다.
3. **경로 상수 재바인딩:** `pixel_first_motion_batch.REPO_ROOT`와 `MANIFEST`를 호출하는 동안만 사용자 출력 폴더의
   작업공간으로 바꾼다. Path 형 상수만 허용하며, 숫자 상수는 건드리지 않는다(`adapters/_engine.py:rebind_paths`).
4. **figures 생략 옵션:** `pixel_first_motion_batch.figures`를 호출 중에만 no-op으로 바꾼다.
5. **행 조립:** `stage_morphology`와 `stage_functional_aggregate`의 행 조립(상태 문자열, BPM 관문, µm 환산)과
   `build_master`의 pf_* 매핑을 adapter로 옮겼다. 원본 함수가 manifest 디렉터리 단위로만 동작하기 때문이다.
   모두 저장된 결과와 대조해 검증했다.
