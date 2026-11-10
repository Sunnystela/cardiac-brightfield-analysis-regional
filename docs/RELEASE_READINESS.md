# RELEASE_READINESS — 구현됨 / 미완성 / 미검증 / 배포 보류

상태 2026-11-03, standalone 0.1.0. **배포 가능 상태가 아니다.** 계산 엔진을 분리했고, 저장된 결과와 대조하는
재현 검증까지 마친 상태다.

## 1. 구현됨, 검증됨
| 항목 | 근거 |
|---|---|
| 엔진 분리 (원본과 바이트 동일, 80개 파일) | `configs/ENGINE_MANIFEST.json`, `tests/test_engine_integrity.py` |
| 실행 버전 식별 (run snapshot 8개 대조, 차이 3개 설명) | `SOURCE_CODE_MAP.md` §1 |
| 엔진 격리 (원본 editable 설치가 몰래 쓰이는 것 차단) | `adapters/_engine.py`, 테스트 |
| mask 관문 (ACCEPT + 사람 provenance만 통과, 실패 사유 보존) | `tests/test_gating.py` |
| Morphology adapter | 저장값 84/84 완전 일치 |
| Legacy BPM/IBI 재집계 (저장 cache) | 저장값 112/112 완전 일치, 3개 판정 상태 모두 포함 |
| Pixel-first 열 매핑 | 83행 × 37열 완전 일치 |
| Pixel-first `derive` 재실행 | 2개 관측: summary, 배열, master 행 완전 일치 |
| 결과 저장과 provenance (엔진 무결성, config 일치, 패키지 버전) | `adapters/results_io.py`, `tests/test_cli.py` |
| CLI와 대화형 메뉴 (`launch.py`) | `tests/test_cli.py` (IMAGE 경로) |

## 2. 구현됨, 부분 검증 (합성 데이터에서만 동작 확인)
| 항목 | 상태 |
|---|---|
| 영상 → legacy BPM 전체 경로 (`analyze_legacy_rhythm`) | 합성 1 Hz 영상에서 BPM 60.0과 IBI 1.0 s가 나왔다. **실제 영상으로는 다시 실행하지 않았다**(MD 보호). 다음 단계: 실제 대표 영상 1~2개를 다시 실행해 저장된 `patch_waveforms_radial.npz`와 대조 |
| 영상 → pixel-first 전체 경로 (`analyze_pixel_first`) | 합성 영상에서 동작하고 재현성을 확인했다. 실제 영상 재계산은 하지 않았다. 다음 단계: a1 D20 primitives 재계산 후 저장 flow chunk와 대조 |
| 통합 분석 (`analyze_integrated`) | 합성 영상과 이미지에서 동작한다. 실제 데이터로는 실행하지 않았다 |
| ROI/Mask 미리보기 (`render_mask_preview`) | PNG 생성까지만 확인했다. 사람이 눈으로 확인한 것은 아니다 |

## 3. 알려진 한계와 관찰 (수정하지 않음, 판단 필요)
1. **`bpm`과 `median_ibi_s`/`ibi_cv`는 다른 추정기에서 나온다.** f3 D20처럼 크게 다를 수 있다(66.85 vs 35.4/min).
   원본 정의 그대로다. 보고서나 GUI에서는 이 둘을 같은 리듬의 두 표현으로 보여주면 안 된다.
2. 원본 first-pass master에서는 NOT_ESTIMABLE 행에도 `median_ibi_s`와 `ibi_cv`가 채워져 있고, 그대로 유지했다.
   통합 분석은 pixel-first master 규약 열(`legacy_*`, ESTIMABLE만)도 함께 내므로, GUI는 **`legacy_*` 열을 기본으로
   표시**하는 것을 권장한다.
3. 합성 정규 박동에서 `ibi_cv` = 0.287이 나왔다. 가장자리 cycle 때문이며 가운데 구간은 정확하다. IBI 변동성 지표로
   쓰기 전에 실제 데이터에서 가장자리 효과를 확인해야 한다.
4. `observable_fraction`과 `reliable_active_fraction`은 영상별 median split에서 나오므로 구성상 약 0.5다. 생물학적
   비율이 아니다(2026-09-29 감사).
5. 기능 판정 임계값(`DEFAULT_PROVISIONAL_THRESHOLDS`)은 configs가 아니라 모듈 상수다. CLAUDE.md §12와 맞지 않지만
   엔진 수정 금지 원칙에 따라 그대로 두었다.
6. legacy 경로에는 영상이 최소 약 303 프레임 필요하다(`periodicity_max_lag` 100 샘플, frame_step 3).
   GUI는 실행 전에 이를 알려야 한다.
7. Morphology는 `PILOT_DISCLAIMER` 대상이다. 중간 프레임이고 cardiac-phase 정규화를 하지 않았으므로, 조건 비교용
   최종값이 아니다. provenance에 disclaimer를 기록한다.
8. VIDEO morphology와 모든 pixel-first 값은 px 단위다(보정 정보 없음). IMAGE 보정은 X 값만 쓴다.
9. `pf_mag_rate_per_min`은 BPM이 아니다(합성 1 Hz에서 2배). 어떤 BPM 열에도 쓰지 않는다.

## 4. 미완성 (설계 결정 필요)
| 항목 | 이유 / 필요한 결정 |
|---|---|
| SAM2 세그멘테이션을 GUI에서 실행 | 이번 범위 밖이다(GPU, 모델 금지). `sam2`, checkpoint, 후보 선택과 사람 검토 흐름이 필요하다. 지금은 **기존 검토된 mask 기록만** 입력으로 받는다 |
| 손으로 그린 mask 등록 | 엔진 인터페이스 v1은 ACCEPT 기록에 SAM2 후보 번호(0/1/2)를 요구한다. 수작업 mask를 나타낼 방법이 없으므로, `make-record`는 `--candidate`를 반드시 받는다. 인터페이스 확장 여부는 **연구자 결정**이 필요하다 |
| 여러 관측 일괄 실행 | 단일 관측 API만 있다. 일괄 실행은 GUI나 batch 단계에서 관측별로 반복 호출하면 된다(실패는 행으로 남음) |
| data_curation QC 자격 연결 | `qc/data_curation.py`를 복사만 했다. first-pass 계산에는 쓰이지 않았으므로 연결하지 않았다. 통합 분석의 입력 자격에 curation 결과를 넣을지는 결정이 필요하다 |
| 엔진의 torch 의존 제거 | torch는 import만 하지만 배포 용량을 크게 늘린다. 없애려면 엔진 수정이 필요하다(`sam2_segmenter.py`). 승인이 필요하다 |
| 버전 관리 | 이 폴더는 아직 git 저장소가 아니다. 출력에는 엔진 manifest sha256을 기록한다. `git init`과 첫 commit 여부를 정해야 한다 |

## 5. 배포 보류
- **Windows 실행 미확인:** `.pyproj`, 문서, `PYTHONUTF8` 설정은 준비했지만 Windows에서 열거나 실행하지 않았다.
- **Windows 패키지 버전 미확인:** 고정 버전 wheel이 Windows에 있는지 확인하지 않았다.
- **EXE 빌드 안 함:** PyInstaller onedir 권장 설정만 문서화했다(`DEPENDENCIES.md` §4).
- **실제 영상 end-to-end 미검증:** §2.
- 위 항목이 해결되기 전에는 연구 결과 산출용으로 배포하지 않는다.

## 6. 다음 단계 제안 (MD 계산 종료 후)
1. 실제 영상 1~2개(a1 D20, f3 D20)로 legacy와 pixel-first 전체 경로를 다시 실행하고, 저장 cache와 대조한다.
2. Windows에서 `.venv`를 만들고 `launch.py info`, `pytest`, `validate`를 실행한다.
3. 수작업 mask 정책과 torch 의존 처리를 결정한다.
4. GUI 설계: 이 문서의 공개 API(`PIPELINE_FLOW.md` §0)만 호출하도록 한다.
