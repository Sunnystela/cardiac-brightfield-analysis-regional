# 변경 기록: 영역별(다중) BPM 기능

## 출처
- 원본: `0. cardiac/cardiac-brightfield-analysis-main/cardiac-brightfield-analysis-main`
- 복제 시각: 2026-11-10T09:40:15Z (UTC). `__pycache__`는 빼고 그대로 복사했어요.
- 원본 폴더와 조합본(`0. cardiac/cardiac-brightfield-analysis`)은 수정하지 않았어요. 조합본의 실행 결과는 검증할 때 **읽기만** 했어요.

## 왜 바꿨나
- 공식 `bpm`은 오가노이드 전체가 하나의 리듬으로 뛴다고 가정해요. 계산 방식은 이래요.
  1. 모든 reliable 패치에서 positive/negative IBI의 신뢰도 가중 중앙값을 구해요.
  2. 둘 중 **낮은 BPM**을 써요.
- 그래서 여러 곳이 따로 뛰면 그 차이가 하나의 값으로 뭉개져요.
- 주기 그룹(`detect_period_families`)은 판정 등급에만 쓰이고, 그림으로 볼 수도 없었어요.
- 이번에 추가한 기능:
  - 주기와 박동 타이밍이 함께 비슷한 패치를 묶어요.
  - 그룹마다 BPM을 따로 내요.
  - 어떻게 묶였는지 그림으로 보여줘요.
  - 사람이 보고 그룹을 직접 합칠 수 있어요.

## 바꾸지 않은 것
- 해시로 잠긴 엔진 80개 파일(`configs/ENGINE_MANIFEST.json`: `src/cardio_organoid/**`, `engine_scripts/*.py`, `configs/default.yaml`)은 **한 글자도 바꾸지 않았어요**. `tests/test_engine_integrity.py`가 통과해요.
- 공식 출력 컬럼(`bpm`, `median_ibi_s`, `ibi_cv`, `functional_qc_status`, `legacy_*`, `pf_*`)의 값과 정의도 그대로예요. 새 값은 모두 `regional_*` 컬럼과 `regional_rhythm/` 폴더에만 써요.

## 파일 변경 목록
| 파일 | 구분 | 내용 |
|---|---|---|
| `adapters/regional_rhythm.py` | 새로 만듦 | 계산: 묶기, 그룹별 지표, 그룹 쌍 비교, 수동 합치기, 출력 |
| `adapters/regional_rhythm_figures.py` | 새로 만듦 | 그림 3종 |
| `regional_gui/__init__.py`, `regional_gui/app.py` | 새로 만듦 | Tk 창: 그룹 확인, 합치기, 되돌리기, 저장 |
| `tests/test_regional_rhythm.py` | 새로 만듦 | 합성 캐시 테스트 6개 |
| `adapters/__init__.py` | 수정 | `RegionalRhythm`, `analyze_regional_rhythm` export 추가 |
| `adapters/integrated.py` | 수정 | legacy 분석 다음에 regional 단계를 실행하고 `regional_*` 컬럼 추가. 인자 `run_regional`. 실패하면 `regional_status=FAILED`와 이유를 기록하고, 공식 컬럼에는 영향이 없어요 |
| `launch.py` | 수정 | 명령 `regional-rhythm`, `regional-gui`와 integrated용 `--skip-regional` 추가. 출력 요약에 `n_rhythm_groups`, `regional_bpms` 추가 |
| `CHANGES_REGIONAL_RHYTHM.md` | 새로 만듦 | 이 문서 |

## 계산 방법 (기존 엔진 함수 조합)
입력은 이미 만들어진 `<obs>/legacy_rhythm` 폴더예요. motion 캐시와 round36 JSON을 읽어요. 영상을 다시 디코딩하거나 optical flow를 다시 계산하지 않아요.

1. **패치별 증거**: `post_motion_functional`과 같은 순서로 계산해요.
   - `per_patch_events_and_features`로 패치별 이벤트를 뽑아요.
   - `patch_own_period_estimate`로 패치 고유 주기를 구해요. IBI+, IBI−, spectral, autocorr 후보의 중앙값이에요.
   - reliable 패치는 state가 C가 아닌 패치예요.
2. **묶기 1단계 (핵심 그룹)**:
   - `build_robust_patch_timing_graph`로 패치 쌍의 거리를 계산해요. 거리 = 1 − 평균(박동 이벤트 일치율, 짧은 lag 상관, 배수 고려 주기 유사도).
   - `cluster_patches_by_robust_timing_similarity_hierarchical`로 묶어요. average linkage를 쓰고, relative-gap이 가장 큰 곳에서 잘라요.
   - 가중치는 신뢰도 **순위**를 (0,1]로 바꾼 값이에요.
3. **묶기 2단계 (남은 패치 붙이기)**: `MIN_GROUP_PATCHES=3` 미만 클러스터에 남은 패치를 처리해요.
   - **가중치 없는** 거리로 가장 가까운 핵심 그룹을 찾아요.
   - 그 그룹 구성원끼리의 평균 거리 중 최댓값 안에 들어오면 그 그룹에 붙여요(`attached`로 표시).
   - 붙지 않은 패치는 미할당(`U*`)이에요.
   - 2단계가 필요한 이유: 엔진의 가중 거리는 `1 − 유사도 × √(w_a·w_b)`라서, 신뢰도가 낮은 패치는 타이밍이 맞아도 모든 패치와 멀어져요. 실제 데이터에서 reliable 42개 중 13개가 이 때문에 1개짜리 클러스터가 됐어요.
4. **그룹별 지표**: 공식 BPM 규칙을 그룹 범위로만 좁혀서 적용해요.
   - `bpm`: `population_candidate_bpm`(IBI+, IBI−, 공식 신뢰도 가중치) 결과 중 **낮은 값**. 공식 `bpm`과 같은 규칙이에요. 그룹 안 공식 가중치가 모두 0 이하이면 순위 가중치로 대신 계산하고 `weight_fallback_rank=True`로 표시해요.
   - `fundamental_bpm`: `resolve_harmonic_scales`와 `weighted_median`
   - `median_ibi_s`, `ibi_cv`, `consensus_beat_count`: `cycle_aware_consensus_for_group`(그룹 자신의 주기 P, 참여율 0.8 이상)
   - `estimator_agreement`: 그룹 안 IBI, autocorr, spectral BPM의 일치도
   - 그 밖에 `n_spatial_components`, bbox, `mean_confidence`
   - 정합성 확인: reliable 패치 전체에 이 규칙을 적용하면 공식 `bpm`과 같은 값이 나와요(테스트와 실제 데이터에서 55.8745로 확인).
5. **그룹 쌍 비교 (합치기 제안, 자동으로 합치지는 않음)**:
   - 주기 관계: `harmonic_aware_period_similarity`
   - 그룹 평균 파형: `region_mean_waveform`
   - lag와 모양 상관: `region_pair_lag_diagnostic`, 탐색 범위 ±반 주기
   - 주기 유사도 ≥ 0.90, 배수 ×1, lag를 맞춘 뒤 모양 상관 ≥ 0.80이면 `merge_suggested`로 표시해요.
6. **수동 합치기**:
   - 항상 **자동 그룹 라벨**(`groups_auto`) 기준으로 기록해요. 예: `G1+G3`
   - 합친 그룹은 4번 지표를 다시 계산해요.
   - 라벨은 크기순으로 다시 매겨요. 패치 3개 이상은 `G*`, 미만은 `U*`예요.

**선언된 파라미터 (검증된 값이 아니라 진단용으로 정한 값)**:
`MIN_GROUP_PATCHES=3`, `MAX_LAG_S=1.0`, `COVERAGE_WINDOW_S=5.0`(진단용 필드에만 쓰고 거리에는 쓰지 않음), `SUGGEST_MIN_PERIOD_SIMILARITY=0.90`, `SUGGEST_MIN_SHAPE_CORRELATION=0.80`, `ATTACH_LEFTOVERS=True`. coincidence tolerance는 엔진 config(`episode_tolerance_period_fraction`)를 따라요.

## 출력
`<obs>/regional_rhythm/` 폴더에 써요(CLI에서 `--output-dir`로 바꿀 수 있어요).
- `regional_rhythm.json`: 아래 내용을 모두 담은 결과예요.
  - 자동 그룹과 최종 그룹, 그룹 쌍 비교
  - 1단계 cut 정보, DBSCAN 교차 확인(`rand_index_vs_final_auto`)
  - 패치별 값: state, 신뢰도, 순위 가중치, 고유 BPM, 후보들, `core`/`attached`/`not_reliable` 구분, 최종 그룹
  - 가중 거리행렬(1단계)과 무가중 거리행렬(2단계)
- `regional_groups.csv`: 그룹당 1행
- `regional_merge.json`: 합친 경우에만 생겨요. `merges`, 메모, 시각
- 그림:
  - `regional_rhythm_map.png`: 영상 프레임 위 그룹 지도(빗금 = 미할당, ○ = 2단계에서 붙은 패치, 점선 = reliable 아님), 패치별 고유 BPM 지도, 그룹 표
  - `regional_rhythm_traces.png`: 그룹 평균 파형과 그룹 순으로 정렬한 패치별 박동 이벤트 raster
  - `regional_rhythm_clustering.png`: 그룹 순으로 정렬한 거리행렬, 1단계 merge 높이와 cut 위치, 2단계에서 붙은 패치와 기준선, 그룹 쌍 표
  - 프레임과 마스크를 주지 않거나, 다시 만든 패치 격자가 캐시와 다르면 추상 row/col 격자로 그리고, 제목에 그 이유를 적어요.

**결과 행에 추가되는 컬럼 (integrated, regional-rhythm)**:
`regional_status`(`SINGLE_RHYTHM` / `MULTIPLE_RHYTHMS` / `NO_GROUP_WITH_MIN_PATCHES` / `NO_RELIABLE_PATCHES` / `FAILED` / `SKIPPED_BY_USER` / `NOT_RUN_NO_LEGACY_CACHE`), `n_rhythm_groups`, `regional_bpms`(큰 그룹 순), `regional_group_n_patches`, `regional_group_fractions`, `regional_n_unassigned`, `regional_merge_applied`, `regional_dir`

## 사용법
```
python -B launch.py regional-rhythm --from-run <out>/<obs>/legacy_rhythm ^
       [--input video.avi --record segmentation_interface_records.json]   (프레임 위 지도용)
       [--merge "G1+G3"] [--merge-file regional_merge.json] [--output-dir DIR] [--no-figures]
python -B launch.py regional-gui [--from-run ...] [--input ... --record ...]
python -B launch.py integrated ... [--skip-regional]
```
GUI 사용 순서: 폴더를 열어요. 그룹 쌍 목록에서 노란 줄(합치기 제안)을 더블클릭하면 두 그룹이 선택돼요. [Merge selected]로 합치고, [Undo]/[Reset]으로 되돌리고, [Save]로 저장해요. 저장할 때 합친 이유 메모를 남겨요.

## 검증 (2026-11-10)
- `pytest tests`: 32개 통과, 4개 건너뜀(느린 테스트·원본 저장소 필요).
- `CARDIAC_RUN_SLOW=1`로 합성 영상 테스트 3개 통과. integrated 경로에서 regional 단계가 실행되는 것까지 확인했어요.
  - 1 Hz 원판 하나 → 48패치 그룹 1개, 60 BPM.
- 합성 3영역 캐시(60 BPM, 36 BPM, 60 BPM이지만 0.5초 어긋남) → 그룹 3개.
  - 같은 속도이고 어긋난 A–C 쌍만 합치기 제안이 떠요(lag 0.50 s, 모양 상관 1.00).
  - 합치면 24패치 60 BPM으로 다시 계산돼요.
  - GUI를 스크립트로 조작해서 불러오기, 쌍 선택, 합치기, 되돌리기, 탭 표시를 확인했어요.
- 실제 데이터(조합본 출력 `20261110T083136Z_f8fa9f75cf`, `P38_D16_slit_c1.avi`, 읽기만 함):
  - 자동 그룹: G1 = reliable 42개 중 41개, 55.9 BPM(공식 `bpm`과 같음). 미할당 1개(고유 10.1 BPM).
  - 패치 격자를 다시 만든 결과가 캐시와 정확히 같아서, 지도가 실제 프레임 위에 그려졌어요.

## 알려진 한계
- 묶는 기준(거리)은 엔진이 정의한 타이밍 거리예요. 공간 인접성은 쓰지 않아서, 떨어진 두 곳이 같이 뛰면 한 그룹이 돼요. 이건 `n_spatial_components`로 확인할 수 있어요.
- 1단계 cut 규칙(merge 순서의 뒤쪽 절반에서 relative-gap이 가장 큰 곳)은 엔진 그대로예요. 리듬이 하나뿐인 영상에서도 항상 어딘가를 자르기 때문에, 2단계 붙이기에 의존해요.
- 위의 임계값들은 검증 데이터로 정한 값이 아니에요. 수동 판단을 돕기 위한 값이에요.
- 조합본 GUI(`cardiac-brightfield-analysis`)에는 연결하지 않았어요. 그 GUI는 원본 main을 해시로 잠긴 복사본(`reference/`)으로 불러서 쓰기 때문이에요.
