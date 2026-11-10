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
| `tests/test_regional_rhythm.py` | 새로 만듦 | 합성 캐시 테스트와 위상 고정 단위 테스트, 7개 |
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
2. **묶기 1단계 (그룹)**:
   - `build_robust_patch_timing_graph`로 패치 쌍의 거리를 계산해요. 거리 = 1 − 평균(박동 이벤트 일치율, 짧은 lag 상관, 배수 고려 주기 유사도).
   - `cluster_patches_by_robust_timing_similarity_hierarchical`로 묶어요. average linkage를 쓰고, relative-gap이 가장 큰 곳에서 잘라요.
   - **신뢰도 가중치는 쓰지 않아요.** 모든 패치에 가중치 1.0을 넣어서, 거리 = 1 − 유사도가 돼요. 이유는 아래 "개선 이력"에 있어요.
3. **묶기 2단계 (남은 패치 붙이기)**: `MIN_GROUP_PATCHES=3` 미만 클러스터에 남은 패치를 처리해요.
   - 가장 가까운 그룹을 찾아요.
   - 그 그룹 구성원끼리의 평균 거리 중 최댓값 안에 들어오면 그 그룹에 붙여요(`attached`로 표시).
   - 붙지 않은 패치는 미할당(`U*`)이에요.
4. **그룹별 지표**: 공식 BPM 규칙을 그룹 범위로만 좁혀서 적용해요.
   - `bpm`: `population_candidate_bpm`(IBI+, IBI−, 공식 신뢰도 가중치) 결과 중 **낮은 값**. 공식 `bpm`과 같은 규칙이에요. 그룹 안 공식 가중치가 모두 0 이하이면 순위 가중치로 대신 계산하고 `weight_fallback_rank=True`로 표시해요.
   - `fundamental_bpm`: `resolve_harmonic_scales`와 `weighted_median`
   - `median_ibi_s`, `ibi_cv`, `consensus_beat_count`: `cycle_aware_consensus_for_group`(그룹 자신의 주기 P, 참여율 0.8 이상)
   - `estimator_agreement`: 그룹 안 IBI, autocorr, spectral BPM의 일치도
   - 그 밖에 `n_spatial_components`, bbox, `mean_confidence`
   - 정합성 확인: reliable 패치 전체에 이 규칙을 적용하면 공식 `bpm`과 같은 값이 나와요(테스트와 실제 데이터에서 55.8745로 확인).
   - **그룹 박동 열**(`beat_train_s`, `beat_train_bpm`, `beat_train_ibi_cv`): 그룹 평균 파형에 엔진의 이벤트 검출기를 돌려서 얻은 positive 박동 시점이에요. 패치 하나는 놓치는 박동도 평균 파형에서는 잡혀요.
5. **기준 리듬과 위상 고정**:
   - **기준 리듬**(`reference`): 자동 그룹(패치 3개 이상) 중 박동 열이 가장 규칙적인(IBI CV가 가장 낮은) 그룹의 박동 열이에요. 관측마다 한 번만 정하고, 수동으로 합쳐도 바뀌지 않아요.
   - 각 그룹의 박동이 기준 리듬의 한 주기 안에서 어디쯤 오는지(위상)를 계산해서 아래 값을 내요.
     - `phase_lock.R`: 위상의 원형 평균 길이. 1이면 늘 같은 시차, 0에 가까우면 무관
     - `R_crit_p01`: Rayleigh 검정의 유의 기준. √(−ln 0.01 / n)
     - `delay_s`: 기준 대비 시차
     - `follow_fraction`: 기준 박동 중 따라간 비율
   - `relation` 판정:
     - `reference`: 기준 그룹을 포함한 그룹
     - `locked`: R > R_crit 이고 R ≥ 0.5. 같은 리듬이고, 늦게 뛰거나 박동을 놓쳐도 돼요
     - `not_locked`: 독립 리듬이거나 타이밍을 쓸 수 없는 경우
     - `insufficient_beats`: 박동이 5개 미만
   - 박동을 몇 개 놓치든 판정이 흔들리지 않아요. BPM 비교로는 일부만 놓친 경우(예: 3박 중 1박)를 같은 리듬으로 판단하지 못해요.
6. **그룹 쌍 비교 (합치기 제안, 자동으로 합치지는 않음)**:
   - 두 그룹 중 더 규칙적인 쪽을 기준으로 위상 고정을 계산해서, 고정되면 `merge_suggested`로 표시해요.
   - 주기 관계(`harmonic_aware_period_similarity`)와 lag를 맞춘 뒤 모양 상관(`region_pair_lag_diagnostic`, ±반 주기)은 참고용으로만 표시해요.
7. **수동 합치기**:
   - 항상 **자동 그룹 라벨**(`groups_auto`) 기준으로 기록해요. 예: `G1+G3`
   - 합친 그룹은 4번 지표를 다시 계산해요.
   - 라벨은 크기순으로 다시 매겨요. 패치 3개 이상은 `G*`, 미만은 `U*`예요.

**선언된 파라미터 (검증된 값이 아니라 진단용으로 정한 값)**:
`MIN_GROUP_PATCHES=3`, `MAX_LAG_S=1.0`, `COVERAGE_WINDOW_S=5.0`(진단용 필드에만 쓰고 거리에는 쓰지 않음), `ATTACH_LEFTOVERS=True`, `LOCK_MIN_R=0.5`, `MIN_BEATS_FOR_PHASE=5`, `FOLLOW_TOLERANCE_FRACTION=0.25`(기준 IBI 중앙값 대비). coincidence tolerance는 엔진 config(`episode_tolerance_period_fraction`)를 따라요.

## 출력
`<obs>/regional_rhythm/` 폴더에 써요(CLI에서 `--output-dir`로 바꿀 수 있어요).
- `regional_rhythm.json`: 아래 내용을 모두 담은 결과예요.
  - 자동 그룹과 최종 그룹, 그룹 쌍 비교
  - 1단계 cut 정보, DBSCAN 교차 확인(`rand_index_vs_final_auto`)
  - 패치별 값: state, 신뢰도, 순위 가중치, 고유 BPM, 후보들, `core`/`attached`/`not_reliable` 구분, 최종 그룹
  - 거리행렬(무가중. 1단계와 2단계에서 같은 행렬을 써요)
  - 기준 리듬(`reference`): 자동 그룹 라벨, 패치, 박동 시점, BPM, CV
- `regional_groups.csv`: 그룹당 1행
- `regional_merge.json`: 합친 경우에만 생겨요. `merges`, 메모, 시각
- 그림:
  - `regional_rhythm_map.png`:
    - 영상 프레임 위 그룹 지도. 라벨 = 박동 열 BPM과 REF/locked/INDEPENDENT. 빗금 = 미할당, ○ = 2단계에서 붙은 패치, 점선 = reliable 아님
    - 패치별 고유 BPM 지도
    - 그룹 표(박동 열 BPM·CV, 공식 규칙 BPM, 기준 대비 관계·R·시차·따라간 비율)
  - `regional_rhythm_traces.png`: 그룹 평균 파형, 그룹 박동(▼), 기준 리듬 박동(회색 점선), 그룹 순으로 정렬한 패치별 박동 이벤트 raster
  - `regional_rhythm_clustering.png`: 그룹 순으로 정렬한 거리행렬, 1단계 merge 높이와 cut 위치, 2단계 붙이기 결과, 그룹 쌍의 위상 고정 표
  - 프레임과 마스크를 주지 않거나, 다시 만든 패치 격자가 캐시와 다르면 추상 row/col 격자로 그리고, 제목에 그 이유를 적어요.

**결과 행에 추가되는 컬럼 (integrated, regional-rhythm)**:
- `regional_status`:
  - `SINGLE_GROUP`: 그룹이 하나
  - `MULTIPLE_GROUPS_SAME_RHYTHM`: 여러 그룹이 모두 기준 리듬에 위상 고정
  - `INDEPENDENT_RHYTHM_PRESENT`: 기준 리듬에 고정되지 않은 그룹이 있음
  - `MULTIPLE_GROUPS_UNRESOLVED`: 박동이 부족하거나 기준이 없어서 판정 불가
  - `NO_GROUP_WITH_MIN_PATCHES`, `NO_RELIABLE_PATCHES`, `FAILED`, `SKIPPED_BY_USER`, `NOT_RUN_NO_LEGACY_CACHE`
- `n_rhythm_groups`, `n_independent_groups`
- `regional_reference_group`, `regional_reference_beat_train_bpm`, `regional_reference_beat_train_cv`
- `regional_bpms`(공식 규칙, 큰 그룹 순), `regional_beat_train_bpms`, `regional_relations`
- `regional_group_n_patches`, `regional_group_fractions`, `regional_n_unassigned`, `regional_merge_applied`, `regional_dir`

## 사용법
```
python -B launch.py regional-rhythm --from-run <out>/<obs>/legacy_rhythm ^
       [--input video.avi --record segmentation_interface_records.json]   (프레임 위 지도용)
       [--merge "G1+G3"] [--merge-file regional_merge.json] [--output-dir DIR] [--no-figures]
python -B launch.py regional-gui [--from-run ...] [--input ... --record ...]
python -B launch.py integrated ... [--skip-regional]
```
GUI 사용 순서:
1. 폴더를 열어요.
2. 그룹 목록에서 각 그룹의 기준 대비 관계를 확인해요. 빨간 글씨는 독립 리듬이에요.
3. 그룹 쌍 목록에서 노란 줄(합치기 제안 = 위상 고정)을 더블클릭하면 두 그룹이 선택돼요.
4. [Merge selected]로 합쳐요. [Undo]/[Reset]으로 되돌릴 수 있어요.
5. [Save]로 저장해요. 저장할 때 합친 이유 메모를 남겨요.

## 개선 이력
### 1차 (2026-11-10, 최초 구현)
- 묶을 때 신뢰도 **순위** 가중치를 쓰고, 2단계에서 남은 패치를 붙였어요.
- 합치기 제안은 주기 유사도와 모양 상관으로 판단했어요.

### 2차 (2026-11-10): 묶기 가중치 제거, 위상 고정 추가
**문제**: P38_D16_slit_c1에서 패치별 고유 BPM 지도는 영역별로 달라 보였는데, 결과는 그룹 하나(`SINGLE_RHYTHM`)였어요. 사용자 지적으로 확인했어요.

**원인**: 엔진의 가중 거리 `1 − 유사도 × √(w_a·w_b)`가 영역 안과 영역 사이의 차이를 눌렀어요.

| | 가중치 없음 | 가중치 곱함 |
|---|---|---|
| 영역 안 거리 | 0.12–0.19 | 0.39–0.57 |
| 영역 사이 거리 | 0.35–0.43 | 0.59–0.68 |

- 같은 cut 위치(41번 중 28번째 합치기)에서 가중치 없이 묶으면 17/5/4/4/3으로 나뉘어요.
- 가중치를 곱하면 29개 한 덩어리가 돼요.
- 신뢰도(`final_provisional_confidence`)는 움직임을 관측할 수 있는지를 재는 값이지, 박동 타이밍 품질을 재는 값이 아니에요. 가장 규칙적인 영역(CV 0.03)의 신뢰도가 0.30으로, 위상 고정이 가장 약한 영역(0.48)보다 낮았어요.

**변경**:
1. 묶을 때 가중치를 빼서 모든 패치에 1.0을 줘요. 신뢰도는 공식 규칙 BPM을 계산할 때만 써요.
2. 그룹 박동 열을 추가했어요.
3. 기준 리듬을 정하고, 위상 고정(R, 시차, 따라간 비율)을 계산해요.
4. 합치기 제안 기준을 위상 고정으로 바꿨어요.
5. 상태 값을 새로 정의했어요.
6. 그림과 GUI 표시를 바꿨어요.

**같이 고친 버그**:
- GUI 작업 스레드가 Tk 변수(`var_mask_root`)를 읽어서, 영상과 record를 함께 줄 때 불러오기가 실패했어요(`main thread is not in main loop`). 메인 스레드에서 미리 읽도록 고쳤어요.
- 처음에는 기준을 "최종 그룹 중 가장 규칙적인 그룹"으로 정했어요. 그래서 기준 그룹을 다른 그룹과 합치면 기준이 다른 작은 그룹으로 바뀌고, 판정이 뒤집혔어요. 기준을 자동 그룹에서 한 번만 정하는 리듬으로 고정했어요.

## 검증 (2026-11-10, 2차 개선 후)
- `pytest tests`: 33개 통과, 4개 건너뜀(느린 테스트·원본 저장소 필요). `test_engine_integrity.py`도 통과해요.
- 위상 고정 단위 테스트:
  - 0.2 s 늦고 35% 박동을 놓쳐도 → locked. 시차 0.2 s, 따라간 비율이 실제 비율과 일치
  - 43 BPM 대 60 BPM → not_locked
  - 2:1 → locked, 따라간 비율 약 50%
- 합성 3영역 캐시(60 BPM / 36 BPM / 60 BPM이지만 0.5 s 어긋남):
  - 그룹 3개, 상태 `INDEPENDENT_RHYTHM_PRESENT`
  - 같은 속도의 어긋난 쌍만 합치기 제안(R > 0.9, 시차 0.5 s)
  - 합치면 다시 계산돼요.
- `CARDIAC_RUN_SLOW=1` integrated 합성 영상(1 Hz 원판 하나):
  - 그룹 2개(35·13패치)로 나뉘지만, 둘 다 60 BPM이고 서로 위상 고정 → `MULTIPLE_GROUPS_SAME_RHYTHM`
  - 엔진 cut 규칙이 항상 어딘가를 자르기 때문이고, 위상 고정 판정이 이걸 같은 리듬으로 판정해요.
- 실제 데이터(조합본 출력 `20261110T083136Z_f8fa9f75cf`, `P38_D16_slit_c1.avi`, 읽기만 함), 패치 격자가 캐시와 정확히 같아서 실제 프레임 위에 지도를 그렸어요:

  | 그룹 | 위치 | 패치 | 박동 열 BPM | CV | 관계 | R (기준값) | 시차 | 따라간 비율 |
  |---|---|---|---|---|---|---|---|---|
  | G2 | 왼쪽 아래 가장자리 | 5 | 61.5 | 0.03 | **기준** | – | – | – |
  | G1 | 왼쪽 위 호와 아래쪽 | 17 | 55.9 | 0.25 | locked | 0.83 (0.46) | −0.04 s | 80% |
  | G3 | 위쪽 | 4 | 47.3 | 0.26 | locked (**경계선**) | 0.50 (0.45) | 0.43 s | 67% |
  | G4 | 오른쪽 | 4 | 43.9 | 0.16 | locked | 0.55 (0.49) | 0.08 s | 57% |
  | G5 | 오른쪽 아래 | 3 | 51.2 | 0.60 | locked | 0.79 (0.55) | 0.18 s | 50% |

  - 미할당 9개(1패치씩)가 있어요.
  - 상태는 `MULTIPLE_GROUPS_SAME_RHYTHM`이에요.
  - 공식 `bpm` 55.9는 G1의 놓친 박동 때문에, 가장 규칙적인 리듬(61.5)보다 낮게 나와요.
  - GUI에서 G1+G2를 합치면 합친 그룹이 기준을 포함하게 되고, 다른 그룹의 판정은 그대로예요. 다만 합친 그룹의 평균 파형 박동 열은 55.9 BPM이에요. 패치가 많은 G1이 평균을 좌우하기 때문이에요. 이 리듬의 BPM은 `regional_reference_beat_train_bpm`(61.5)으로 보는 게 맞아요.

## 알려진 한계
- 묶는 기준(거리)은 엔진이 정의한 타이밍 거리예요. 공간 인접성은 쓰지 않아서, 떨어진 두 곳이 같이 뛰면 한 그룹이 돼요. 이건 `n_spatial_components`로 확인할 수 있어요.
- 1단계 cut 규칙(merge 순서의 뒤쪽 절반에서 relative-gap이 가장 큰 곳)은 엔진 그대로예요. 리듬이 하나뿐인 영상에서도 항상 어딘가를 잘라요. 그래서 그룹 수 자체보다 **기준 대비 관계(`relation`)**를 봐야 해요.
- 위상 고정 판정은 기준 리듬의 박동 검출이 정확하다는 전제에 기대요. 기준은 가장 규칙적인 그룹이지만, 그 그룹 박동이 맞는지는 영상으로 확인하는 게 좋아요.
- 시점 해상도는 프레임 간격(이 영상은 약 0.1 s)으로 제한돼요.
- 임계값(`LOCK_MIN_R` 등)은 검증 데이터로 정한 값이 아니에요. 수동 판단을 돕기 위한 값이에요. R이 기준값 근처인 그룹(예: G3)은 경계선으로 보고 따로 확인해야 해요.
- 조합본 GUI(`cardiac-brightfield-analysis`)에는 연결하지 않았어요. 그 GUI는 원본 main을 해시로 잠긴 복사본(`reference/`)으로 불러서 쓰기 때문이에요.
