# Cardiac Bright-field Analysis — Regional Rhythm

심장 오가노이드 bright-field 영상에서 **영역별 박동수(BPM)** 를 따로 구하는 확장판이다.
기존 분석의 공식 `bpm`은 오가노이드 전체가 하나의 리듬으로 뛴다고 보고 값 하나만 낸다. 그래서 여러 곳이 서로 다른
속도로 뛰면 그 차이가 하나의 값에 묻힌다. 이 저장소는 그 위에 다음 기능을 더한다.

- 주기와 박동 타이밍이 비슷한 motion 패치를 그룹으로 묶고, 그룹마다 BPM, IBI, IBI CV를 계산한다.
- 그룹 지도, 그룹별 파형, 거리행렬 그림을 만든다.
- 같은 리듬으로 보이는 그룹 쌍에 합치기를 제안한다. 실제로 합치는 것은 사람이 Tk GUI에서 결정한다.

해시로 잠긴 엔진 파일과 공식 출력 컬럼은 바꾸지 않았다. 새 값은 `regional_*` 컬럼과 `regional_rhythm/` 폴더에만
쓴다. 계산 방법, 출력, 검증, 한계는 [CHANGES_REGIONAL_RHYTHM.md](CHANGES_REGIONAL_RHYTHM.md)에 있다.

```text
python launch.py regional-rhythm --from-run <out>/<obs>/legacy_rhythm [--input <영상> --record <records.json>] [--merge "G1+G3"]
python launch.py regional-gui    [--from-run ...]
python launch.py integrated ...  [--skip-regional]     # integrated는 legacy 다음에 regional 단계를 자동 실행
```

---

아래는 원본 standalone 엔진(`cardiac-brightfield-analysis-main`)의 설명이다.

Cardiac organoid bright-field 분석 엔진을 원본 연구 저장소(`cardiac-organoid-agent`)에서 분리한 독립 프로젝트다.
나중에 GUI가 호출할 공개 API(`adapters/`)와 명령행 진입점(`launch.py`)을 제공한다. 엔진 자체에는 GUI가 없다.
GUI는 regional 그룹 검토용 `regional_gui/`만 있다.

- 계산 엔진은 2026-09-28/29 P36/Conv first-pass와 Pixel-first run을 실행한 코드와 **바이트 단위로 같은 복사본**이다
  (`src/`, `engine_scripts/`, `configs/default.yaml`). 수정하지 않았고 수정하면 안 된다. 모든 출력의 provenance에
  엔진 무결성 검사 결과가 들어간다.
- 저장된 원본 결과와 대조한 결과는 3,867개 값 중 불일치 0건이다(`docs/VALIDATION_REPORT.md`).
- 배포 준비 상태는 `docs/RELEASE_READINESS.md`를 본다. **아직 배포용이 아니다.**

## 구성
| 경로 | 내용 |
|---|---|
| `src/cardio_organoid/` | 원본 분석 라이브러리 (그대로 복사) |
| `engine_scripts/` | 계산 코드가 들어 있는 원본 스크립트 (그대로 복사, 원본 `scripts/`) |
| `adapters/` | GUI/CLI용 공개 API (새로 작성). regional 계산은 `regional_rhythm.py`, 그림은 `regional_rhythm_figures.py` |
| `regional_gui/` | regional 그룹 확인, 합치기, 저장용 Tk 창 |
| `configs/default.yaml` | 검증 run과 같은 설정 (sha256 `aab9e85b…`) |
| `configs/ENGINE_MANIFEST.json` | 엔진 파일 목록, sha256, run snapshot 대조 |
| `launch.py` | 명령행과 대화형 메뉴 진입점 |
| `tests/` | 무결성, 관문, 수치, CLI, 합성 영상, 원본 재현 테스트 |
| `validation/` | 원본 결과 대조 스크립트와 결과 |
| `tools/` | 엔진 동기화(`sync_engine.py`), import 폐포 분석, 지표 사전 생성 |
| `docs/` | 보고서 7종 (아래) |

보고서: [SOURCE_CODE_MAP](docs/SOURCE_CODE_MAP.md) · [PIPELINE_FLOW](docs/PIPELINE_FLOW.md) ·
[METRIC_DICTIONARY.csv](docs/METRIC_DICTIONARY.csv) · [DEPENDENCIES](docs/DEPENDENCIES.md) ·
[VISUAL_STUDIO_START](docs/VISUAL_STUDIO_START.md) · [VALIDATION_REPORT](docs/VALIDATION_REPORT.md) ·
[RELEASE_READINESS](docs/RELEASE_READINESS.md)

## 설치
Python 3.11, 패키지 버전 고정(`requirements.txt`). 자세한 내용은 `docs/DEPENDENCIES.md`.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python launch.py info
```
Visual Studio: `Cardiac_Brightfield_Standalone.pyproj`를 연다 → `docs/VISUAL_STUDIO_START.md`.

## 실행
```text
python launch.py                       # 대화형 메뉴
python launch.py info                  # 엔진 무결성, config, 패키지 버전
python launch.py preview      --input <영상|이미지> --record <records.json> [--well a1 --day 20] [--mask-root <폴더>] --out-png <png>
python launch.py morphology   --input ... --record ... --output-dir <출력 폴더> [--plate P1] [--organoid-id ...]
python launch.py legacy-rhythm --input <영상> ... --output-dir ...
python launch.py pixel-first  --input <영상> ... --output-dir ... [--no-figures]
python launch.py integrated   --input ... --record ... --output-dir ... [--skip-legacy] [--skip-pixel-first] [--no-figures]
python launch.py make-record  --batch P36 --condition Conv --well a1 --day 20 --modality VIDEO --mask <mask.npz|png>
                              --status ACCEPT --provenance human_confirmed --reason "<검토 사유>" --candidate 1 --out <records.json>
python launch.py validate     --original-repo <원본 저장소>   [--skip-rederive]
```
코드에 고정된 개인 경로는 없다. 입력, 기록, 출력 경로는 모두 인자나 메뉴로 받는다.

## 입력
- **영상/이미지:** VIDEO `.avi`(FPS는 파일에 기록된 값을 사용), IMAGE `.tif/.tiff`(같은 폴더의 `<이름>_metadata.xml`이 있으면 µm 보정).
- **세그멘테이션 기록** (`segmentation_interface_records.json`, 엔진 형식 v1.0): `ACCEPT` + `human_confirmed`/`human_corrected`
  기록만 분석된다. 그 밖의 기록은 측정 없이 사유와 함께 행으로 남는다. mask는 기준 프레임(영상은 `frame_count//2`)과
  shape가 같아야 한다.
- legacy BPM은 영상이 약 303 프레임 이상이어야 한다.

## 출력 (`--output-dir`)
- `<분석>_<observation_id>.csv` / `.json` / `_provenance.json`: 결과 행과 provenance(엔진 무결성, config sha256,
  패키지 버전, 원본 git, 시각).
- `<observation_id>/legacy_rhythm/`: motion cache, round-36 patch 상태, 전체 결과 JSON.
- `<observation_id>/pixel_first/observation/`: flow chunk, validity, summary, 시계열, 그림/동영상(선택).
- 모든 행에 batch, condition, plate, well, organoid_id, day, modality, source_file, mask_path,
  segmentation_qc_status, qc_reason이 들어간다.

## 반드시 알아둘 해석 규칙
- `bpm`과 `median_ibi_s`/`ibi_cv`는 **다른 추정기**다(원본 정의). `bpm`은 ESTIMABLE/LOW_CONFIDENCE일 때만 채워진다.
- `pf_mag_rate_per_min`은 **BPM이 아니다.** BPM 대신 쓰지 않는다.
- Morphology는 PILOT 값이다(중간 프레임, cardiac-phase 정규화 없음).
- 지표 정의 전체: `docs/METRIC_DICTIONARY.csv`.

## 테스트
```text
python -m pytest tests -q                                   # 기본 26개 (~1 s)
CARDIAC_RUN_SLOW=1 python -m pytest tests -q                # + 합성 영상 3개 (~3 min, CPU)
CARDIAC_ORIGINAL_REPO=<원본 저장소> python -m pytest tests -q  # + 원본 결과 재현
```
