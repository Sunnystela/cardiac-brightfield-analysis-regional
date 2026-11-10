# VISUAL_STUDIO_START — Visual Studio에서 열고 실행하기

> 이 문서와 `Cardiac_Brightfield_Standalone.pyproj`는 Linux/WSL에서 준비했고, **Windows Visual Studio에서는 아직
> 실행해 보지 않았다.** 처음 열 때 아래 순서대로 확인해 주세요.

## 0. 준비
- Visual Studio 2022에 **Python development** workload를 설치한다(Visual Studio Installer → Modify).
- Python 3.11 64-bit를 설치한다(python.org 설치본 또는 VS Installer의 Python 3.11).
- 프로젝트 폴더를 Windows 쪽으로 복사한다. 예: `D:\Analysis\Cardiac_Brightfield_Standalone`
  (WSL 경로 `\\wsl$\…`에서 바로 열면 venv와 성능 문제가 생기기 쉽다).
  필요한 것: `adapters/ src/ engine_scripts/ configs/ tests/ validation/ tools/ docs/ launch.py pyproject.toml
  requirements.txt README.md Cardiac_Brightfield_Standalone.pyproj`. `tests/_work/`와
  `validation/results/pixel_first_rederive/`는 복사하지 않아도 된다.

## 1. 방법 A — Python 프로젝트로 열기 (권장)
1. **File → Open → Project/Solution** → `Cardiac_Brightfield_Standalone.pyproj`를 연다.
2. Solution Explorer → **Python Environments** → 우클릭 **Add Environment…** → *Virtual environment*를 고른다.
   - Location `.venv`, Base interpreter Python 3.11 (64-bit)
   - "Install packages from requirements.txt"를 체크한다.
   - (`.pyproj`가 `.venv`를 기본 인터프리터로 지정해 두었으므로 이름을 `.venv`로 해야 한다.)
3. `launch.py`가 시작 파일로 지정되어 있다(굵게 표시). **F5**(디버그) 또는 **Ctrl+F5**로 실행하면 대화형 메뉴가 뜬다:
   ```
     1. info  2. morphology  3. legacy-rhythm  4. pixel-first  5. integrated  6. preview  7. quit
   ```
4. 명령행 인자로 바로 실행하려면 프로젝트 우클릭 → **Properties → Debug → Script Arguments**에 넣는다. 예:
   ```
   integrated --input "D:\data\a1.avi" --record "D:\data\records.json" --well a1 --day 20 --mask-root "D:\data" --output-dir "D:\out"
   ```
5. 환경변수 `PYTHONUTF8=1`은 `.pyproj`에 이미 들어 있다(Properties → Debug → Environment). 한글 경로나 비 ASCII
   문자가 들어간 JSON을 쓸 때 인코딩 오류를 막는다.

## 2. 방법 B — 폴더로 열기
1. **File → Open → Folder** → 프로젝트 폴더를 연다.
2. 하단 상태 표시줄 또는 Python Environments 창에서 `.venv`를 만들거나 선택한다(방법 A의 2번과 같음).
3. Solution Explorer에서 `launch.py` 우클릭 → **Set as Startup Item** → F5.
4. 인자가 필요하면 `launch.py` 우클릭 → **Add Debug Configuration**으로 생성되는 `launch.vs.json`에 `"args": [...]`를 추가한다.

## 3. 처음 실행 확인 순서
1. `info`를 실행한다. `engine_verification.engine_unmodified: true`와 `config_matches_validated_runs: true`가 나와야 한다.
   `packages`에서 실제 설치 버전을 확인한다.
2. Test Explorer(**Test → Test Explorer**)에서 pytest 테스트를 실행한다. 기본 26개가 통과하고, 느린 테스트와 원본 대조
   테스트는 skip된다. 느린 테스트는 환경변수 `CARDIAC_RUN_SLOW=1`, 원본 대조는 `CARDIAC_ORIGINAL_REPO=<원본 저장소 경로>`로 켠다.
3. 원본 결과에 접근할 수 있으면 `validate --original-repo "<원본 저장소 경로>"`를 실행한다. 출력의
   `all_match: true`를 확인한다.

## 4. 입력 준비 (분석 전 필수)
- 분석에는 **검토된 세그멘테이션 기록**이 필요하다(`segmentation_interface_records.json`).
  - 원본 manifest v2를 그대로 쓸 수 있다: `--record <원본>\results\_scratch_runs\p36_conv_segmentation_manifest_v2_human_reviewed_20260928T155748Z\segmentation_interface_records.json --mask-root <원본 저장소>`
  - 새 mask는 `make-record`로 기록을 만든다. **status, provenance, reason, candidate를 검토자가 직접 지정하며 기본값이 없다.**
    `automatic` provenance로는 ACCEPT 기록을 만들 수 없다(엔진 규칙).
- `preview`로 mask 외곽선과 functional ROI를 먼저 확인한다.
- legacy BPM은 영상 길이가 **약 303 프레임(frame_step 3 기준 101 쌍) 이상**이어야 한다. 이보다 짧으면 FAILED로
  기록된다.

## 5. 문제 해결
| 증상 | 원인과 조치 |
|---|---|
| `EngineResolutionError` | 다른 위치의 `cardio_organoid`(예: 원본 저장소 editable 설치)가 먼저 import되었다. 이 프로젝트 전용 `.venv`를 쓴다 |
| `engine_unmodified: false` | 엔진 파일이 바뀌었다. 바뀐 파일 목록이 provenance에 나온다. 원본에서 `python tools/sync_engine.py <원본>`으로 다시 복사한다 |
| `ModuleNotFoundError: torch` | torch가 필요하다(import만 함). `pip install torch==2.14.0` |
| 영상이 열리지 않음 | `opencv-python-headless` Windows 빌드의 코덱 지원을 확인한다(AVI/MJPG는 기본 지원) |
