# DEPENDENCIES — 실행에 필요한 패키지와 환경

## 1. 기준 환경 (검증 run을 만든 원본 `.venv`)

run manifest에는 패키지 버전이 기록되어 있지 않다. 아래 버전은 원본 `.venv`(uv로 생성)에 지금 설치된 것이다.
설치 시각은 모두 run(2026-09-28, 09-29) **이전**(2026-09-03, 09-04, 09-27)이고 그 이후 설치는 없으므로,
실행 당시 환경과 같다고 **추정**한다.

| 항목 | 버전 | 이 프로젝트에서의 쓰임 |
|---|---|---|
| CPython | 3.11.16 (Linux x86_64, WSL2) | 필수. 3.11 외 버전은 테스트하지 않음 |
| numpy | 2.4.6 | 필수 |
| opencv-python-headless | 5.0.0.93 | 필수 (영상 디코딩, Farneback, contour). `opencv-python`은 설치되어 있지 않음 |
| scipy | 1.17.1 | pixel-first `derive`(connected components) |
| scikit-learn | 1.9.1 | pixel-first `derive`(AgglomerativeClustering) |
| matplotlib (+ Pillow 12.3.0) | 3.11.1 | 엔진 모듈 import 시 필요, 그림 |
| PyYAML | 6.0.3 | `configs/default.yaml` |
| torch | 2.14.0 (원본은 CUDA 빌드 `+cu130`) | **import만 함.** `segmentation/sam2_segmenter.py`가 모듈 수준에서 `import torch`를 하고, legacy 경로가 이 모듈을 import한다. 이번 분석 경로에서 torch 연산이나 GPU는 쓰지 않으므로 **CPU 빌드로 충분** |
| pytest | 9.1.1 | 테스트만 |

이번 경로에 **필요 없는** 것: `sam2`(SAM-2 1.0, SAM2 추론용), SAM2 checkpoint(176 MB), `joblib`(selector 진단용),
`torchvision`, `hydra-core`, `iopath`. 정적 import 폐포에는 `joblib`, `sam2`가 보이지만 이번 분석 경로에서는
실행 중 import되지 않는다. 실행 시 실제 로드를 확인한 결과 numpy, cv2, matplotlib, PIL, torch, yaml이 로드되었고
scipy, sklearn은 pixel-first derive 안에서 지연 로드된다.

정확한 버전 고정(`pyproject.toml`, `requirements.txt`)을 권장한다. Farneback flow와 float 요약값은 OpenCV/NumPy
버전에 따라 미세하게 달라질 수 있다. 다른 버전으로 실행한 결과는 provenance의 `packages`에 기록되며,
`python launch.py validate`로 저장된 결과와 다시 대조해야 한다.

## 2. 이번 단계에서 한 것과 하지 않은 것
- 새 Python 환경을 **만들지 않았다.** 검증과 테스트는 원본 `.venv` 인터프리터를 **읽기 전용으로 사용**했다
  (`PYTHONDONTWRITEBYTECODE=1`, 패키지 설치 없음). MD 계산과 공유하는 환경이 아니다.
- 실행은 모두 `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, `ionice -c3`, `OMP_NUM_THREADS=1` 조건이었다.

## 3. 새 환경 구축 (나중에, MD 계산이 끝난 뒤)

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt      # torch: Windows PyPI wheel은 CPU 빌드
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python launch.py info
.\.venv\Scripts\python -m pytest tests -q
```

### Linux / WSL
```bash
python3.11 -m venv .venv
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # CPU 빌드, 용량 절감
.venv/bin/pip install -r requirements.txt
.venv/bin/python launch.py info
```

**미확인 사항:** 위 고정 버전의 Windows wheel이 모두 존재하는지는 이번에 확인하지 않았다(네트워크 사용과 설치를
하지 않음). 특히 `opencv-python-headless==5.0.0.93`과 `torch==2.14.0`의 Windows 배포를 먼저 확인해야 한다.

## 4. Windows 실행 파일 배포 설정 (문서화만, 이번에 빌드하지 않음)

권장: **PyInstaller `--onedir`** (onefile은 권장하지 않음)
- 엔진은 `Path(__file__).parent.parent`로 `src/`, `configs/`, `engine_scripts/`를 찾는다(원본 REPO_ROOT 규약).
  onefile 모드는 실행 시 임시 폴더로 풀리므로 이 구조를 깨기 쉽다. onedir 모드에서 세 폴더를 **같은 상대 구조로**
  data로 포함해야 한다.
- 엔진 스크립트는 `sys.path`로 동적 import되므로 PyInstaller가 자동으로 찾지 못한다.
  `--add-data "engine_scripts;engine_scripts" --add-data "src;src" --add-data "configs;configs"`와
  `--hidden-import`(scipy.sparse.csgraph, sklearn.cluster 등)가 필요하다.
- torch는 import만 하지만 번들에 포함되면 CPU 빌드도 수백 MB 이상이다. torch를 빼려면 엔진을 수정해야 하므로
  (`sam2_segmenter.py`의 모듈 수준 import) **이번 원칙상 하지 않았다.** 결정이 필요하다(`RELEASE_READINESS.md`).
- 콘솔 앱으로 빌드해야 한다(`--console`). 진행 메시지와 오류가 콘솔로 나온다.
- 빌드 후 `<dist>\launch.exe validate --original-repo …` 또는 `pytest`에 해당하는 검증을 반드시 다시 한다.

예시(실행하지 않음):
```powershell
pyinstaller launch.py --onedir --console --name CardiacBrightfield `
  --add-data "src;src" --add-data "engine_scripts;engine_scripts" --add-data "configs;configs" `
  --hidden-import scipy.sparse.csgraph --hidden-import sklearn.cluster
```
위 명령에서 `launch.py`의 `PROJECT_ROOT`가 번들 구조와 맞는지는 아직 확인하지 않았다.
