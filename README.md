# LTX GUI

**독립 개발한 Windows 데스크톱 영상 복원 작업 관리자**입니다. Jasna의 공개 GUI/작업 큐 구조를 비교·연구하여 새로 작성했습니다. 모자이크 탐지·복원 및 GPU 커널 자체를 구현한 독립 추론 엔진은 아닙니다. **정상 설치·활성화된 Jasna 실행 파일을 별도 프로세스로 제어**합니다.

> **중요:** 지원자 전용 Jasna LTX 모델과 활성화 로직은 본 저장소에 포함되지 않습니다. 무료 BasicVSR++는 별도 설치한 Jasna의 공개 기능 범위에서 사용할 수 있습니다. LTX 정품 라이선스는 원 저작자로부터 취득해야 합니다. `--ltx-trial`은 복원 영상을 생성하지 않습니다.

## 설치 (Windows 10/11)

1. Python 3.11 이상 설치 (권장 3.12).
2. 프로젝트 폴더에서 `py -3.12 -m venv .venv` → `.venv\Scripts\activate`.
3. `python -m pip install -e ".[monitoring]"`.
4. `python -m ltx_gui` 또는 `ltx-gui` 실행.
5. GUI의 **Jasna 실행 파일**에 공식 `jasna.exe`의 실제 경로를 지정합니다. Jasna 설치 및 정품 활성화는 Jasna 공식 문서대로 별도 진행합니다.

Windows EXE 생성: `powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1`. 생성물은 `dist/LTX_GUI/`에 놓입니다. GitHub Actions에서도 Windows 빌드 아티팩트를 생성하도록 준비했습니다. Jasna 실행 파일, GPU 드라이버 또는 모델은 번들에 포함되지 않습니다.

## 실제 제공 기능

- 다중 파일·폴더 드래그 앤 드롭, 중복 입력 제거, FIFO 작업 큐
- 파일 단위 복원 범위 편집 (`00:10-00:25,01:10-01:30`)
- LTX Standard/Good/High/Top, BasicVSR++, GPU 선택
- LTX seed, RTX 50 fast, large canvas, less overlap, big-mosaic auto split, trial
- 출력 파일 자동 생성 및 기존 파일 덮어쓰기 방지, 충돌 감지
- BasicVSR++ 클립/오버랩, 하드웨어 CQ, H.264/HEVC/AV1 설정
- CLI 실행/로그/상태/취소, Jasna 체크포인트 기반 재실행
- CPU/RAM 및 NVIDIA GPU/VRAM 현황 (3초 주기, 백그라운드)
- 설정 및 작업 목록 안전한 JSON 저장 (원자적 파일 교체)
- 소스 전용 테스트와 Windows PyInstaller 패키징

## 사용 방법

1. 파일을 목록으로 드롭하고 출력 폴더, 복원 모델, GPU 번호를 선택합니다.
2. 일부 장면만 복원한다면 영상 한 개를 선택해 범위를 입력하고 **범위 적용**을 클릭합니다.
3. **대기열 실행**으로 순서대로 처리합니다. 한 번에 하나의 GPU 프로세스만 실행해 VRAM 경합을 방지합니다.
4. **중지**는 Jasna에 소프트 인터럽트를 우선 전달합니다. 같은 입력·출력·설정으로 다시 실행하면 Jasna가 지원하는 LTX 체크포인트에서 이어질 수 있습니다. 실패한 작업의 기존 임시 데이터는 자동 삭제하지 않습니다.
5. 같은 이름의 출력 파일이 이미 있다면 실행을 거부합니다. 덮어쓰기를 원할 때는 사용자가 명시적으로 해당 파일을 이동/이름 변경해야 합니다.

## 요구사항 및 제한사항

- 이 프로그램만 설치해서 LTX 영상 복원이 가능한 것은 **아닙니다**. GPU 드라이버·Jasna·모델·라이선스는 외부 요구사항입니다.
- **Jasna 공식 CLI**의 `--input`, `--output`, `--restoration-model-name`, `--ltx-*`, `--segments` 등을 호출합니다. 릴리스에 따라 옵션이 변경될 수 있으므로 사용할 버전에서 `jasna --help`로 확인하세요.
- 진행률은 CLI 텍스트에서 식별 가능한 퍼센트만 표시합니다. 일부 버전/단계에서는 로그만 갱신되며 전체 작업 진행률이 아닙니다.
- 선택 구간(`--segments`) 처리 시 Jasna가 원본 코덱을 보존하므로 GUI의 코덱/CQ 값은 전달하지 않습니다.
- `--ltx-auto-split`과 수동 복원 범위는 동시에 사용할 수 없습니다.
- RTX 50 전용 빠른 모드를 지원하지 않는 GPU에서 사용하면 Jasna가 오류를 반환할 수 있습니다.
- 모델의 GPU 연산, 정밀도, TensorRT/CUDA 커널은 Jasna 실행 엔진이 관리합니다. GUI는 이를 임의로 변경하지 않습니다.
- DRM/라이선스 우회, 비공개 체크포인트 복호화 또는 배포 기능은 제공하지 않습니다.
- Windows GUI 실기기 시험과 실제 NVIDIA/AMD GPU 성능 측정은 별도로 필요합니다.

## 속도 및 검증

알고리즘 분석: [docs/ANALYSIS.md](docs/ANALYSIS.md) / [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

테스트: `python -m pip install -e ".[test]"` 후 `python -m pytest -q`. 테스트는 가상 CLI로 프로세스 성공/오류/취소 경로를 검증하며, 실제 모자이크 제거 품질이나 GPU 속도를 측정하지 않습니다.

## 출처·라이선스

- Jasna: https://github.com/Kruk2/jasna — 자체 소스는 AGPL-3.0-only, 일부 LTX 구성요소는 LTX-2.x Community License, 비공개 보호 모듈 및 유료 모델은 별도 약관입니다.
- Jasna 명령줄 안내: https://github.com/Kruk2/jasna/blob/main/docs/en/cli.md
- Jasna 라이선스 설명: https://github.com/Kruk2/jasna/blob/main/LICENSING.md
- 본 프로젝트의 **독립 개발 GUI/작업 관리 코드**는 MIT License입니다. Jasna 코드는 복사/링크/패키징하지 않았습니다. Jasna를 번들로 묶거나 개조해 배포하려면 원 프로젝트 및 제3자 구성요소의 라이선스 의무를 따로 확인해야 합니다.
