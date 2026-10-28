# LTX GUI — Independent Restoration Engine (v0.2.0)

**JASNA를 실행·연동·복사하지 않는 독립 Windows GUI 영상 인페인팅 응용프로그램입니다.**
Python/PySide6 UI + OpenCV(C++ 내부 최적화 연산) + 별도 설치된 FFmpeg를 사용합니다.
JASNA, ComfyUI 또는 상용 LTX 모델/키/비공개 모듈은 필요 없습니다.

> ⚠️ 현재 **LTX diffusion 모델을 구현한 것은 아닙니다.** 자체 구현한 고전적 공간·시간축
> 인페인팅으로 선택 영역을 근사 복원하는 실동작 버전입니다. 큰 면적·복잡한 원본 콘텐츠를
> 재구성하거나 기존에 사라진 픽셀을 진짜 원본으로 되찾을 수는 없습니다.

## 기능
- 네이티브 GUI에서 다수의 영상 파일·폴더 입력, 파일당 한 개 사각형 영역 선택 또는 마스크 이미지 지정
- Telea 빠른 공간 인페인팅 / Farneback optical flow 기반 시간축 보정 인페인팅
- 바깥 영역은 복원 엔진이 변경하지 않음 (재인코딩에 따른 압축 손실은 존재)
- 입력 영상과 원본 오디오를 조합한 H.264 영상 출력(.mkv/.mp4), 오디오는 AAC로 변환
- CPU `libx264` 인코더 또는 NVIDIA `h264_nvenc` 하드웨어 인코더 선택
- 별도 작업 스레드, 진행률/FPS, 정지, 처리 전 검증, 충돌·덮어쓰기 방지, 임시 파일 정리
- Unicode 경로 마스크 로딩, 환경설정 JSON 원자 저장, 별도 CLI/API/자동 테스트

## 설치 (Python 3.12 권장)
1. Windows에 `ffmpeg.exe`를 설치하고 PATH에 추가하세요. `ffmpeg -version` 확인.
2. `py -3.12 -m venv .venv`
3. `.venv\Scripts\activate`
4. `python -m pip install -e ".[test]"`
5. `python -m ltx_gui` (GUI), `python -m pytest -q` (테스트)

독립 CLI 실행 예시 (사각형 정규화 좌표 0~1):
```
python -m ltx_gui.cli "C:\videos\in.mp4" --rectangle 0.25,0.25,0.75,0.75 --method temporal --output "C:\videos\out.mkv"
```
마스크 PNG의 **흰색은 생성/변경**, 검정색은 원본 유지:
```
python -m ltx_gui.cli in.mp4 --mask mask.png --encoder libx264
```

GPU NVENC는 영상 **인코딩**에만 사용합니다. `--encoder h264_nvenc`를 선택하려면
FFmpeg가 해당 인코더와 NVIDIA 드라이버를 지원해야 합니다. GPU 복원 추론 자체는 없습니다.

GUI 빌드: `powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1`.
GitHub Actions의 Windows ZIP 아티팩트에는 Python 패키지가 묶이며 `ffmpeg.exe`는 포함되지 않습니다.

## 한계
- 자동 모자이크/개체 감지 및 이동하는 영역을 추적하는 마스크는 구현하지 않았습니다.
- 복원은 근사 인페인팅으로 생성적 LTX-2.5 또는 학습형 ProPainter 품질을 보장하지 않습니다.
- VFR은 캡처된 FPS로 CFR 재인코딩되므로 타임스탬프 일치가 중요한 자료에 사용 전 검증 필요.
- `h264_nvenc`가 없는 GPU/FFmpeg에서는 직접 CPU 인코딩으로 설정 변경 필요.
- FFmpeg 설치와 OS/GPU 드라이버가 있어야 미디어 통합 기능이 실행됩니다.
- UI 실제 Windows 구동은 Linux 테스트 환경에서 직접 확인하지 못했습니다.

## 기술·라이선스 문서
- [연구 검토](docs/RESEARCH.md)
- [구현 구조 및 제약](docs/ANALYSIS.md)
- [복잡도 및 성능](docs/PERFORMANCE.md)
- 독립 코드: MIT (LICENSE); OpenCV>=4.5: Apache-2.0; FFmpeg(외부 설치): 빌드별 라이선스 별도
- JASNA: https://github.com/Kruk2/jasna (아이디어·비교 문서 참고, 소스 복사·실행 없음)
- LTX-2.x 사용 조건: https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x

원본 JASNA 버전과 모델, 가중치, 실측 GPU 결과를 본 프로그램의 성능이나 라이선스에 포함해
표현하지 않습니다. 이 프로그램의 독립성은 **실행 중 JASNA 미사용**을 뜻합니다.
