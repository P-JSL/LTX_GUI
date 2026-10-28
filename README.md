# LTX GUI — Independent Restoration Engine (v0.4.0)

**JASNA를 실행·연동·복사하지 않는 독립 Windows GUI 영상 인페인팅 응용프로그램입니다.**
Python/PySide6 UI + OpenCV(C++ 내부 최적화 연산) + 별도 설치된 FFmpeg를 사용합니다.
기본 고전적 복원 모드에서는 JASNA 또는 생성형 모델이 필요하지 않습니다. 선택적 LTX-2.5 생성 모드는 공식 공개 학습 가중치와 별도 설치한 LTX 구성요소가 필요합니다.

> ⚠️ 기본 GUI 복원은 **고전적 공간·시간축 인페인팅**입니다. 별도 LTX-2.5 생성형 CLI는 공식 학습 가중치가 있을 때만 동작합니다. 고전적 공간·시간축
> 인페인팅으로 선택 영역을 근사 복원하는 실동작 버전입니다. 큰 면적·복잡한 원본 콘텐츠를
> 재구성하거나 기존에 사라진 픽셀을 진짜 원본으로 되찾을 수는 없습니다.


## v0.4.0 — 독립 신경망 실행 및 병목 개선

**기존 고전적 복원 모드는 변경하지 않았습니다.** 새 GUI의 `독립 학습 신경망
(DiT + VAE)` 옵션을 선택하면 `ltx-core`, `ltx-pipelines`, `JASNA`를 호출하지
않고 이 저장소의 `ltx_gui/neural/` 모델을 실행합니다.

- Conv VAE + Rectified Flow DiT + 3D RoPE + 학습 가능한 *causal temporal*
  depthwise 필터(훈련 때 옵션 지정).
- `scaled_dot_product_attention`, 선택적 CUDA `torch.compile` **동일 모델에
  대해 1회 컴파일**, 고정 3D 윈도 어텐션으로 token pair O(N²) → O(N×K).
- 8k+1 영상 윈도/1프레임 문맥 중첩, 전체 비디오 버퍼링 없이 FFmpeg 출력 및
  오디오 remux, 빈 마스크의 GPU 추론 생략, 출력 덮어쓰기 방지.
- GUI와 `ltx-restore --method neural`, `ltx-train-native` 경로에서 이용 가능.

**필수 제약:** 자체 모델은 연구용으로 경량화하여 공식 **LTX-2.5 22B
Transformer/VAE와 가중치가 호환되지 않습니다.** 공식 가중치로 실행되지
않으며, 학습된 자체 체크포인트를 준비해야 합니다. 학습되지 않은 모델은
의미 있는 복원 품질을 제공하지 않습니다. CUDA 성능·실제 품질은 미검증입니다.

학습 데이터 형식: 작은 데이터는 `.npz`, 대용량은 메모리 매핑 가능한
`videos.npy`·`masks.npy`를 담은 폴더입니다. `videos` = `[N,T,H,W,3] uint8`,
`masks` = `[N,T,H,W] uint8`; 올바른 복원 레이블을 가진 적법한 데이터가 필요합니다.

```shell
python -m pip install -e ".[native,test]"
ltx-train-native train.npz --output checkpoints/native.pt --vae-steps 1000 --flow-steps 1000 --device cuda --causal-temporal
ltx-restore input.mp4 --rectangle 0.20,0.20,0.70,0.70 --method neural --checkpoint checkpoints/native.pt --device cuda --steps 8 --clip-frames 9
python -m scripts.bench_neural_attention --device cpu --iterations 15
```

`1000`단계는 단지 CLI 실행 예이며 충분한 훈련이라고 보장하지 않습니다.
CUDA가 없는 환경에서는 `--device cpu`를 사용하고, 전용 모델은 작은
테스트 클립에서 검증한 뒤 전체 영상에 사용하세요. 8k+1 이상의 클립 길이는
GPU 메모리와 품질을 고려해 직접 선택합니다.

윈도 attention의 지역성 때문에 장거리 연속성은 기존 global attention보다
낮을 수 있습니다. 컴파일·윈도 길이·FPS/화질 측정은 동일 체크포인트와
마스크 조건에서 평가해야 합니다.

[타 저장소 비교/라이선스](docs/EXTERNAL_REPOSITORIES.md) ·
[성능/복잡도 분석](docs/PERFORMANCE.md)

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

기본 고전적 복원 모드에서 GPU NVENC는 영상 **인코딩**에만 사용합니다. `--encoder h264_nvenc`를 선택하려면
FFmpeg가 해당 인코더와 NVIDIA 드라이버를 지원해야 합니다. 독립 신경망 모드에서는 적절히 학습된 체크포인트와 CUDA가 있을 때 PyTorch GPU 추론을 사용합니다.

GUI 빌드: `powershell -ExecutionPolicy Bypass -File scripts/build_windows.ps1`.
GitHub Actions의 Windows ZIP 아티팩트에는 Python 패키지가 묶이며 `ffmpeg.exe`는 포함되지 않습니다.

## 한계
- v0.3.0에서는 ONNX 자동 탐지·IoU 이동 추적이 구현되었습니다. 다만 학습된 모자이크 탐지 모델 파일이 없으면 실행되지 않습니다.
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

## v0.3.0 — 자동 탐지 / LTX 모델 어댑터 / 검증 도구

JASNA 없이 다음을 **별도 모듈로 구현**했습니다.

- `ltx_gui/detection.py`: 사용자 지정 *학습된* 모자이크 ONNX 모델에서 상자 탐지,
  클래스 필터, NMS. `onnxruntime` 별도 설치 필요; 모델 파일은 배포하지 않습니다.
- `ltx_gui/tracking.py`: 탐지 결과를 짧은 프레임 간격에서 추적, 장면 전환 시 초기화,
  프레임별 동적 마스크 생성. 진짜 ByteTrack/RAFT 구현이 아닌 경량 IoU 기반 추적기입니다.
- `ltx_gui/engine.py`: 탐지/추적 자동 선택 및 동적 마스크 복원, 또는 LTX 생성
  결과의 지정 영역 합성. 모델 생성 결과 외부 영역은 입력 프레임에서 직접 복사됩니다.
- `ltx_gui/ltx_runtime.py`: 별도로 합법적으로 설치한 Lightricks의 **공식** LTX-2.5
  `ltx-core`/`ltx-pipelines` 및 학습 체크포인트를 이용해 DiT/Video VAE 추론.
  GUI 프로그램은 JASNA를 사용하지 않습니다. *이 모델 코어는 자체 재구현이 아닌
  공식 컴포넌트 사용*이며 별도 LTX-2.x Community License가 적용됩니다.
- `ltx_gui/gpu_ops.py`: CUDA Tensor의 마스크 합성을 `torch.compile`로 선택적
  융합. 실제 GPU 커널 속도 개선은 장치별 검증 전까지 주장하지 않습니다.
- `ltx_gui/quality.py`: 정답 영상을 이용한 PSNR, masked PSNR, SSIM, 비복원영역
  MAE 검사. `scripts/compare_results.py`는 JASNA를 **실행하지 않고** 미리 생성한
  두 결과를 동일 해상도·FPS·실험 명세로 비교하는 벤치마크 도구입니다.

### 독립 자동 탐지 CLI

```shell
python -m pip install -e '.[detector]'
python -m ltx_gui.cli input.mp4 --auto-detect --detector-model mosaic_detector.onnx --method temporal
```

이 ONNX 파일은 반드시 **모자이크 클래스로 별도 학습된 모델**이어야 합니다.
단순 일반 객체 탐지기의 출력으로는 자동 모자이크 영역을 정확히 찾을 수 없습니다.

### LTX-2.5 시간구간 재생성 + 영역 합성 CLI (실제 체크포인트 필요)

공식 LTX-2 리포지토리의 설치 지침대로 사용자 CUDA 환경에 `ltx-core`,
`ltx-pipelines`를 별도 설치한 뒤 공식 공개 가중치를 `models/LTX-2.5`에 구성합니다.

```shell
ltx-retake input.mp4 --output output.mp4 --weights-root models/LTX-2.5 \
  --prompt 'Natural consistent textures' --start 0 --end 4 \
  --rectangle 0.2,0.2,0.8,0.8
```

공식 retake 모델은 **시간 구간만 조건화**합니다. 위 사각형은 모델 자체의
공간 조건이 아니라 생성 뒤 원본 프레임과 합성할 영역이며, 진짜 공간 조건부
모자이크 복원 LTX가 완성된 것은 아닙니다. 영상 프레임 수는 `8k+1`, 폭과 높이는
32의 배수, 입력 모델 체크포인트는 전부 실제 학습된 파일이어야 합니다.
시작/종료 구간, 음성/화질, GPU 메모리 제약 등은 공식 LTX 설정에도 영향을 받습니다.

### 정량 비교 실험

`python -m scripts.compare_results --source censored.mp4 --reference clean.mp4 --ours ours.mp4 --jasna jasna.mp4 --mask mask.png --manifest experiment.json --output report.json`

`experiment.json`은 입력·마스크 SHA-256, GPU/드라이버, 해상도, FPS, 코덱,
양측 버전, 워밍업 정책을 기록해야 합니다. 실행 시간은 측정값을 기록해야 하며,
완료된 벤치마크 데이터가 없는 경우 속도 개선을 임의로 표시하지 않습니다.
동영상 정답이 없으면 PSNR/SSIM을 원본 영상의 '품질'로 보고하지 않습니다.

자세한 연구·가중치 검토: [docs/LTX25_WEIGHTS.md](docs/LTX25_WEIGHTS.md).

### 실측 검증 상태

현재 CI/로컬은 **CPU 및 합성 테스트**입니다. NVIDIA CUDA GPU, 22B 모델
가중치, 전용 ONNX 탐지 모델이 없는 환경에서는 실제 LTX 출력 및 JASNA 대비
품질·속도를 검증할 수 없으며 그런 결과를 만들어내지 않습니다. CUDA 연산
커널 전체를 독립적으로 재작성하거나 LTX-2.5 모델을 새로 학습한 것은 아닙니다.
