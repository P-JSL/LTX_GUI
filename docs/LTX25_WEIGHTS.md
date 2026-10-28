# LTX-2.5 학습 가중치 및 수학적 검토 (2026-10-28)

## 1. 왜 '평균 가중치'로 복원 모델을 만들 수 없는가

행렬 `W∈R^(m×n)`를 모든 원소가 동일한 평균 `μ`로 바꾸면 `W'=μ 1_m 1_n^T`이며
`rank(W')≤1`입니다. `W'x = μ(sum_j x_j)1_m`이므로 개별 특징별 학습 정보가 사라집니다.
다층 비선형 모델의 학습된 가중치·정규화·위치 임베딩을 이렇게 대체하면 원 모델과
전혀 다른 함수를 계산합니다. Xavier/Kaiming 초기화처럼 평균 0인 *확률 분포로 샘플링*
하는 초기화도 **이미 학습한 확산 모델**을 대신할 수 없습니다.

## 2. 평균이 유효할 수 있는 범위

호환 모델 매개변수 `θ1...θk`의 평균은 `θ_avg = Σ α_i θ_i`, `Σα_i=1`, `α_i≥0`입니다.
Model Soups (ICML 2022), Diffusion Soup (2024)는 여러 이미 학습한 유사한
모델의 매개변수 평균에 대한 사례입니다. 항상 성립하는 정리가 아니며 모델 구조,
토큰 임베딩 순서, 텐서 이름 및 정렬, 학습 영역이 맞아야 합니다. 실측 평가가 필요합니다.
`weights.validate_weight_mix()`는 평균 실험의 기본 입력 제약만 검사합니다.
실제 merge·생성 품질 향상을 구현했다고 주장하지 않습니다.

## 3. 공식 기본값 (distilled)

LTX-2.5 공개 split checkpoint 팩:
- DiT: `diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors`
- Text: `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`
- Video VAE (lighter): `vae/ltx-2.5-video-vae-conv-bf16.safetensors`
- Audio VAE: `vae/ltx-2.5-audio-vae-bf16.safetensors`

이 모델은 8-step distilled schedule / CFG=1을 사용합니다. 다른 체크포인트로
자동 대체하거나 평균값으로 초기화하지 않습니다. `ltx_runtime.py`는 모델 파일을
검증하고 *공식 LTX-2 파이프라인*을 사용자 환경에서 직접 가져오며 JASNA는 호출하지 않습니다.
LTX 공개 구성요소 및 모델의 LTX-2.x Community License는 별도로 적용됩니다.

## 4. 기능·제약 경계

- `ltx_retake_cli.py`: 실제 DiT/VAE를 사용하는 시간 구간 *재생성* 어댑터
  + 선택적인 마스크 영역만 합성. 생성기 자체가 공간 마스크 조건으로 학습된 것은 **아닙니다**.
- `detection.py`: 전용으로 학습된 ONNX 모자이크 탐지기 필요. 임의의 YOLO 가중치를
  모자이크 탐지기로 오인하지 않습니다.
- `tracking.py`: IoU 단기 추적. ByteTrack 전 기능/RAFT를 구현했다고 주장하지 않습니다.
- `gpu_ops.py`: PyTorch CUDA mask compositing optional `torch.compile`.
  DiT 커널의 custom CUDA 구현/속도 개선 검증은 아닙니다.
- `quality.py`: 진짜 깨끗한 기준 영상이 있는 경우 PSNR·masked PSNR·SSIM·outside-MAE.
  모자이크 입력 자체는 기준 정답이 될 수 없습니다.

## 출처

- LTX-2 architecture: https://arxiv.org/abs/2601.03233
- LTX-Video architecture: https://arxiv.org/abs/2501.00103
- Official LTX-2.5 checkpoints: https://huggingface.co/Lightricks/LTX-2.5
- Official LTX code: https://github.com/Lightricks/LTX-2
- LTX-2.x Community License: https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- Model soups: https://proceedings.mlr.press/v162/wortsman22a.html
- Diffusion Soup: https://arxiv.org/abs/2406.08431
- ProPainter: https://arxiv.org/abs/2309.03897
- RAFT: https://arxiv.org/abs/2003.12039
- ByteTrack: https://arxiv.org/abs/2110.06864
