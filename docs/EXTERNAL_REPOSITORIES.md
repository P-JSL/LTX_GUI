# 공개 구현 비교 및 채택 근거 (2026-10-28)

본 문서는 확인한 저장소의 **알고리즘·아이디어**와 본 프로젝트에 실제 구현한 코드를 구분합니다.
다른 저장소의 코드를 그대로 복사하거나 가중치를 본 저장소에 재배포하지 않습니다.
버전과 라이선스가 변경될 수 있으므로 패키징 직전 원본 파일을 다시 검토해야 합니다.

| 원본 저장소 | 확인한 범위 | 라이선스 (확인한 LICENSE 기준) | LTX_GUI 적용 |
| --- | --- | --- | --- |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | LTX-Video/LTX2 DiT, VAE, 스케줄러, 모델 변환과 메모리 절약 | Apache-2.0 | 컴포넌트 책임 분리와 scheduler 개념만 참조; 구현 소스 직접 작성 |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | 타일링, FlashAttention 계열 IO 최적화 | BSD-3-Clause | PyTorch의 `scaled_dot_product_attention` 활용; CUDA 커널 직접 복제/패키징 안 함 |
| [facebookresearch/xformers](https://github.com/facebookresearch/xformers) | attention 연산 및 메모리 절약 패턴 | BSD-3-Clause | 런타임 의존성 없이 연산 단위 병목을 비교 분석 |
| [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) | LTX 2.5 DiT/VAE, 8k+1, 모델 위치, 체크포인트/텍스트 조건 | LTX 2.x Community License 등 구성요소별 별도 조건 | 모델 규격만 연구. 이 프로젝트의 소형 모델은 22B 가중치와 비호환 |
| [sczhou/ProPainter](https://github.com/sczhou/ProPainter) | 영상 인페인팅, 광학 흐름 기반 전파 | S-Lab License 1.0, 비상업 제한 | 코드를 복사하지 않음. 자체 마스크/광학 흐름 처리 유지 |
| [MCG-NKU/E2FGVI](https://github.com/MCG-NKU/E2FGVI) | Flow-guided video inpainting | CC BY-NC 4.0 | 코드를 복사하지 않음. 시공간 품질 검증 지표만 연구 |

## 새 구현의 정확한 범위

- `ltx_gui/neural/model.py`: 프로젝트에서 직접 작성한 소형 DiT, 3D RoPE,
  AdaLN/MLP, Conv VAE, 선택적 causal temporal depthwise convolution.
- `ltx_gui/neural/window_ops.py`: lossless reshape 윈도 분할 및 복원,
  패딩 키 마스크, 원래 위치를 보존하는 Rotary embedding 인덱싱.
- `ltx_gui/native_engine.py`: 8k+1 시간창 단위 스트리밍 추론, 한 프레임 문맥 재사용,
  모자이크 마스크 합성, 기존 FFmpeg 출력·오디오 보존 경로 연결.
- `ltx_gui/neural/train_cli.py`: 별도 학습 데이터로 모델 체크포인트를 생성하는 CLI.
- 외부 PyTorch/OpenCV/FFmpeg는 일반 연산 라이브러리로 사용합니다.

## 논문·원리

1. [FlashAttention](https://arxiv.org/abs/2205.14135): Flash/memory-efficient attention을
   사용하면 NxN attention score 전체를 저장하지 않을 수 있습니다.
2. [Swin Transformer](https://arxiv.org/abs/2103.14030): 지역 윈도 attention의
   복잡도 감소 개념. 본 구현은 shifted window를 포함하지 않으므로 장거리 의존성이 제한됩니다.
3. [LTX-Video](https://arxiv.org/abs/2501.00103): 비디오 잠재공간 처리 및 고압축 VAE 설계.
4. [LTX-2](https://arxiv.org/abs/2601.03233): 영상·오디오 이중 스트림 및 컨디셔닝.
5. [ProPainter](https://arxiv.org/abs/2309.03897): 영상 복원 시간 일관성 연구.
6. [RAFT](https://arxiv.org/abs/2003.12039): 정밀 optical-flow 추정 비교 기준.

논문을 **모두** 검토했다고 주장하지 않습니다. 위 범위에서 핵심 구조와 실행 가능한 개선을 선별했습니다.

## 법적·기술적 경계

- 기존 프로젝트의 `ltx_runtime.py` / `ltx_retake_cli.py`는 **선택적 공식 LTX 2.5
  파이프라인 어댑터**입니다. 독립 모드에서 호출하지 않지만 코드베이스에 남아 있습니다.
- 새 독립 신경망은 Transformer+VAE를 자체 구현했으나 공식 22B 모델의 동일한
  계층·텐서 매핑·Gemma 4 텍스트 처리·오디오 스트림을 재구현한 것이 아닙니다.
- 평균 가중치 또는 무작위 초기화가 학습된 LTX 2.5를 대체하지 않습니다.
  `ltx-train-native`으로 별도 학습해야 하며 학습 데이터의 사용 허락이 필요합니다.
- MIT 라이선스는 **독립 작성된 이 프로젝트 코드**에만 적용됩니다. 사용자가
  설치하는 모델·라이브러리는 각각의 조건을 따릅니다.
