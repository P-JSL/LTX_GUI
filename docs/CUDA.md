# CUDA 추론 구현 및 검증 범위 (v0.5.0)

## 구현 현황 및 비구현 기능 구분

- 독립 모델(연구용 경량 DiT/VAE)의 `LatentFlow.sample()` 매 단계에 **독립 CUDA C++ / Triton 융합 Euler + 원본 latent 보존 커널** 연결. 메모리 병목 후보인 포인트별 연산을 1회 커널 실행으로 처리합니다.
- 최종 masked pixel blend 또한 독립 CUDA C++ / Triton 커널에 연결했습니다. 마스크가 0인 픽셀은 생성값이 아닌 원본을 그대로 통과합니다 (다만 비디오 재인코딩의 손실은 발생할 수 있습니다).
- Triton `RMSNorm` (추론 전용), `uint8 RGB + mask`의 GPU 정규화/레이아웃 변환 융합. 선택적인 **pinned memory** 업로드는 큰 클립에만 적용하여 할당 비용을 줄입니다.
- 3D RoPE 캐시(최대 4개의 영상 geometry)로 동일 프레임/해상도 확산 단계를 다시 계산하지 않습니다.
- `torch.compile(mode="reduce-overhead")`는 **실제 DiT forward에 적용**됩니다. 기존에 일반적인 `nn.Module.forward`가 없는 `LatentFlow`에 적용하던 코드를 수정했습니다.
- 자동 모드: 빌드된 CUDA C++ 확장 → Triton → PyTorch 기본 연산. CPU는 항상 PyTorch 대체 경로입니다. 강제 백엔드(`cuda`/`triton`)가 불가능하면 **명시적으로 오류**를 반환하며, CPU 연산을 GPU라고 표시하지 않습니다.
- `bf16` / `fp16` AMP는 옵트인입니다(기본 `none`). GPU와 모델의 정밀도 허용 범위에 따라 오차/품질이 달라질 수 있습니다. 학습/gradient 경로는 변경하지 않습니다.

**v0.5.0 당시 미구현이었으나 v0.6.0에서 실험 경로 추가:** TensorRT/FP8 시도, CUDA Graph 추론, GPU 영상 I/O (상세: [ACCELERATION.md](ACCELERATION.md)). **여전히 미구현:** 정식 LTX-2.5 22B 가중치 호환 Transformer/VAE, 맞춤형 FlashAttention 전체 재작성, 검증된 GPU 비디오 전체 zero-copy 구현. `torch.compile`의 내부 CUDA Graph 사용 가능성은 PyTorch 환경에 의존합니다. 현재 커널이 위 기능을 대체한다고 주장하지 않습니다.

## 실행 전 요건

1. NVIDIA GPU + 호환 드라이버, CUDA 대응 PyTorch wheel(시스템 Toolkit과 PyTorch CUDA ABI 확인), **학습된 LTX_GUI `.pt` 모델**
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) `nvcc` 및 C++ 컴파일러(Windows: MSVC Build Tools) — CUDA C++ 확장을 직접 빌드할 때만 필요
3. Triton 백엔드는 호환 OS/아키텍처/버전에서만 지원; Windows는 Triton 배포판 지원 여부를 직접 확인해야 합니다. 정상적인 `pip install -e .`는 CUDA 빌드를 요구하지 않습니다.

```powershell
# GPU PyTorch wheel은 공식 PyTorch 설치 안내에서 장치/드라이버에 맞게 설치
python -m pip install -e ".[native,test]"

# 1) PyTorch 순수 CUDA (별도 컴파일 없이 작동)
ltx-restore video.mp4 --rectangle 0.2,0.2,0.8,0.8 --method neural --checkpoint checkpoints/native.pt --device cuda:0 --gpu-backend torch

# 2) 선택형 CUDA C++ 확장 빌드 (nvcc와 컴파일러 필요)
python setup_cuda.py build_ext --inplace
ltx-restore video.mp4 --rectangle 0.2,0.2,0.8,0.8 --method neural --checkpoint checkpoints/native.pt --device cuda:0 --gpu-backend cuda

# 3) 사전 설치한 호환 Triton 백엔드
ltx-restore video.mp4 --rectangle 0.2,0.2,0.8,0.8 --method neural --checkpoint checkpoints/native.pt --device cuda:0 --gpu-backend triton

# 혼합정밀도 (지원 GPU에서만): --amp bf16 (또는 fp16)
```

GUI에서도 **GPU 커널**과 **혼합 정밀도**를 설정할 수 있습니다.

## 복잡도 및 병목 근거

입력 텐서가 `[B,C,T,H,W]`일 때 전체 출력 원소 수를 `M=B·C·T·H·W`로 둡니다. 커널은 thread당 원소 1개를 처리하고 채널 차원의 마스크를 broadcast하므로 작업량은 **Θ(M)**, 출력만 기록해도 **Ω(M)**의 메모리 쓰기가 필요합니다. 스칼라 GPU 커널은 이 하한에 도달하지만 CPU↔GPU 전송/전체 모델의 시간이 반드시 줄어드는 것은 아닙니다.

일반 PyTorch 표현식 `(z-step*v)*mask + known`에는 여러 중간 텐서 할당과 여러 커널 실행이 포함될 수 있습니다. 명시적으로 융합된 CUDA 커널은 같은 계산을 연속된 GPU 쓰기 한 번으로 처리하여 임시 메모리 왕복을 감소시킬 수 있습니다. *수치적 오차는 FP16/BF16 및 연산 순서에 따라 존재*합니다.

성능상의 부작용 가능성: noncontiguous Transformer output의 연속 메모리 변환(추가 쓰기), 첫 CUDA JIT/torch.compile 비용, 소형 텐서의 런치 오버헤드, Tensor Core의 타일 제약. **반드시 장치별로 벤치마크해 더 빠른 쪽을 선택**해야 합니다.

## 검증 명령

```bash
# CPU 자동 테스트. CUDA 장치가 없으면 CUDA 테스트는 skip됩니다.
python -m pytest -q
python -m scripts.bench_cuda_ops --device cpu --backend auto --iterations 50
# GPU correctness + CUDA Event 측정 (GPU 장비에서 실행)
python -m pytest -q tests/test_cuda_backend.py
python -m scripts.bench_cuda_ops --device cuda --backend auto --iterations 100
# 모델별 CUDA kernel 프로파일링 (학습된 체크포인트 필요)
python -m scripts.profile_cuda --checkpoint checkpoints/native.pt --trace perf/profile.json
```

**현재 CI는 NVIDIA GPU 장비를 제공하지 않으므로** CUDA C++ NVCC 빌드, Triton 실구동, Windows GPU exe, throughput·VRAM 비교 결과는 미검증입니다. CPU green 테스트만으로 CUDA 성능/빌드 성공을 주장하지 않습니다.

## 기술 참조

- [PyTorch custom Triton kernels](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
- [Triton fused normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [CUDA C++ Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA async transfer and overlap](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc)
