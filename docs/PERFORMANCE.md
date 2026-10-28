# 실행 성능 및 수학적 하한

프레임 수 F, 가로 W, 세로 H일 때 모든 출력 픽셀을 직렬화해야 한다면
출력 쓰기만으로 Ω(F·W·H) 시간이 필요하다. 전체 영상 처리를 O(n) 미만으로
만드는 것은 출력 방식/문제 정의를 바꾸지 않는 한 불가능하다.

- 정규화 경로 해시 중복 검사: 기대 O(1)/건.
- 사각형/이미지 마스크 생성: O(W·H), 입력 영상당 1회.
- Telea: ROI에 한정한 작업. 영역 크기·근방·경계 처리 비용은 구현과 데이터에 좌우됨.
- Dense optical flow: ROI (A)의 축소판 `flow_scale² · A`에서 계산, 시간 상수항 절약;
  이미지 경계 변위 보정은 폭 비율/높이 비율을 각각 사용해 방향별 오차 축적을 방지.
- 프레임 1장당 RAM O(W·H), 영상 길이 F와 독립적.
- 별도 JASNA 실행/초기화/모델 메모리 사용이 없으므로 작업 특성이 완전히 다르다.
  모델 품질을 일치시키지 못하므로 공정한 상대 성능 향상률은 계산할 수 없다.

## 실행 가능한 마이크로벤치마크
```
python -m scripts.benchmark --width 1920 --height 1080 --iterations 30
```
결과는 기기별 median/p95(컴퓨트만), 해상도, OpenCV 버전을 출력한다.
디코딩/인코딩/디스크를 포함한 처리량은 headless CLI에서 별도로 측정한다.

## 향후 GPU AI 복원 엔진 구축 순서
1. 자체 마스크/장면 추적(재현율/정밀도 수치 필요)
2. Flow-confidence, scene-cut gating + temporal consistency regression tests
3. 공개 가중치가 별도 라이선스인 pretrained diffusion 사용 vs 독자 학습 중 결정
4. CUDA/Triton fused attention & Torch compile; 모델 정확도, 피크 VRAM 검증
5. GPU decode/compute/encode overlap (bounded queues), 중간 host-memory 복사 절감
6. 동일 출력 품질·시드·GPU·입력의 latency/throughput/energy 공정 벤치마크

특히 `CUDA Graph`, FlashAttention, BF16/FP8은 하드웨어·모델의 numerical
stability에 따라 유효성이 달라진다. 해당 최적화 코드는 v0.2.0에 없다.

## v0.4.0: 제한된 3D 윈도 어텐션 (독립 소형 DiT)

고정 깊이 D, head 수 H, 영상 latent token 수 N, 윈도당 토큰 K라 할 때,
전체 토큰 전역 어텐션의 이론적 score-pair 연산은 `O(D·H·N²)`이고,
지역 윈도 어텐션은 `O(D·H·N·K)`입니다. **K를 고정**하면 영상 길이/화면
크기에 대해 attention score-pair의 처리량이 O(N)이 됩니다. 각 block의
linear/MLP/Conv와 인코딩 출력 하한도 O(N) 이상입니다.

구현은 `window_ops.partition_3d`와 `valid_window_keys`를 통해 padding을
masking하고 위치 embedding의 원래 절대 좌표를 유지합니다. 창 경계를 넘는
상호작용은 없습니다. 따라서 **원래 전역 모델과 기능적으로 동일하지 않으며**
실제 훈련 데이터에서 품질 저하 여부를 검증해야 합니다.

동일한 내부 파라미터를 복제하여 계산한 **CPU 마이크로벤치마크**
(2026-10-28, AMD EPYC 9V74 가상화 환경, PyTorch 2.10.0+cpu,
단일 Torch CPU thread, T=2,H=W=32,N=2048, window=2×4×4(K=32),
warmup 3회, 15회 중앙값) 예시:

- 전역: 16.113 ms / forward
- 윈도: 3.854 ms / forward
- 관측 비율: 약 4.18× (이 조건의 단일 CPU 실행에만 해당)
- token pair 상한: 전역 4,194,304, 윈도 65,536 (이론적 64배 감소)
- 작은 N=512에서는 동일 단일 스레드 환경에서 1.893 ms vs 1.023 ms
  (약 1.85×); multi-thread 측정의 편차도 커서 보편적인 속도 향상률이 아님.

재현: `python -m scripts.bench_neural_attention --iterations 15 --warmup 3 --threads 1 --height 32 --width 32 --device cpu`

전체 영상 처리량·품질과 CUDA 개선은 측정하지 않았고, `torch.compile`은 동일한
model 객체에서 컴파일 러너를 재사용하도록 변경했습니다. clip 단위 생성은
이웃 chunk의 원본 마지막 프레임 1장을 문맥으로 재사용하며, chunk 경계의
생성 일관성은 추가 검증이 필요합니다.
