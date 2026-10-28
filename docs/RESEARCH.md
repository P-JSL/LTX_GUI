# 관련 연구·구현 판단 — 독립 프로그램 v0.2.0 (2026-10-28)

이번 보고는 **선별한 주요 논문 9편 및 공식 소스·라이선스 문서**를 검토한 범위다.
관련 연구 '전체'를 망라했다고 주장하지 않는다. 논문의 아이디어를 재구현할 수 있어도
저작권, 가중치, 사전학습 데이터와 특허 문제는 별도로 검토해야 한다.

| 연구 | 기여 | 적용 판정 |
|---|---|---|
| Telea (2004), Fast Marching Inpainting | 공간적 손상 영역의 근방 기반 보간 | **구현**: OpenCV Telea API 호출, ROI만 계산 |
| Farnebäck (2003), Polynomial Expansion Optical Flow | 연속 프레임 운동 추정 | **구현**: OpenCV Farneback, 축소 해상도 추정 및 변위 보정 |
| Barnes et al. (2009), PatchMatch | 근사 최근접 패치 탐색 | 미구현: 큰 마스크용 향후 후보; 참고 소스는 별도 비상업 연구 라이선스 |
| Teed & Deng (2020), RAFT | 학습 기반 밀집 Optical Flow | 미구현: 정확도·속도 비교 필요 |
| Li et al. (2022), E2FGVI | Flow 복원 + propagation + content completion | 미구현: 자체 학습 모델 가중치·데이터 필요 |
| Zhou et al. (2023), ProPainter | mask-guided sparse transformer, dual domain propagation | **설계 참고만**: 불필요 영역 제외; 모델/커널 미복사 |
| Zhang et al. (2024), AVID | Any-length video diffusion inpainting, temporal guidance | 미구현: 실제 diffusion inference 필요 |
| HaCohen et al. (2024), LTX-Video | 높은 VAE 압축률, video diffusion transformer | 미구현: PyTorch/CUDA 및 호환 checkpoint 필요 |
| HaCohen et al. (2026), LTX-2 | Audio-video dual-stream transformer | 미구현: 대규모 공식 모델 별도 라이선스, 메모리·학습 의존성 |

## 원문
- Telea: https://doi.org/10.1080/10867651.2004.10487596
- Farnebäck: https://dblp.org/rec/conf/scia/Farneback03.html
- PatchMatch: https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/
- RAFT: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php
- E2FGVI: https://openaccess.thecvf.com/content/CVPR2022/html/Li_Towards_an_End-to-End_Framework_for_Flow-Guided_Video_Inpainting_CVPR_2022_paper.html
- ProPainter: https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_ProPainter_Improving_Propagation_and_Transformer_for_Video_Inpainting_ICCV_2023_paper.html
- AVID: https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_AVID_Any-Length_Video_Inpainting_with_Diffusion_Model_CVPR_2024_paper.html
- LTX-Video: https://arxiv.org/abs/2501.00103
- LTX-2: https://arxiv.org/abs/2601.03233

## 라이선스 판단
- 기존 Jasna 코드/공식 지원자 모델 및 복호화 구현은 어떠한 형태로도 복사·연동하지 않는다.
  참고한 것은 공개적으로 설명된 설계 아이디어뿐이다.
- 신규 코드는 MIT, OpenCV 4.5+는 Apache-2.0, 별도 설치 FFmpeg 실행 파일의 라이선스는
  사용자 설치본의 인코더 옵션 및 빌드 구성에 따라 달라진다. 설치본을 번들링하지 않는다.
- LTX-2.5 모델과 weights는 별도의 **LTX-2.x Community License**다.
  대규모 사업자 상업 이용, 제한된 사용 목적, 배포 의무 등을 별도로 확인해야 한다.
  '오픈 웨이트'는 MIT 또는 자유로운 무조건 상업 이용을 뜻하지 않는다.
- v0.3부터 LTX-2.5 공개 ltx-core/ltx-pipelines를 **선택적 별도 설치 의존성**으로 사용할 수 있다. JASNA는 호출/복사하지 않고 모델 가중치를 번들링하지 않는다. 공식 LTX 컴포넌트의 라이선스를 MIT라고 표시하지 않는다.
- 현대 딥러닝 기반 인페인팅 엔진으로 동일 성능을 내기 위해서는 공개/직접 학습한 모델,
  실제 입력 영상, VRAM 환경, 품질 지표를 이용한 검증이 필요하다.

## 사실 관계
모자이크 등으로 이미 소실된 픽셀의 **실제 원본 복구를 증명할 수 없다.**
인페인팅 결과는 근접 픽셀 및 움직임으로 만들어낸 가상적 대체 영상이다.


## v0.3.0 보강 검토 (2026-10-28)
- Model Soups (Wortsman et al., ICML 2022): 사전학습 후 유사 학습 경로의 가중치 평균 조건. https://proceedings.mlr.press/v162/wortsman22a.html
- Diffusion Soup (Biggs et al., 2024): 실제 학습된 diffusion checkpoint들의 병합 연구. https://arxiv.org/abs/2406.08431
- ByteTrack (Zhang et al., ECCV 2022): 미검출과 짧은 공백을 다루는 저신뢰 검출 연계. 본 프로그램은 완전한 ByteTrack 아닌 IoU-based 추적만 구현. https://arxiv.org/abs/2110.06864
- LTX-2.5 공식 모델 팩: 22B distilled / Gemma4 / Video VAE / Audio VAE의 **실제 학습된 체크포인트 필수**, 평균 상수 가중치로 대체 불가. https://huggingface.co/Lightricks/LTX-2.5

**독립성 정의:** JASNA 미사용. 생성형 모델 내부 연산을 완전히 처음부터 재구현한 것은 아니며, 선택적으로 공식 Lightricks 공개 라이브러리를 로드한다.
