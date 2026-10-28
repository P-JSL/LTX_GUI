# Jasna GUI 및 LTX 기술 분석 (2026-10-28)

## 실제 검토한 공개 소스 (Kruk2/jasna)

- `jasna/gui/app.py`, `jasna/gui/processor.py`: Tk 기반 UI 제어, 백그라운드 워커, 작업 재사용 및 자원 정리
- `jasna/gui/models.py`, `jasna/gui/queues.py`: 작업 상태/옵션, 메인스레드 이벤트 디스패치
- `jasna/ltx/plan.py`: 감지 영역 추적, 121프레임 윈도와 88/104프레임 이동 간격
- `jasna/ltx/restore.py`, `jasna/ltx/inference.py`, `jasna/ltx/kernels.py`: 모델 추론, VRAM/커널 연동 구조 (소스 레벨 항목 확인, 각 경로 전체에 대한 벤치마크는 미수행)
- `docs/en/cli.md`, `docs/en/models.md`, `docs/en/tuning.md`, `LICENSING.md`: 외부 실행 계약, 하드웨어 제약, 라이선스

Jasna는 이미 GPU 중심 최적화, 불필요한 프레임 작업 감소, LTX 윈도 중복 최소화 및 재실행을 제공합니다. 따라서 새로운 알고리즘이 무조건 더 빠르다고 주장하지 않습니다. 동일 입력/품질/시스템에서 비교 벤치마크가 필요합니다.

## 신규 구성 원칙

```
PySide6(Qt UI, GUI thread)
  ├── 배열 기반 목록 모델 + hash set 파일 중복 검사
  ├── 별도 QThread -> BatchRunner -> 안전한 외부 Jasna CLI 프로세스 1개
  ├── 별도 QThread -> 3초마다 CPU/RAM + NVIDIA nvidia-smi 조회
  └── JSON 설정 파일 (fsync -> atomic replace)
Jasna CLI -> GPU 복원 모델/커널 -> 영상 출력
```

- GPU 연산 중복 방지: 한 번에 하나의 무거운 추론 프로세스만 실행
- GUI 스레드 격리: 파일 읽기, 모델 추론, stderr/stdout 수집, nvidia-smi 실행을 GUI 스레드에서 수행하지 않음
- 재시도 정확성: 입력/출력/옵션 유지, 작업 디렉터리는 절대 경로 조합의 SHA-256 prefix로 결정
- 실행 경로 안전성: `subprocess.Popen(argv)`와 `shell=False` 사용, 공백/한글 파일명에 대한 shell injection 없음
- 파일 안전성: 입력-출력 동일 경로/기존 파일/대기열 대상 중복 실행 전 거부
- 중지: POSIX 세션의 SIGINT, Windows 신규 프로세스 그룹의 CTRL_BREAK_EVENT로 먼저 소프트 중지 시도; 일정 시간 초과 시 강제 종료 가능
- 오류 복구: 개별 작업 실패 시 다음 작업 계속; 체크포인트/임시 파일은 백엔드가 관리

## 왜 Python과 Qt인가

| 역할 | 선택 | 근거 |
|---|---|---|
| GUI | Python + PySide6 (Qt) | 네이티브 데스크톱, signal/slot 및 배포 지원 |
| 모델 추론 | 외부 Jasna 런타임 | 기존 검증된 CUDA/TensorRT/ROCm 설정·라이선스 처리 유지 |
| 영상 처리/커널 (향후 자체 엔진) | C++/CUDA, HIP, Triton | 연산 집중 구간의 kernel fusion, 최소 CPU↔GPU 전송 |
| 배포 | PyInstaller/Windows CI | GUI 빌드 단순화 (별도 Jasna 설치 필요) |

GUI·작업 제어를 C++로 다시 작성해도 대부분의 영상 처리 시간은 DiT 추론/영상 디코딩·인코딩에 지배됩니다. 실제 속도는 엔진의 최적화와 모델 품질·VRAM 조건이 결정합니다.

## 정확히 구현하지 않은 부분

- Jasna의 감지·복원 코드 복제/개조/재라이선싱
- 암호화된 후원자 전용 LTX 체크포인트 또는 보호 모듈 접근
- 영상 내부 구간 편집용 프레임 썸네일 미리보기, 원본/복원 프레임 직접 비교
- CPU만으로 동작하는 LTX 복원 추론 엔진
- 실제 GPU 렌더링/출력 화질에 대한 정량 검증

이 부분은 추가 작업 또는 사용자 환경( GPU, 모델, 적법한 사용 권한)이 있어야 완성 가능합니다.
