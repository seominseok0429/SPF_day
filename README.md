# SSP126-only SPF

기존 SPF month-to-day 구조와 학습 손실을 정리한 **코드 전용** 저장소입니다.
SFNO 및 DiffESM은 포함하지 않습니다. 모델 데이터, 가중치, 생성 결과도 포함하지 않습니다.

## 변경 사항

- 기본 학습과 두 추가 학습 단계 모두 **SSP126만** 사용합니다.
- 평균/표준편차 정규화도 **SSP126 학습 데이터만**으로 새로 계산합니다.
- 기본 flow-matching 21,000 updates → extreme 보조 손실 3,000 updates →
  rain-budget 보조 손실 3,000 updates. 총 27,000 updates입니다.
- 4개 GPU, global month batch 24, 월 길이별 배치, Euler 30 steps와 마지막
  2 steps의 truncated gradient, 원래 손실 식/가중치를 유지합니다.
- 26개 ensemble은 각 member의 **연간 지수를 먼저 계산한 뒤 지수 평균**입니다.
  일별 날씨 평균의 지수나 member 점수 평균이 아닙니다.

## 조건 입력

이 모델은 **월별 4-field 조건 → 일별 생성기**입니다.
학습 조건은 SSP126 학습 데이터의 월평균입니다.
추론 시에는 외부 월별 조건 입력과 입력 메타데이터 JSON을 제공합니다.
월별 조건 생성 모델 자체는 이 저장소에 포함되어 있지 않습니다.

## 설치

Linux / Python 3.12 / CUDA GPU 네 개가 원래 학습 구조의 기준입니다.
새 환경에 upstream 의존성을 설치하세요. 아래 명령은 설치 예시이며 자동 실행되지 않습니다.

```bash
git clone https://github.com/stanfordmlgroup/spf.git third_party/spf
git -C third_party/spf checkout a7624ae25b47b474d5a31ca22b3ac6e6907795dd
python -m pip install -e third_party/spf
python -m pip install -e '.[test]'
export SPF_SOURCE="$PWD/third_party/spf"
```

Upstream의 flash-attention 요구사항은 CUDA 환경에 맞춰 설치해야 합니다.
Upstream forward의 단일 no-grad 구문을 gradient-enabled 문맥으로 바꾸는 어댑터를
사용합니다. upstream 파일은 수정하지 않고 commit과 원본 파일 hash를 검사합니다.

## 데이터 준비

입력은 NorESM2-MM 한 member의 **SSP126** NetCDF 하나입니다.
변수는 tas/tasmax/tasmin/pr, 단위는 K/K/K/kg m-2 s-1, 격자는
192 lat × 288 lon, calendar는 noleap입니다.
공통 완전 월 범위 2015-01-01~2100-11-30의 1,031개월/31,359일만 학습합니다.
불완전한 최종 2100년 12월은 학습에서 제외합니다.

정적 AR6 geometry NPZ는 `mask`(int, 192×288, ocean=-1, land regions 46개),
`lat`, `lon`으로 구성하고 NetCDF 좌표와 정확히 같아야 합니다.
실험에서 사용한 동일한 정적 region mask를 별도로 제공하세요.

```bash
mkdir -p data runs
python -m spf_ssp126.prepare \
  --ssp126 /path/to/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp126.nc \
  --region-mask /path/to/ar6_geometry.npz \
  --output data/ssp126
```

새 data 디렉터리에만 씁니다. 출력이 이미 존재하거나 여유 공간이 부족하면 실패하며
다른 파일을 삭제하지 않습니다. 실패한 부분 출력은 자동 재시도하지 않습니다.

## 학습

학습을 실행하려는 사용자가 각 단계를 명시적으로 실행합니다. 자동 평가/후속 작업은 없습니다.
GPU 0–3만 사용하고 GPU4는 사용하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m spf_ssp126.train --stage base --data data/ssp126 --output runs/base

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m spf_ssp126.train --stage extreme --data data/ssp126 --output runs/extreme \
  --resume runs/base/step_021000.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m spf_ssp126.train --stage budget --data data/ssp126 --output runs/budget \
  --resume runs/extreme/step_024000.pt
```

AdamW model/optimizer 및 4-rank RNG를 이어받으며 LR은 base=1e-4,
extreme/budget=2e-5입니다. stage 중간 재개도 `--resume`을 쓰되 새 output 경로를
지정해야 합니다. Budget 단계 시작 때만 원래대로 새 calendar-shuffle epoch를 시작합니다.

## 추론 및 앙상블

외부 monthly condition NPZ의 필드는 `monthly`(N,4,192,288; 물리 단위),
`days`(N; 28/30/31), `year`, `month`, `lat`, `lon`입니다.
입력 메타데이터는 `--condition-provenance` JSON으로 전달합니다.
필수 필드는 [sample.py](spf_ssp126/sample.py)에 정의되어 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m spf_ssp126.sample \
  --checkpoint runs/budget/step_027000.pt \
  --conditions /path/to/independent_monthly_conditions.npz \
  --condition-provenance /path/to/conditions_provenance.json \
  --members 26 --output outputs/members
```

member 일별 결과를 **변경 없는 공식 annual-index 계산기**에 각각 넣으세요.
이 저장소는 공식 계산기/역사적 threshold/reference 자료를 다시 구현하거나 배포하지 않습니다.
그 결과를 `years`(2015..2100)와 15개 field(TXx/TNn/SU/TR/FD/ID/WSDI/CSDI/GSL/
Rx5day/CDD/CWD/R95p/R10mm/SDII), 각 shape=(86,192,288)의 NPZ로 전달합니다.

```bash
python -m spf_ssp126.ensemble --count 26 --output outputs/ensemble_annual.npz \
  outputs/annual/member_*.npz
python -m pytest
```

SSP126 이외 시나리오는 학습/정규화로 열지 않습니다.
결과는 공식 scorer로 별도 평가할 수 있습니다.
