# Embed Trainer (General-Purpose Embedding Fine-tuning)

간결하고 제너럴한 임베딩 모델 파인튜닝 프로젝트입니다. 기존 `embedding/qwen` 폴더의 트레이너를 참고하여, 다양한 Hugging Face 임베딩 모델을 대상으로 InfoNCE 기반 컨트라스티브 학습을 지원합니다.

## 주요 특징
- 모델/토크나이저/학습 하이퍼파라미터를 `config.json`으로 설정
- `pairs.jsonl`(anchor, positive, hard_negatives) 형식 데이터 로딩
- `mean pooling` + InfoNCE 인배치 대조 학습
- CUDA/MPS/CPU 자동 디바이스 선택 및 AMP 사용
- 체크포인트(모델/토크나이저)와 학습 스냅샷 저장

## 프로젝트 구조
- `train.py`: 메인 트레이너 (CLI: `--config`)
- `data_utils.py`: 데이터셋/콜레이터/로더
- `utils.py`: 디바이스/풀링/손실/저장 유틸
- `config.example.json`: 설정 예시
- `requirements.txt`: 의존성

## Tools (데이터 준비)
- `tools/dataset_crawler.py`: 위키/커스텀 URL에서 용어집 크롤링 → CSV/JSONL(pairs) 생성
  - 예: `python embed_trainer/tools/dataset_crawler.py --sources wikipedia_glossary --limit_per_source 500`
  - 오프라인 데모: `--dry-run`
- `tools/dataset_compose.py`: CSV/TSV(용어집 또는 anchor/positive 2열) → pairs.jsonl 변환
  - 예: `python embed_trainer/tools/dataset_compose.py --input glossary.csv --output pairs.jsonl --hard-k 3`

크롤러 의존성 설치: `pip install -r embed_trainer/tools/requirements_crawler.txt`

## 빠른 시작
1) 의존성 설치
```
pip install -r embed_trainer/requirements.txt
```

2) 설정 파일 준비 후 학습 실행
```
python embed_trainer/train.py --config embed_trainer/config.example.json
```

기본 예시는 `tools/data/pairs.jsonl` 또는 `data/pairs.jsonl`를 자동 탐색합니다. 커스텀 경로를 쓰려면 `data.pairs_path`에 절대/상대 경로를 적어주세요.

## 데이터 형식 (JSONL)
한 줄에 하나의 샘플:
```
{"anchor": "...", "positive": "...", "hard_negatives": ["...", "..."]}
```
`hard_negatives`는 생략 가능.

## 설정 파일 필드 요약
- `model_id`: Hugging Face 모델 ID (예: `BAAI/bge-small-en-v1.5`)
- `tokenizer_id`(옵션): 별도 토크나이저가 필요할 경우 지정
- `device`(`auto|cuda|mps|cpu`), `dtype`(`auto|float32|float16|bfloat16`)
- `train`: `batch_size`, `epochs`, `max_length`, `grad_accum_steps`, `steps_per_epoch`(옵션: 고정 step)
- `optim`: `name`(AdamW 등), `lr`, `weight_decay`, `warmup_ratio`
- `data`: `pairs_path`(옵션), 없으면 자동 탐색
- `output`: `save_dir`, `save_name`, `save_metrics`

## 주의/권장
- GPU가 있을 경우 `device=auto`로 CUDA/MPS를 자동 사용
- 너무 큰 배치를 고정하기보다 `grad_accum_steps`로 누적 권장
- 모델/아키텍처 변경 시에도 설정 파일만 교체하여 재사용

## 변경 이력
- v0: 초기 공개. `embedding/qwen`의 구조/로직을 단순화하여 일반화.
