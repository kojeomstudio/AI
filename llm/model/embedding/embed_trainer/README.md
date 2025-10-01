# Embed Trainer (General-Purpose Embedding Fine-tuning)

`embed_trainer`는 `sentence-transformers` 라이브러리를 기반으로 구축된 간결하고 효율적인 임베딩 모델 파인튜닝 프로젝트입니다. `MultipleNegativesRankingLoss`를 사용하여 주어진 텍스트 쌍(anchor-positive)에 대해 강력한 대조 학습을 수행합니다.

**보다 상세한 파인튜닝 원리 및 코드 아키텍처에 대한 설명은 [docs/fine_tuning_principles.md](./docs/fine_tuning_principles.md) 문서를 참고하세요.**

## 주요 특징

- **고성능 학습:** `sentence-transformers`의 `MultipleNegativesRankingLoss`를 통해 배치 내 네거티브를 효과적으로 활용하여 높은 학습 효율과 정확도를 달성합니다.
- **간결한 설정:** `config.json` 파일 하나로 모델, 데이터, 학습 파라미터를 모두 관리합니다.
- **쉬운 사용법:** 복잡한 과정 없이 단일 명령으로 학습을 시작할 수 있습니다.
- **모듈화된 구조:** `Trainer`, `EmbeddingModel` 등 역할이 명확히 분리된 코드로 유지보수와 확장이 용이합니다.

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
# sentence-transformers가 포함되어 있어야 합니다.
# requirements.txt에 sentence-transformers 추가가 필요할 수 있습니다.
```

### 2. 데이터 준비

학습 데이터는 `jsonl` 형식이어야 하며, 각 줄은 `anchor`와 `positive` 텍스트를 포함하는 JSON 객체여야 합니다.

**`data/pairs.jsonl` 예시:**
```json
{"anchor": "의자", "positive": "앉는 가구"}
{"anchor": "CPU", "positive": "중앙 처리 장치"}
```

### 3. 설정 파일 작성

`config.json` 파일을 생성하고 학습 설정을 구성합니다.

**`config.json` 예시:**
```json
{
  "model_id": "BAAI/bge-small-en-v1.5",
  "hf": {
    "trust_remote_code": true
  },
  "train": {
    "batch_size": 32,
    "epochs": 5
  },
  "optim": {
    "lr": 2e-5
  },
  "data": {
    "pairs_path": "./data/pairs.jsonl"
  },
  "output": {
    "save_dir": "./outputs/my-finetuned-model"
  }
}
```

### 4. 학습 실행

다음 명령어로 학습을 시작합니다.

```bash
python -m embed_trainer.train --config config.json
```

학습이 완료되면 `output.save_dir`에 지정된 경로에 파인튜닝된 모델이 저장됩니다.

## 프로젝트 구조

- `train.py`: 메인 실행 스크립트
- `trainer.py`: 핵심 학습 로직을 담은 `Trainer` 클래스
- `model.py`: `SentenceTransformer` 모델을 래핑하는 `EmbeddingModel` 클래스
- `data_utils.py`: 데이터 로딩 및 전처리 유틸리티
- `config.json`: 학습 설정 파일
- `docs/`: 상세 기술 문서 폴더