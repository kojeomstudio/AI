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
```

### 2. 데이터 준비

학습 데이터는 `jsonl` 형식이어야 하며, 각 줄은 `anchor`와 `positive` 텍스트를 포함하는 JSON 객체여야 합니다.

**`data/pairs.jsonl` 예시:**
```json
{"anchor": "의자", "positive": "앉는 가구"}
{"anchor": "CPU", "positive": "중앙 처리 장치"}
```

### 3. 설정 파일 작성

`config.json` 파일을 생성하고 학습 설정을 구성합니다. `config.qwen.json`은 Qwen 모델을 위한 예시 설정입니다.

**`config.json` 예시 (주요 설정 필드):**
```json
{
  "model_id": "BAAI/bge-small-en-v1.5",
  "hf": {
    "trust_remote_code": true
  },
  "device": {
    "device": "auto",      // auto | cuda | mps | cpu
    "dtype": "auto"        // auto | float32 | float16 | bfloat16
  },
  "train": {
    "batch_size": 8,
    "epochs": 15,
    "max_length": 128,
    "grad_accum_steps": 1,   // 메모리 제약 시 배치 크기를 줄이고 이 값을 늘려 유효 배치 크기 유지
    "pad_to_multiple_of": 8, // MPS/CUDA 성능 최적화 (8 또는 16 권장)
    "fp16": "auto"           // auto | true | false (혼합 정밀도 학습 활성화)
  },
  "optim": {
    "lr": 5e-5
  },
  "data": {
    "pairs_path": "./tools/data/pairs.jsonl"
  },
  "output": {
    "save_dir": "./outputs",
    "save_name": "embed-ft"
  }
}
```

**메모리 관리 팁 (특히 MPS 환경):**

*   **`device.dtype`:** `float16` 또는 `bfloat16` (CUDA만 해당)을 사용하면 메모리 사용량을 절반으로 줄일 수 있습니다. `auto`로 설정하면 시스템이 자동으로 최적의 타입을 선택합니다.
*   **`train.batch_size`:** 가장 먼저 줄여야 할 값입니다. 메모리 부족 시 이 값을 줄이세요.
*   **`train.grad_accum_steps`:** `batch_size`를 줄인 만큼 이 값을 늘려주면, 실제 학습에 사용되는 유효 배치 크기(effective batch size = `batch_size` * `grad_accum_steps`)는 유지하면서 메모리 사용량을 줄일 수 있습니다.
*   **`train.max_length`:** 입력 시퀀스 길이를 줄이면 메모리 사용량이 크게 감소합니다.
*   **`train.pad_to_multiple_of`:** 8 또는 16과 같은 값으로 설정하면 GPU/MPS의 텐서 코어(Tensor Cores) 활용을 최적화하여 성능을 향상시키고 메모리 효율을 높일 수 있습니다.
*   **`train.fp16`:** `true`로 설정하면 혼합 정밀도 학습을 활성화하여 메모리 사용량을 줄이고 학습 속도를 높일 수 있습니다. `auto`로 설정하면 `device.dtype`이 `float16` 또는 `bfloat16`일 때 자동으로 활성화됩니다.

### 4. 학습 실행

다음 명령어로 학습을 시작합니다.

```bash
python -m embed_trainer.train --config config.json
# Qwen 모델 학습 예시:
# python -m embed_trainer.train --config config.qwen.json
```

학습이 완료되면 `output.save_dir`에 지정된 경로에 파인튜닝된 모델이 저장됩니다.

## 프로젝트 구조

- `train.py`: 메인 실행 스크립트
- `trainer.py`: 핵심 학습 로직을 담은 `Trainer` 클래스
- `model.py`: `SentenceTransformer` 모델을 래핑하는 `EmbeddingModel` 클래스
- `data_utils.py`: 데이터 로딩 및 전처리 유틸리티
- `config.json`: 학습 설정 파일
- `docs/`: 상세 기술 문서 폴더
