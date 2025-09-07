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
python embed_trainer/train.py --config embed_trainer/config.json
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
  - `grad_checkpointing`(옵션, bool): 메모리 절약을 위해 그라디언트 체크포인팅 활성화
  - `num_workers`(옵션, int): DataLoader 워커 수. CPU 토크나이즈 병렬화
  - `pad_to_multiple_of`(옵션, int): 패딩 길이를 8/16 등 배수로 맞춰 가속기 효율 향상
- `optim`: `name`(AdamW 등), `lr`, `weight_decay`, `warmup_ratio`
- `data`: `pairs_path`(옵션), 없으면 자동 탐색
  - `stream`(옵션, bool): JSONL을 스트리밍(라인 단위)으로 로딩하여 메모리 사용 최소화
- `output`: `save_dir`, `save_name`, `save_metrics`

### HF(remote code) 관련 옵션
- `hf.trust_remote_code`: 원격 래퍼 사용 여부. 추론 전용 래퍼일 경우 학습 시 문제가 될 수 있음.
- `hf.local_files_only`: 로컬 캐시만 사용
- `hf.default_task`(옵션): 원격 래퍼가 `default_task` 속성을 지원할 때 설정(예: Jina v3에서 `classification`).
- `hf.loader`(옵션): `"automodel"`(기본) 또는 `"sentence_transformer"`.
  - Jina v3 등 SentenceTransformer 래퍼가 권장되는 모델은 `sentence_transformer`로 설정하세요.
  - 이 경우 `pip install -U sentence-transformers`가 필요합니다.

## 주의/권장
- GPU가 있을 경우 `device=auto`로 CUDA/MPS를 자동 사용
- 너무 큰 배치를 고정하기보다 `grad_accum_steps`로 누적 권장
- 모델/아키텍처 변경 시에도 설정 파일만 교체하여 재사용

### 원격 코드(trust_remote_code)와 Jina 임베딩 관련 메모
- 일부 모델(예: `jinaai/jina-embeddings-v3`)은 원격 코드 래퍼가 추론 전용으로 구성되어 `torch.no_grad()`가 적용되거나 파라미터가 기본적으로 freeze 되어 있을 수 있습니다. 이 경우 `loss.backward()`에서 "does not require grad" 오류가 날 수 있습니다.
- 트레이너는 다음을 자동으로 시도합니다.
  - 모든 파라미터를 `requires_grad=True`로 설정
  - `__call__` 대신 하위 모듈의 `forward`를 직접 호출하여 no-grad 래퍼를 우회 (예: `model.encoder`, `model.roberta`, `model.xlm_roberta` 등)
- 설정 팁:
  - `config.json`의 `hf.trust_remote_code` 값을 명시적으로 조정하세요. 원격 래퍼로 인한 문제가 의심되면 `false`로 두고, 기본 아키텍처 로딩을 시도해보세요.
  - 원격 래퍼가 `default_task`를 요구하는 경우 `hf.default_task`를 설정하세요(예: `"classification"`). 트레이너는 가능한 경우 해당 속성을 모델 또는 하위 모듈에 설정합니다.
  - 캐시 문제로 충돌 시, 동적 모듈 캐시를 비우고 최신 `transformers`를 사용하세요.
    - `rm -rf ~/.cache/huggingface/modules/transformers_modules/jinaai/xlm-roberta-flash-implementation`
    - `pip install -U "transformers>=4.44" "huggingface_hub>=0.24" accelerate`

### 트러블슈팅: loss가 grad를 갖지 않을 때
- 증상: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
- 조치:
  1) `hf.trust_remote_code`를 `false`로 바꾸고 재시도
  2) 위 캐시 삭제/업데이트 수행 후 재시도
  3) 문제가 계속되면 기본 백본 서브모듈이 존재하는지 확인(`encoder`, `roberta`, `xlm_roberta` 등) 후 이 리포의 트레이너가 선택하는지 로그에서 확인
  4) 여전히 실패하면 추론 전용 래퍼일 수 있으니, 학습 가능한 임베딩 백본 모델(e5/bge 등)로 교체 권장
  5) Jina v3를 반드시 쓰고자 한다면 `hf.loader="sentence_transformer"`, `hf.trust_remote_code=true`, `hf.default_task="classification"`를 설정하세요.

### 트러블슈팅: NaN loss / 정확도 정체(=1/batch)
- 증상: 학습 초반부터 `loss=nan`, 정확도 `~1/batch_size`에서 고정
- 원인: fp16/bf16에서 정규화(`normalize`)의 eps 언더플로우 또는 마스킹 평균에서 0으로 나누기
- 조치(트레이너에 반영됨):
  - 풀링/정규화를 float32에서 수행 후 원래 dtype으로 캐스팅하여 수치적 안정성 확보
  - `normalize(..., eps=1e-6)`로 안전한 eps 사용
- 추가 팁:
  - 여전히 NaN이면 `dtype: "float32"`로 변경하여 재시도
  - `temperature`를 0.07→0.1~0.2로 높여 logits 스케일을 낮추면 안정화에 도움이 될 수 있음

### 트러블슈팅: MPS OOM
- 증상: `MPS backend out of memory ... Tried to allocate ...` 메시지와 함께 종료
- 조치:
  - `dtype: "float16"`(기본 auto로 MPS면 fp16) 유지 + SentenceTransformer 경로에서도 모델을 fp16으로 캐스팅하도록 트레이너가 처리합니다.
  - `train.batch_size`를 줄이세요(예: 8 → 2 또는 1) 후 `grad_accum_steps`로 보상
  - `train.max_length`를 줄이세요(예: 128 → 64)
  - `train.grad_checkpointing: true`를 설정해 메모리 사용량을 낮추세요
  - 트레이너는 MPS에서 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`을 기본 설정하여 상한 완화(필요 시)합니다

## 메모리 절약 팁(대용량 데이터)
- `data.stream: true`로 설정하면 `pairs.jsonl`을 메모리에 올리지 않고 라인 단위로 읽습니다.
  - 스트리밍 모드에서는 DataLoader 길이를 알 수 없을 수 있으므로 `train.steps_per_epoch`를 명시하거나, 트레이너가 안전하게 라인 수를 1회 카운트하여 steps를 추정합니다.
- `train.grad_accum_steps`: 배치를 줄이는 대신 누적으로 유효 배치 크기를 맞추세요.
- `train.grad_checkpointing: true`: 피처맵 저장 대신 재계산으로 VRAM 사용을 줄입니다.
- `train.max_length`: 문장 길이를 줄이면 토큰 수가 줄어 메모리 압박이 크게 완화됩니다.
- `train.pad_to_multiple_of: 8`(또는 16): 텐서 코어/벡터화에 유리해 성능/메모리 효율이 개선될 수 있습니다.
- CUDA의 경우 `num_workers>0` + `pin_memory=True`(기본)로 호스트→디바이스 전송 효율이 향상됩니다.

## 테스트
본 프로젝트에는 간단한 학습 검증용 테스트가 포함됩니다.

- 실행:
```
pip install -r embed_trainer/requirements.txt
pip install pytest
pytest -q
```

- 포함된 테스트:
  - `tests/test_training_smoke.py::test_train_end_to_end`: 초소형 HF 모델로 2 스텝 학습 후 저장물/메트릭 확인
  - `tests/test_training_smoke.py::test_streaming_loader`: 스트리밍 DataLoader가 배치 생성 여부 확인

## 변경 이력

## 변경 이력
- v0: 초기 공개. `embedding/qwen`의 구조/로직을 단순화하여 일반화.
- v0.1: no-grad 래퍼를 우회하기 위해 서브모듈 `forward` 직접 호출 및 파라미터 unfreeze 로직 추가. README 트러블슈팅/설정 안내 보강.
- v0.2: `hf.trust_remote_code` 사용자 설정을 엄격히 준수하도록 변경(특정 모델에 대해 강제 활성화 제거). Jina 원격 래퍼 사용 시 발생하던 `loss.backward()`의 no-grad 오류를 회피하기 쉬워짐.
- v0.3: `hf.default_task`(예: `classification`)를 지원하여, 원격 래퍼가 해당 속성을 사용할 때 자동 설정하도록 개선.
- v0.4: `hf.loader=sentence_transformer` 경로 추가. Jina v3 등 ST 기반 모델을 Transformer 모듈 기준으로 학습할 수 있도록 forward 어댑터를 구현.
- v0.5: 메모리 세이프 학습을 위해 `data.stream`, `train.num_workers`, `train.pad_to_multiple_of` 추가. 스트리밍 길이 추정 및 소형 E2E 테스트/스트리밍 테스트 추가.
### 평가 (Evaluation)
학습 결과 디렉터리와 pairs.jsonl로 간단한 검색 성능을 측정할 수 있습니다.

```
python -m embed_trainer.eval --model-dir outputs/embed-ft --pairs tools/data/pairs.jsonl --device auto
```

출력 예:
```
recall@1: 0.6250
recall@5: 1.0000
mrr@10: 0.7500
```
