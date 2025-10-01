# 임베딩 모델 파인튜닝: 원리 및 `embed_trainer` 아키텍처

이 문서는 임베딩 모델의 파인튜닝, 특히 대조 학습(Contrastive Learning)의 기본 원리를 설명하고, 이 프로젝트(`embed_trainer`)의 코드 아키텍처를 기술합니다.

## 1부: 임베딩 파인튜닝의 원리

### 임베딩이란?

임베딩(Embedding)은 텍스트, 이미지, 오디오와 같은 복잡한 데이터를 저차원의 연속적인 벡터(vector)로 변환하는 과정 또는 그 결과물을 의미합니다. 이 벡터 공간에서는 데이터 간의 의미적 유사성이 벡터 간의 거리(예: 유클리드 거리) 또는 각도(예: 코사인 유사도)로 표현됩니다. 예를 들어, "강아지"와 "개"라는 단어는 매우 가까운 벡터를 갖게 됩니다.

최신 언어 모델(LLM)은 이미 대규모 데이터셋으로 사전 학습된 고품질의 범용 임베딩을 제공합니다. 하지만 특정 도메인(예: 의료, 법률, 게임)이나 특정 과제(예: 정보 검색)에서는 그 성능이 최적이 아닐 수 있습니다.

### 왜 파인튜닝이 필요한가?

범용 임베딩을 특정 도메인에 맞게 **파인튜닝(Fine-tuning)**하면 다음과 같은 이점을 얻을 수 있습니다.

1.  **도메인 특화:** "마나"와 "체력" 같은 게임 용어나, "판례"와 "법규" 같은 법률 용어의 미묘한 의미 차이를 모델이 더 잘 이해하게 됩니다.
2.  **과제 최적화:** 정보 검색(Retrieval) 과제에서 사용자의 질문(Query)과 가장 관련 있는 문서(Document)가 벡터 공간에서 더 가깝게 위치하도록 조정할 수 있습니다.
3.  **성능 향상:** 최종적으로 검색 정확도(Recall, MRR), 분류 정확도 등 과제 성능이 크게 향상됩니다.

### 대조 학습 (Contrastive Learning)

대조 학습은 파인튜닝의 핵심적인 방법론 중 하나입니다. 기본 아이디어는 **"비슷한 것은 가깝게, 다른 것은 멀게"** 만드는 것입니다.

학습 데이터는 보통 세 쌍(triplet) 또는 쌍(pair)으로 구성됩니다.

-   **Anchor:** 기준이 되는 데이터 (예: 특정 게임 아이템에 대한 설명)
-   **Positive:** Anchor와 의미적으로 유사한 데이터 (예: 해당 아이템의 다른 이름 또는 관련 설명)
-   **Negative:** Anchor와 의미적으로 관련 없는 데이터 (예: 전혀 다른 아이템에 대한 설명)

학습 과정에서 모델은 다음을 목표로 파라미터를 업데이트합니다.
-   Anchor와 Positive 벡터 간의 거리는 **최소화** (유사도는 최대화)
-   Anchor와 Negative 벡터 간의 거리는 **최대화** (유사도는 최소화)

#### InfoNCE와 MultipleNegativesRankingLoss

이러한 목표를 달성하기 위해 다양한 손실 함수(Loss Function)가 사용됩니다.

-   **InfoNCE (Noise-Contrastive Estimation):** 대조 학습에서 가장 널리 쓰이는 손실 함수 중 하나입니다. 배치(batch) 내에서 하나의 (Anchor, Positive) 쌍을 정답으로 간주하고, 나머지 모든 데이터(다른 Anchor, Positive 포함)를 네거티브로 간주하여 분류 문제처럼 만듭니다. 즉, Anchor가 수많은 "노이즈"들 사이에서 자신의 Positive를 찾아내도록 학습시킵니다.

-   **MultipleNegativesRankingLoss:** `sentence-transformers` 라이브러리에서 제공하는 매우 효율적이고 강력한 손실 함수입니다. InfoNCE의 원리를 기반으로, 배치 내의 모든 Anchor에 대해 각각의 Positive는 정답으로, 그리고 **배치 내 다른 모든 Anchor의 Positive 샘플들**을 매우 효과적인 "하드 네거티브"로 자동 활용합니다. 이는 모델이 미묘한 차이를 학습하도록 강제하여 성능을 크게 향상시킵니다. 우리 `embed_trainer`는 이 손실 함수를 핵심 학습 알고리즘으로 사용합니다.

---

## 2부: `embed_trainer` 아키텍처

`embed_trainer`는 `sentence-transformers` 라이브러리를 기반으로 위에서 설명한 파인튜닝 원리를 구현한 프로젝트입니다. 코드의 복잡성을 줄이고, 명확성과 유지보수성을 높이기 위해 다음과 같이 모듈화된 구조를 가집니다.

```
embed_trainer/
├─── train.py             # 1. 메인 실행 스크립트
├─── trainer.py           # 2. 핵심 학습 로직 (Trainer 클래스)
├─── model.py             # 3. 모델 로딩 및 래핑 (EmbeddingModel 클래스)
├─── data_utils.py        # 4. 데이터 로딩 및 전처리
├─── config.json          # 5. 모든 설정을 담는 파일
└─── docs/
     └─── fine_tuning_principles.md
```

### 데이터 흐름 및 모듈별 역할

1.  **`train.py` (진입점):**
    -   사용자가 `--config` 인자로 `config.json` 파일의 경로를 전달하여 스크립트를 실행합니다.
    -   `config.json`을 파싱하고, `data_utils.py`의 함수를 호출하여 학습 데이터를 로드합니다.
    -   설정 정보와 학습 데이터를 `Trainer` 클래스에 전달하고, `train()` 메소드를 호출하여 전체 학습 프로세스를 시작합니다.

2.  **`config.json` (설정):**
    -   학습에 필요한 모든 하이퍼파라미터와 경로를 정의하는 중앙 저장소입니다.
    -   `model_id`, `batch_size`, `epochs`, `learning_rate`, 데이터 경로, 출력 경로 등을 포함합니다.

3.  **`data_utils.py` (데이터 파이프라인):**
    -   `config.json`에 명시된 `pairs_path`를 바탕으로 `pairs.jsonl` 파일을 읽습니다.
    -   각 줄의 `{"anchor": "...", "positive": "..."}` 쌍을 `sentence-transformers`가 요구하는 `InputExample(texts=[anchor, positive])` 객체로 변환하여 리스트를 생성합니다.
    -   이 데이터 리스트는 `Trainer`에게 전달됩니다.

4.  **`model.py` (모델 캡슐화):**
    -   `EmbeddingModel` 클래스는 `sentence-transformers`의 `SentenceTransformer` 모델을 감싸는 래퍼(wrapper)입니다.
    -   `config.json`의 `model_id`를 받아 Hugging Face Hub에서 사전 학습된 모델을 로드합니다.
    -   모델의 풀링(pooling) 방식 설정 등 모델과 관련된 복잡한 내부 구현을 캡슐화하여, `Trainer`가 일관된 인터페이스로 모델을 사용할 수 있도록 합니다.

5.  **`trainer.py` (핵심 학습 엔진):**
    -   `Trainer` 클래스는 학습의 모든 과정을 총괄합니다.
    -   **초기화 (`__init__`):**
        -   설정(`config`)과 데이터(`train_samples`)를 받습니다.
        -   `EmbeddingModel`을 인스턴스화하여 모델을 준비합니다.
        -   `sentence-transformers`의 `DataLoader`를 생성합니다.
        -   `MultipleNegativesRankingLoss` 손실 함수를 준비합니다.
    -   **학습 (`train`):**
        -   `sentence-transformers`가 제공하는 고수준의 `model.fit()` 메소드를 호출하여 학습을 진행합니다.
        -   `fit()` 메소드는 내부적으로 다음과 같은 작업을 자동으로 처리합니다.
            -   에포크(epoch) 및 스텝(step) 기반 반복
            -   배치(batch) 단위로 데이터 공급
            -   모델을 통해 임베딩 계산
            -   `MultipleNegativesRankingLoss`를 이용한 손실(loss) 계산
            -   역전파(backpropagation) 및 옵티마이저(optimizer)를 통한 모델 파라미터 업데이트
            -   학습 진행률 표시, 체크포인트 저장 등
        -   이러한 추상화 덕분에, `trainer.py`의 코드는 매우 간결하고 직관적입니다.

### 요약

개선된 `embed_trainer`는 **관심사의 분리(Separation of Concerns)** 원칙에 따라 명확하게 역할을 나눈 모듈들로 구성됩니다. `sentence-transformers`의 강력하고 표준화된 기능을 적극 활용하여, 복잡했던 학습 로직을 대폭 단순화하고 안정성과 성능을 동시에 확보했습니다. 이를 통해 사용자는 `config.json` 파일 수정만으로 다양한 모델과 데이터셋에 대한 파인튜닝을 손쉽게 수행할 수 있습니다.
