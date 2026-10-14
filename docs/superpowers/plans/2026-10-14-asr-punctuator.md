# ASR punctuator (#26) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `ASRPunctuator`，用 Qwen3-ASR 一邊聽音檔一邊強制解碼下標點，並接上 `example.py`。

**Architecture:** 把文字 LLM 的強制解碼抽成 `ForcedPunctuator`，只搬不改。
`ASRPunctuator` 繼承它，只換掉 `forward`：把 `<|audio_pad|>` 的 embedding 換成音訊向量。
模型用 transformers 5.3 的類別自己組裝，不依賴 `qwen-asr` 套件。

**Tech Stack:** Python 3.10+、torch、transformers 5.3、soundfile、soxr、pytest。

**Spec:** `docs/superpowers/specs/2026-10-14-asr-punctuator-design.md`。需求以 #26 為準。

## Global Constraints

- worktree：`/volume/tts-data/project/willy/LLM-Punctuator/.claude/worktrees/feature-GH-26-punctuate-text-with-an-asr`，以下簡稱 `$WT`。
  git 一律 `git -C $WT`，其他指令一律 `cd $WT && ...`。
- Python：用主 checkout 的 `.venv`，以下簡稱 `$PY` ＝ `/volume/tts-data/project/willy/LLM-Punctuator/.venv/bin/python`。
- 測試：`cd $WT && $PY -m pytest tests -q -p no:cacheprovider`。
- 跑測試時設 `HF_HUB_OFFLINE=1`。tokenizer 已在 `~/.cache/huggingface/hub`，連網會卡住。
- lint：`cd $WT && bin/lint.sh`。
- pyright 一定要過：`cd $WT && timeout 300 uvx --from 'pyright[nodejs]' pyright --pythonpath $PY llm_punctuator scripts example.py tests`。
  沒有 pre-commit hook，每次 commit 前手動跑 lint 與 pyright。
- commit：`git -C $WT -c user.name=willywsm1013 -c user.email=willywsm1013@gmail.com commit`。
  訊息用 Conventional Commits，經 `generate-commit-message` skill 產生。不准 `git add -A`，不准 `--no-verify`。
- 文字 LLM 的模型輸入與對外介面不變。
- 支援的模型名稱：`Qwen/Qwen3-ASR-1.7B`、`Qwen/Qwen3-ASR-0.6B`。
- 取樣率：16000。重取樣：`soxr.resample(..., quality="HQ")`。
- 語言對應表：spec 2.2 的 30 種，逐字照抄。
- GPU：跑之前看 `nvidia-smi`。有別人的 process 或使用率不是 0 就停，回報等決定。
- 大檔寫到 `/volume`。0.6B 下載到 `/volume/tts-data/project/willy/LLM-Punctuator/exp/hf-cache`。

## Plan 自己決定、spec 沒定的事

1. `language` 不是 zh/en 又沒給標點時，ASR 用自己的訊息，不用 `default_punctuations` 的訊息。那個訊息說語言不支援，對 ASR 是錯的。見 Task 4。
2. 音訊向量由第一次 `forward` 用掉，用完就清空。這是 spec 1.3 第 5 點「結束後清掉」的做法。見 Task 3。
3. #26 只定了 `TypeError` 的訊息。其餘錯誤訊息的字面由本 plan 定。見 Task 4 與 Task 5 的表。
4. `numpy` 列進依賴，因為程式直接 import 它。`soundfile`、`soxr` 裝進主 checkout 共用的 `.venv`。見 Task 2。
5. ASR 模型放在 `cuda`，沒有 GPU 時放 `cpu`。dtype 取 config 的 `thinker_config.dtype`，目前是 bfloat16。見 Task 3。
6. `tests/test_punctuator.py` 的 #24 zh 案例抽成 `ZH_UNIT_CASES`，讓 ASR 測試直接引用。案例只有一份。見 Task 3。
7. README 不改。新參數由 `example.py --help` 說明。

## Review Focus

1. `audios` 第二個路徑不存在：第一篇也不該先跑。預期在讀任何音檔前就拋 `FileNotFoundError`。測試在 Task 4。
2. 文字 LLM 沒給 `-c`：`-c` 預設改成 `None` 後，仍要用 50。測試在 Task 5。
3. 立體聲音檔：預期兩個聲道取平均，不是只取一個聲道。測試在 Task 2。
4. 整數尺度的陣列，例如 int16 轉成 float 沒除 32768：預期縮放到峰值 1。測試在 Task 2。
5. `context` 裡有 `{text}` 以外的大括號：預期原樣保留，不報 `KeyError`。測試在 Task 3。

---

### Task 1: 抽出 `ForcedPunctuator`

**Files:**
- Modify: `llm_punctuator/punctuator.py`

**Interfaces:**
- Produces，在 `llm_punctuator/punctuator.py`：
  - `class ForcedPunctuator`：沒有 `__init__`。子類別要設 `self.tokenizer`、`self.model`、`self.device`。
  - 原封不動搬進來：`split_segments`、`encode_text`、`decode`、`greedy_search`。
  - `@staticmethod default_punctuations(language: str) -> str`：`zh` 回 `ZH_PUNCTUATIONS`，`en` 回 `EN_PUNCTUATIONS`。
    其他照現有程式碼拋 `ValueError(f"Language {language} is not supported. Supported languages: zh, en")`。
  - `punctuation_token_ids(self, punctuations: str) -> list[int]`：現有 183–194 行照搬。
  - `forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, cache_position: torch.Tensor, past_key_values: DynamicCache) -> torch.Tensor`：
    回傳 logits。內容是現有 456–463 行的 `self.model(...)` 呼叫，含 `use_cache=True`。
  - `greedy_search` 改呼叫 `self.forward(...)`，參數名稱照原本的 keyword。
  - `class TransformersLLMPunctuator(ForcedPunctuator)`：`add_punctuation` 改呼叫上面兩個小方法。其餘不動。

- [ ] **Step 1: 確認基準**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests -q -p no:cacheprovider`
Expected: 全過。記下通過的數量。

- [ ] **Step 2: 搬移與抽方法**

照 Interfaces 做。`ChunkJob`、`ChunkLoop`、`ALNUM_RUN`、`ZH_UNIT`、`GreedyRow` 留在原處。
`chunk_segments`、`remove_punctuation`、`run_loops`、`punctuate_chunks`、`apply_chat_template` 留在 `TransformersLLMPunctuator`。

- [ ] **Step 3: 跑全部測試**

Run: 同 Step 1。
Expected: 全過，數量跟 Step 1 相同。這個 task 不加新測試。行為由既有的 greedy_search 與 #24 測試保護。

- [ ] **Step 4: lint、pyright、commit**

```bash
git -C $WT add llm_punctuator/punctuator.py
```
訊息類型 `refactor(punctuator)`。

---

### Task 2: 音檔前處理 `load_audio`

**Files:**
- Modify: `pyproject.toml`、`uv.lock`
- Create: `llm_punctuator/asr_punctuator.py`
- Test: `tests/test_asr_punctuator.py`

**Interfaces:**
- Produces，在 `llm_punctuator/asr_punctuator.py`：
  - `SAMPLE_RATE = 16000`
  - `load_audio(audio: str | tuple[np.ndarray, int]) -> np.ndarray`：回傳一維 float32、16 kHz 的波形。
    假設輸入已經檢查過：路徑存在，陣列是一維。檢查在 Task 4。
    步驟照 spec 2.3：`soundfile.read(path, dtype="float32")`；二維時 `mean(axis=1)`；取樣率不同時 soxr HQ；峰值大於 1 時除以峰值，再 clip 到 [-1, 1]。

- [ ] **Step 1: 加依賴**

```bash
cd $WT && uv add --no-sync soundfile soxr numpy
uv pip install --python $PY soundfile soxr
```
`numpy` 是直接 import，所以列進依賴。`--no-sync` 讓 uv 不在 worktree 另建 `.venv`。
第二行裝進主 checkout 共用的 `.venv`，只新增套件。

- [ ] **Step 2: 寫失敗的測試**

```python
"""Tests for ASRPunctuator: audio input, prompt, and forced decoding."""

from pathlib import Path

import numpy as np
import pytest
import soundfile

from llm_punctuator.asr_punctuator import SAMPLE_RATE, load_audio


def tone(seconds: float, sample_rate: int, peak: float = 0.5) -> np.ndarray:
    """A 440 Hz sine as float32."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (peak * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_path_and_array_load_the_same(tmp_path: Path) -> None:
    """A file and the same samples as (array, 16000) give the same waveform."""
    signal = tone(1.0, SAMPLE_RATE)
    path = tmp_path / "tone.wav"
    soundfile.write(path, signal, SAMPLE_RATE, subtype="FLOAT")

    np.testing.assert_array_equal(load_audio(str(path)), load_audio((signal, SAMPLE_RATE)))


def test_48k_audio_is_resampled_to_16k() -> None:
    """One second at 48 kHz becomes 16000 samples."""
    assert load_audio((tone(1.0, 48000), 48000)).shape == (SAMPLE_RATE,)


def test_stereo_file_is_averaged_to_mono(tmp_path: Path) -> None:
    """Each sample is the mean of the two channels."""
    left, right = tone(1.0, SAMPLE_RATE, 0.2), tone(1.0, SAMPLE_RATE, 0.6)
    path = tmp_path / "stereo.wav"
    soundfile.write(path, np.stack([left, right], axis=1), SAMPLE_RATE, subtype="FLOAT")

    np.testing.assert_allclose(load_audio(str(path)), (left + right) / 2, atol=1e-7)


def test_peak_above_one_is_scaled_to_one() -> None:
    """An int-scaled array is divided by its peak."""
    signal = tone(1.0, SAMPLE_RATE, peak=0.5)

    np.testing.assert_allclose(load_audio((signal * 4, SAMPLE_RATE)), signal * 2, atol=1e-7)
```

- [ ] **Step 3: 確認失敗**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests/test_asr_punctuator.py -q -p no:cacheprovider`
Expected: FAIL，`ModuleNotFoundError: llm_punctuator.asr_punctuator`。

- [ ] **Step 4: 實作 `load_audio`**

模組 docstring 一行。不要在這個 task 放 `ASRPunctuator`。

- [ ] **Step 5: 確認通過**

Run: 同 Step 3。Expected: 4 passed。

- [ ] **Step 6: 跟 librosa 對照，不進 repo**

在 `exp/envs/asr` 的 python 跑：同一個 1 秒 48 kHz 訊號，`soxr.resample(x, 48000, 16000, quality="HQ")` 跟 `librosa.resample(x, orig_sr=48000, target_sr=16000)` 逐點比。
先印兩個套件的 `__version__` 與 librosa 的預設 `res_type`。
Expected: 最大絕對差為 0。不是 0 就停，回報差值，不要 commit。
結果寫到 `exp/runs/20261014-gh26/resample.txt`。

- [ ] **Step 7: lint、pyright、commit**

```bash
git -C $WT add pyproject.toml uv.lock llm_punctuator/asr_punctuator.py tests/test_asr_punctuator.py
```
訊息類型 `feat(asr)`。

---

### Task 3: `ASRPunctuator` 的載入、prompt 與強制解碼

**Files:**
- Modify: `llm_punctuator/asr_punctuator.py`、`llm_punctuator/__init__.py`、`tests/test_punctuator.py`
- Test: `tests/test_asr_punctuator.py`

**Interfaces:**
- Consumes：Task 1 的 `ForcedPunctuator`，Task 2 的 `load_audio`、`SAMPLE_RATE`。
- Produces，在 `llm_punctuator/asr_punctuator.py`：
  - `ASR_MODELS = ("Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-0.6B")`
  - `QWEN3_ASR_LANGUAGES: dict[str, str]`：spec 2.2 的表，標籤對名稱。
  - `build_prompt(context: str, text: str, n_audio: int, language: str) -> str`：
    `context.replace("{text}", text)` 後，填進 spec 1.3 的 prompt。`language` 是標籤，經 `QWEN3_ASR_LANGUAGES` 轉成名稱。
  - `load_model(model_name: str) -> tuple[Qwen3OmniMoeAudioEncoder, Qwen3ForCausalLM, int]`：
    回傳 audio encoder、語言模型、`audio_token_id`。做法照 `exp/scripts/native_asr.py`：
    `snapshot_download`，讀 `config.json` 的 `thinker_config`，`strict=True` 載權重。
    dtype 用 `thinker_config["dtype"]`。device 是 `cuda`，沒有 GPU 時是 `cpu`。
  - `class ASRPunctuator(ForcedPunctuator)`：
    - `__init__(self, model_name: str) -> None`：名稱不在 `ASR_MODELS` 就先拋 `ValueError`，再做任何載入。
      訊息：`f"Unsupported model {model_name!r}. Supported: {', '.join(ASR_MODELS)}"`。
      之後設 `audio_encoder`、`model`、`audio_token_id`（來自 `load_model`），
      `tokenizer`（`AutoTokenizer`），`feature_extractor`（`WhisperFeatureExtractor`），
      `device = self.model.device`，`pending_audio = torch.empty(0)`。
    - `encode_audio(self, wav: np.ndarray) -> torch.Tensor`：回傳 `(N, hidden)`。
      特徵抽取參數照原型：`sampling_rate=SAMPLE_RATE, padding=True, truncation=False, return_attention_mask=True, return_tensors="pt"`。
      依 attention mask 截掉 padding 的 frame，轉成 `self.model.dtype` 後呼叫
      `self.audio_encoder(features, feature_lens=...)`，取 `.last_hidden_state`。
    - `forward(...)`：簽名同 Task 1。把 `input_ids` 經 `self.model.get_input_embeddings()` 轉成 embedding。
      `<|audio_pad|>` 的個數必須等於 `len(self.pending_audio)`，不等就拋
      `ValueError(f"{n_pads} audio pads but {n_vectors} audio vectors")`。
      換掉之後把 `pending_audio` 設成空的。第一次 forward 用完向量，之後的 forward 沒有 pad，個數都是 0。
      再呼叫 `self.model(inputs_embeds=..., attention_mask=..., position_ids=..., cache_position=..., past_key_values=..., use_cache=True).logits`。
    - `add_punctuation(self, texts, audios, language="zh", context="{text}", punctuations=None, k=-1) -> list[str]`：
      簽名照 spec 1.2，加 `@torch.no_grad()`。這個 task 只做資料流，檢查在 Task 4。
      逐篇：`load_audio` → `encode_audio` 存到 `pending_audio` → `build_prompt` → tokenizer 編碼 prompt，`add_special_tokens=False`
      → `split_segments(text, language)` → `CustomLogitsProcessor(text_tokens + [eos], punctuation_token_ids, has_prev_input=False, unit_ends=accumulate(segment 長度))`
      → `greedy_search([(prompt_ids, rule)], k)` → `decode`。
  - `llm_punctuator/__init__.py` 匯出 `ASRPunctuator`。
- Produces，在 `tests/test_punctuator.py`：
  - `ZH_UNIT_CASES: list`：`test_zh_marks_never_split_a_unit` 目前的參數清單，改成 `pytest.param(text, chunk_size, expected, id=...)`。
    測試本身改用它，內容不變。

- [ ] **Step 1: 確認 ASR tokenizer 的前提**

已知：`Qwen/Qwen3-ASR-1.7B` 的 tokenizer 對 #24 的 15 條 zh 輸入，編碼跟 `Qwen/Qwen3-1.7B` 相同。
`eos_token_id` 是 151645，就是 `<|im_end|>`。`<|audio_pad|>` 是 151676，跟 config 的 `audio_token_id` 相同。
這一步不用重跑。只有 Step 7 的結果跟 #24 不同時，才回頭查這裡。

- [ ] **Step 2: 抽出 `ZH_UNIT_CASES`**

改 `tests/test_punctuator.py`。跑 `tests/test_punctuator.py`，通過數不變。

- [ ] **Step 3: 寫失敗的測試**

加進 `tests/test_asr_punctuator.py`。import 照需要補上。

```python
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
HIDDEN = 8
N_AUDIO = 3
K_VALUES = pytest.mark.parametrize("k", [-1, 1], ids=["k_all", "k_1"])


class FakeAudioEncoder:
    """Turns any features into N_AUDIO zero vectors, and counts its calls."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls = 0

    def __call__(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> SimpleNamespace:
        """Return N_AUDIO vectors of size HIDDEN."""
        self.calls += 1
        return SimpleNamespace(last_hidden_state=torch.zeros(N_AUDIO, HIDDEN))


class ConstantScoreLM:
    """Fake LM fed by embeddings that gives every position the same scores."""

    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self, scores: torch.Tensor) -> None:
        """Keep the scores of one position, shaped (1, vocab_size)."""
        self.scores = scores
        self.embedding = torch.nn.Embedding(scores.shape[1], HIDDEN)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        """Return the token embedding."""
        return self.embedding

    def __call__(self, inputs_embeds: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        """Return the scores at every fed position."""
        return SimpleNamespace(logits=self.scores.expand(*inputs_embeds.shape[:2], -1))


@pytest.fixture(scope="module")
def asr_punctuator() -> ASRPunctuator:
    """Real Qwen3-ASR tokenizer with a fake model that picks the first mark whenever one is allowed."""
    tokenizer = AutoTokenizer.from_pretrained(ASR_MODEL)
    scores = torch.ones(1, len(tokenizer))
    scores[0, tokenizer.eos_token_id] = 0.0
    for punctuations in (ZH_PUNCTUATIONS, EN_PUNCTUATIONS):
        scores[0, tokenizer.encode(punctuations[0])[-1]] = 2.0
    with patch("llm_punctuator.asr_punctuator.load_model") as load_model:
        load_model.return_value = (FakeAudioEncoder(), ConstantScoreLM(scores), AUDIO_TOKEN_ID)
        return ASRPunctuator(ASR_MODEL)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    """One second of tone at 16 kHz."""
    path = tmp_path / "tone.wav"
    soundfile.write(path, tone(1.0, SAMPLE_RATE), SAMPLE_RATE, subtype="FLOAT")
    return path


@K_VALUES
@pytest.mark.parametrize(
    ("text", "expected"),
    [pytest.param(*case.values[::2], id=case.id) for case in ZH_UNIT_CASES if case.values[1] == 50],
)
def test_zh_marks_never_split_a_unit(
    asr_punctuator: ASRPunctuator, wav_path: Path, text: str, expected: str, k: int
) -> None:
    """The #24 zh cases without a chunk cut hold for the ASR punctuator."""
    [result] = asr_punctuator.add_punctuation([text], [str(wav_path)], language="zh", k=k)

    assert result == expected


def test_each_text_gets_its_own_result_in_input_order(
    asr_punctuator: ASRPunctuator, wav_path: Path
) -> None:
    """Two texts come back in the order given."""
    results = asr_punctuator.add_punctuation(["鼓勵長輩", "中共2017年"], [str(wav_path)] * 2)

    assert results == ["鼓，勵，長，輩，", "中共，2017，年，"]


def test_build_prompt_fills_context_and_keeps_other_braces() -> None:
    """Only {text} is replaced, the pads match n_audio, and the tag becomes the model's name."""
    prompt = build_prompt("主題：{text}{x}", "今天", 2, "yue")

    assert prompt == (
        "<|im_start|>system\n主題：今天{x}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\nlanguage Cantonese<asr_text>"
    )


@pytest.fixture
def tiny_asr_punctuator(asr_punctuator: ASRPunctuator) -> ASRPunctuator:
    """A randomly initialized real Qwen3ForCausalLM behind the ASR forward."""
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(asr_punctuator.tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    punctuator = ASRPunctuator.__new__(ASRPunctuator)
    punctuator.model = Qwen3ForCausalLM(config).eval()
    punctuator.device, punctuator.audio_token_id = punctuator.model.device, AUDIO_TOKEN_ID
    return punctuator


def forward_all(punctuator: ASRPunctuator, ids: list[int]) -> torch.Tensor:
    """Run the ASR forward on ids as one row with a fresh cache."""
    input_ids = torch.tensor([ids])
    positions = torch.arange(len(ids))
    return punctuator.forward(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=positions[None],
        cache_position=positions,
        past_key_values=DynamicCache(),
    )


def test_forward_feeds_audio_vectors_at_audio_pads(tiny_asr_punctuator: ASRPunctuator) -> None:
    """The logits equal the real model fed the audio vectors in place of the pads."""
    audio = torch.randn(2, 32)
    tiny_asr_punctuator.pending_audio = audio
    embed = tiny_asr_punctuator.model.get_input_embeddings()
    before, after = torch.tensor([151644]), torch.tensor([151670, 151645])

    with torch.no_grad():
        logits = forward_all(tiny_asr_punctuator, [151644, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 151670, 151645])
        expected = tiny_asr_punctuator.model(
            inputs_embeds=torch.cat([embed(before), audio, embed(after)])[None]
        ).logits

    torch.testing.assert_close(logits, expected)


def test_forward_rejects_pad_count_that_differs_from_audio_vectors(
    tiny_asr_punctuator: ASRPunctuator,
) -> None:
    """Two pads with three vectors raises ValueError."""
    tiny_asr_punctuator.pending_audio = torch.randn(3, 32)

    with pytest.raises(ValueError, match="2 audio pads but 3 audio vectors"):
        forward_all(tiny_asr_punctuator, [151644, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 151645])
```

`k=1` 的 zh 測試會跑多次 forward。如果第一次 forward 沒把 `pending_audio` 清空，第二次就會拋錯。所以消耗語意由它們保護。

- [ ] **Step 4: 確認失敗**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests/test_asr_punctuator.py -q -p no:cacheprovider`
Expected: collection 失敗，`ImportError: cannot import name 'ASRPunctuator'`。

- [ ] **Step 5: 實作**

照 Interfaces。`load_model` 的權重前綴：
- audio encoder：`thinker.audio_tower.`
- 語言模型：`thinker.model.` 改成 `model.`，`lm_head.weight` 來自 `thinker.lm_head.weight`
- config：`Qwen3OmniMoeAudioEncoderConfig(**audio_config)` 與 `Qwen3Config(**text_config)`，都先拿掉 `model_type`。attention 用 `sdpa`。

- [ ] **Step 6: 確認通過**

Run: 同 Step 4。Expected: 全過。

- [ ] **Step 7: 對照 qwen_asr 的 prompt，不進 repo**

在 `exp/envs/asr` 用 qwen_asr 的 `_build_text_prompt`，`context` 有內容與空字串各一次，`force_language="Chinese"`。
跟 `build_prompt` 加上 N 個 `<|audio_pad|>` 的結果比。qwen_asr 的 prompt 只有一個 `<|audio_pad|>`，由 processor 再展開成 N 個。比之前先把它展開成相同的 N。
Expected: 兩組都逐字相同。不同就停，回報兩邊的字串。結果寫到 `exp/runs/20261014-gh26/prompt.txt`。

- [ ] **Step 8: lint、pyright、全部測試、commit**

```bash
git -C $WT add llm_punctuator/asr_punctuator.py llm_punctuator/__init__.py tests/test_asr_punctuator.py tests/test_punctuator.py
```
訊息類型 `feat(asr)`。

---

### Task 4: 輸入檢查

**Files:**
- Modify: `llm_punctuator/asr_punctuator.py`
- Test: `tests/test_asr_punctuator.py`

**Interfaces:**
- Consumes：Task 3 的 `ASRPunctuator`、`QWEN3_ASR_LANGUAGES`。
- Produces：`add_punctuation` 開頭依 spec 2.1 的順序檢查，全部在讀第一個音檔之前。訊息：

| 順序 | 情況 | 例外與訊息 |
|---|---|---|
| 1 | `texts` 是 `str` | `TypeError("texts must be a list of str, got str. Wrap a single text as [text].")` |
| 2 | `k` 是 0 或小於 -1 | `ValueError(f"k must be -1 or a positive integer, got {k}")` |
| 3 | 長度不同 | `ValueError(f"texts and audios must have the same length, got {len(texts)} and {len(audios)}")` |
| 4 | 標籤不在表裡 | `ValueError(f"Unsupported language {language!r}. Supported: {', '.join(QWEN3_ASR_LANGUAGES)}")` |
| 5 | 不是 zh/en 又沒給標點 | `ValueError(f"No default punctuations for language {language!r}. Pass punctuations.")` |
| 6 | 陣列不是一維 | `ValueError(f"audios[{i}] must be a 1-D array, got shape {array.shape}")` |
| 7 | 路徑不存在 | `FileNotFoundError(f"audios[{i}]: no such file: {path}")` |

第 1、2 條的訊息跟文字 LLM 現有的相同。第 5 條不用 `default_punctuations` 的訊息，因為那個訊息說語言不支援，對 ASR 是錯的。

- [ ] **Step 1: 寫失敗的測試**

```python
@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"texts": "今天"}, TypeError, re.escape("texts must be a list of str, got str. Wrap a single text as [text].")),
        ({"k": 0}, ValueError, "k must be -1 or a positive integer, got 0"),
        ({"k": -2}, ValueError, "k must be -1 or a positive integer, got -2"),
        ({"texts": ["今天", "明天"]}, ValueError, "texts and audios must have the same length, got 2 and 1"),
        ({"language": "zh-TW"}, ValueError, "Unsupported language 'zh-TW'. Supported: zh, en, yue, "),
        ({"language": "ko"}, ValueError, "No default punctuations for language 'ko'"),
        (
            {"audios": [(np.zeros((2, SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)]},
            ValueError,
            re.escape("audios[0] must be a 1-D array, got shape (2, 16000)"),
        ),
        ({"audios": ["/no/such.wav"]}, FileNotFoundError, "audios\\[0\\]: no such file: /no/such.wav"),
    ],
    ids=["str_texts", "k_zero", "k_below_minus_one", "length_mismatch", "regional_tag", "no_default_marks", "2d_array", "missing_file"],
)
def test_add_punctuation_rejects_bad_input(
    asr_punctuator: ASRPunctuator,
    wav_path: Path,
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    """Each failure in #26 raises its error with a message that names the problem."""
    call = {"texts": ["今天"], "audios": [str(wav_path)], **kwargs}

    with pytest.raises(error, match=message):
        asr_punctuator.add_punctuation(**call)


def test_missing_second_file_raises_before_any_audio_is_encoded(
    asr_punctuator: ASRPunctuator, wav_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All paths are checked before the first text runs."""
    encoder = FakeAudioEncoder()
    monkeypatch.setattr(asr_punctuator, "audio_encoder", encoder)

    with pytest.raises(FileNotFoundError, match="audios\\[1\\]"):
        asr_punctuator.add_punctuation(["今天", "明天"], [str(wav_path), "/no/such.wav"])

    assert encoder.calls == 0


def test_unsupported_model_raises_before_loading() -> None:
    """A name outside ASR_MODELS raises ValueError listing the supported names, and loads nothing."""
    with (
        patch("llm_punctuator.asr_punctuator.load_model") as load_model,
        patch("llm_punctuator.asr_punctuator.AutoTokenizer") as tokenizer_cls,
        pytest.raises(ValueError, match="Supported: Qwen/Qwen3-ASR-1.7B, Qwen/Qwen3-ASR-0.6B"),
    ):
        ASRPunctuator("Qwen/Qwen3-1.7B")

    assert not load_model.called
    assert not tokenizer_cls.from_pretrained.called
```

`test_unsupported_model_raises_before_loading` 在 Task 3 實作後可能已經通過。它還是放在這個 task，因為它屬於 #26 的錯誤清單。

- [ ] **Step 2: 確認失敗**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests/test_asr_punctuator.py -q -p no:cacheprovider -k "bad_input or missing_second"`
Expected: 多數 FAIL，例外類型或訊息不符。

- [ ] **Step 3: 實作檢查**

照表的順序放在 `add_punctuation` 開頭。用 `Path(path).is_file()` 檢查路徑。

- [ ] **Step 4: 確認通過**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests/test_asr_punctuator.py -q -p no:cacheprovider`
Expected: 全過。

- [ ] **Step 5: lint、pyright、commit**

```bash
git -C $WT add llm_punctuator/asr_punctuator.py tests/test_asr_punctuator.py
```
訊息類型 `feat(asr)`。

---

### Task 5: `example.py` 的 ASR 路徑

**Files:**
- Modify: `example.py`
- Test: `tests/test_example.py`

**Interfaces:**
- Consumes：Task 3 的 `ASR_MODELS`、`ASRPunctuator`，從 `llm_punctuator.asr_punctuator` import。
- Produces：
  - `--audio PATH`、`--context TEMPLATE`，兩個預設都是 `None`。
  - `-c` 預設改成 `None`。help 寫明文字 LLM 沒給時用 50，ASR 不能用。
  - `get_args` 在 parse 之後檢查。`is_asr = args.model_name_or_path in ASR_MODELS`。依序：

| 情況 | `parser.error` 的訊息 |
|---|---|
| ASR，給了 `--list` | `--list does not support an ASR model` |
| ASR，沒給 `--audio` | `--audio is required with an ASR model` |
| ASR，給了 `-c` | `--chunk_size does not apply to an ASR model` |
| 文字 LLM，給了 `--audio` 或 `--context` | `--audio and --context need an ASR model` |

  - ASR 時：`ASRPunctuator(model)`，呼叫 `add_punctuation(clean_texts, [args.audio], language=args.language, **options)`。
    `options` 有 `k` 與 `context`，各自只在使用者有給時放進去。
  - 文字 LLM 時：照舊，`chunk_size=50 if args.chunk_size is None else args.chunk_size`。

- [ ] **Step 1: 寫失敗的測試**

`run_example` 加一個參數 `asr_punctuator: type = FakeASRPunctuator`，並 patch `llm_punctuator.asr_punctuator.ASRPunctuator`。

```python
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"


class FakeASRPunctuator:
    """Fake ASR punctuator that ends each text with a full stop and keeps the last call."""

    last_call: dict[str, object] = {}

    def __init__(self, model_name: str) -> None:
        """Skip loading a model."""

    def add_punctuation(self, texts: list[str], audios: list[str], **kwargs: object) -> list[str]:
        """Append a full stop to each text."""
        FakeASRPunctuator.last_call = {"texts": texts, "audios": audios, **kwargs}
        return [f"{text}。" for text in texts]


def test_asr_model_punctuates_cleaned_text_with_its_audio(capsys: pytest.CaptureFixture[str]) -> None:
    """An ASR -m gets the cleaned text, the audio path, and the language."""
    run_example(["-m", ASR_MODEL, "--audio", "a.wav", "--text", "今天 天氣"])

    assert capsys.readouterr().out == "今天天氣。\n"
    assert FakeASRPunctuator.last_call == {"texts": ["今天天氣"], "audios": ["a.wav"], "language": "zh"}


def test_asr_context_reaches_add_punctuation(capsys: pytest.CaptureFixture[str]) -> None:
    """--context is passed through as context."""
    run_example(["-m", ASR_MODEL, "--audio", "a.wav", "--text", "今天", "--context", "主題：{text}"])

    assert FakeASRPunctuator.last_call["context"] == "主題：{text}"


def test_text_llm_without_chunk_size_still_uses_50(capsys: pytest.CaptureFixture[str]) -> None:
    """-c defaults to None in argparse, but the text LLM still gets 50."""
    run_example(["--text", "今天"])

    assert FullStopPunctuator.last_kwargs["chunk_size"] == 50


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["-m", ASR_MODEL, "--text", "今天"], "--audio is required with an ASR model"),
        (["-m", ASR_MODEL, "--audio", "a.wav", "--text", "今天", "-c", "50"], "--chunk_size does not apply to an ASR model"),
        (["-m", ASR_MODEL, "--audio", "a.wav", "--list", "x.jsonl"], "--list does not support an ASR model"),
        (["--audio", "a.wav", "--text", "今天"], "--audio and --context need an ASR model"),
        (["--context", "{text}", "--text", "今天"], "--audio and --context need an ASR model"),
    ],
    ids=["asr_without_audio", "asr_with_chunk_size", "asr_with_list", "text_llm_with_audio", "text_llm_with_context"],
)
def test_invalid_model_and_option_combination_exits_2(
    capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    """Each combination #26 forbids exits 2 with its message on stderr."""
    with pytest.raises(SystemExit) as exc_info:
        run_example(argv)

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
```

- [ ] **Step 2: 確認失敗**

Run: `cd $WT && HF_HUB_OFFLINE=1 $PY -m pytest tests/test_example.py -q -p no:cacheprovider`
Expected: 新測試 FAIL，argparse 不認得 `--audio` 等。

- [ ] **Step 3: 實作**

照 Interfaces。`-l` 選項、`-b` 與 `preprocess` 不動。

- [ ] **Step 4: 確認通過**

Run: 同 Step 2。Expected: 全過，舊測試也過。

- [ ] **Step 5: lint、pyright、全部測試、commit**

```bash
git -C $WT add example.py tests/test_example.py
```
訊息類型 `feat(example)`。

---

### Task 6: 真模型驗證，用 GPU

結果寫到 `/volume/tts-data/project/willy/LLM-Punctuator/exp/runs/20261014-gh26/`。腳本放 `exp/scripts/gh26_*.py`。exp 不進 repo。
每個 process 先印 `llm_punctuator.__file__`，確認載入的是 `$WT` 的程式碼。

- [ ] **Step 1: 看 GPU**

Run: `nvidia-smi`
Expected: 沒有別人的 process，使用率 0。否則停，回報。

- [ ] **Step 2: 下載 0.6B 並確認結構**

```bash
HF_HUB_CACHE=/volume/tts-data/project/willy/LLM-Punctuator/exp/hf-cache uvx --from huggingface_hub hf download Qwen/Qwen3-ASR-0.6B
```
之後跑 0.6B 都帶這個 `HF_HUB_CACHE`。1.7B 用預設快取。
`load_model` 用 `strict=True`，結構不同會直接報錯。報錯就停，回報缺少或多出的 key。

- [ ] **Step 3: AC 1**

`data/pts-tai` 的 42 筆。文字經過 `example.preprocess(text, "zh")`，`language="zh"`，其他用預設。兩個模型各跑一次。
輸出 `ac1-1.7B.jsonl` 與 `ac1-0.6B.jsonl`，每行 `{"id", "text"}`。
Expected: 每筆拿掉 `ZH_PUNCTUATIONS` 後都等於清理後的原文。回報兩個模型各幾筆相同。

- [ ] **Step 4: AC 2 與 AC 3**

- AC 2：`001.wav` 用路徑跑一次，用 `soundfile.read` 讀成 `(陣列, 取樣率)` 再跑一次。Expected：輸出逐字相同。
- AC 3：用 soxr 把 `001.wav` 轉成 48 kHz，寫成 `001-48k.wav` 再跑。Expected：拿掉標點後等於原文。
- 結果寫到 `ac2-ac3.txt`。

- [ ] **Step 5: AC 5**

```bash
cd $WT && $PY example.py -m Qwen/Qwen3-ASR-1.7B --audio /volume/tts-data/project/willy/LLM-Punctuator/data/pts-tai/001.wav --file /volume/tts-data/project/willy/LLM-Punctuator/data/pts-tai/001.txt
```
Expected: 印出加上標點的文字。完整輸出存到 `ac5.txt`。

- [ ] **Step 6: 跟原型對照，只是參考**

`exp/runs/20261003-v2/native.jsonl` 裡 `cond == "ctx"` 的行，取 `native_cached` 欄。
跟 Step 3 的 1.7B 輸出比。只比拿掉標點與空白後兩邊文字相同的樣本。
回報可比的筆數與逐字相同的筆數，不同的列出 id 與兩邊文字。結果寫到 `parity.txt`。不是 gate。

- [ ] **Step 7: 文字 LLM 回歸**

- main 的快照：`git -C $WT archive main llm_punctuator scripts | tar -x -C <scratchpad>/main-snap`。
- 兩邊各跑 `$PY scripts/run_benchmark.py --k -1 --output-dir <各自的目錄>`，cwd 是各自的樹。
- `diff -r` 兩個輸出目錄。Expected：沒有差異。有差異就停，回報。
- 結果寫到 `regression.txt`。

---

### Task 7: 收尾

- [ ] **Step 1: 全部檢查**

lint、pyright、全部測試都過。確認沒有殘留的舊名稱：`grep -rn "TransformersLLMPunctuator\.greedy_search\|TransformersLLMPunctuator\.split_segments" $WT --include=*.py`，Expected：沒有結果。

- [ ] **Step 2: push**

```bash
git -C $WT push -u origin feature/GH-26-punctuate-text-with-an-asr
```

- [ ] **Step 3: 開 MR**

先 `cd $WT`，再 `Skill("merge-request")`，不給 target。Task 6 的數字寫進 MR 描述。
