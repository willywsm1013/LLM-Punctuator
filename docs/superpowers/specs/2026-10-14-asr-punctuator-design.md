# ASR punctuator design (#26)

Status: 第 1、2、3 段已核可，待整份 review。

需求以 #26 為準。本檔只寫票上沒定的設計決定。

## 已定的決定

- 載入方式：native 組裝。用 transformers 5.3 的 `Qwen3OmniMoeAudioEncoder` 與 `Qwen3ForCausalLM`，
  從 safetensors 依前綴載權重。原型在 `exp/scripts/native_asr.py`。
  - qwen_asr 的 `get_rope_index` 永遠產生 3 列相同的位置。3 列相同時 mrope 等於 1-D RoPE。
- ASR 不切 chunk，見 #26。`add_punctuation` 沒有 `chunk_size` 參數，硬給會得到 Python 本身的 `TypeError`。
- 音檔依賴：`soundfile` 讀檔，`soxr` 重取樣。

## 第 1 段：架構與資料流

### 1.1 抽出共用的強制解碼，只搬不改

`punctuator.py` 新增 `ForcedPunctuator`。它沒有 `__init__`，也沒有 `add_punctuation`。

從 `TransformersLLMPunctuator` 原封不動搬過去的方法：
- `split_segments`
- `encode_text`
- `decode`
- `greedy_search`

另外抽兩個小方法，內容照搬 `add_punctuation` 現有的程式碼：
- 預設標點：`zh` 與 `en` 以外報 `ValueError`。
- 標點字元轉 token id。

`greedy_search` 裡 `self.model(...)` 那一行抽成 `forward()`。

文字 LLM 的模型輸入不變，對外介面也不變。

不直接繼承 `TransformersLLMPunctuator` 的原因：ASR 的 `add_punctuation` 簽名不同，不該是 override。

### 1.2 新檔 `llm_punctuator/asr_punctuator.py`

```python
class ASRPunctuator(ForcedPunctuator):
    def __init__(self, model_name: str) -> None: ...
    def add_punctuation(
        self,
        texts: list[str],
        audios: list[str | tuple[np.ndarray, int]],
        language: str = "zh",
        context: str = "{text}",
        punctuations: str | None = None,
        k: int = -1,
    ) -> list[str]: ...
```

- `__init__`：名稱不是 `Qwen/Qwen3-ASR-1.7B` 或 `Qwen/Qwen3-ASR-0.6B` 時報 `ValueError`。
  這個檢查在下載之前做。
- `__init__` 用 native 組裝載入：audio encoder、`Qwen3ForCausalLM`、`WhisperFeatureExtractor`、tokenizer。
- `llm_punctuator/__init__.py` 匯出 `ASRPunctuator`。

### 1.3 一篇的資料流

逐篇做，不切 chunk。

1. 音檔轉成單聲道 16 kHz，抽特徵，經過 audio encoder，得到音訊向量。
2. 組 prompt，逐字照 qwen_asr 的 `_build_text_prompt`。實作時會跟它對一次字串。

   ```
   <|im_start|>system\n{context 填好}<|im_end|>\n
   <|im_start|>user\n<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>\n
   <|im_start|>assistant\nlanguage {2.2 對應表的名稱}<asr_text>
   ```

3. 文字經過 `split_segments`，建 `CustomLogitsProcessor`。
4. 呼叫 `greedy_search([這一篇], k)`。
5. ASR 的 `forward` 把 `<|audio_pad|>` 的 embedding 換成音訊向量。
   - 音訊向量在呼叫 `greedy_search` 前存到 `self`，結束後清掉。
   - `forward` 檢查 `<|audio_pad|>` 的個數跟向量數一致，不一致就 raise。

## 第 2 段：輸入前處理與錯誤

### 2.1 檢查順序

所有檢查都在讀音檔、跑模型之前做完。輸入錯了，不會跑到一半才失敗。
錯誤的種類與訊息見 #26 的 Output (Failures)。這裡只定順序。

建構時：
1. 模型名稱不支援 → `ValueError`，訊息列出支援的名稱。這在下載之前。

`add_punctuation` 依序檢查：
1. `texts` 是 `str`。
2. `k` 是 0 或小於 -1。
3. `texts` 與 `audios` 長度不同。
4. `language` 不在 2.2 的對應表。
5. `language` 不是 `zh` 或 `en`，又沒給 `punctuations`。
6. `audios` 裡有不是一維的陣列。
7. `audios` 裡有路徑指到不存在的檔案。先檢查全部路徑，再開始做第一篇。

### 2.2 語言標籤

`language` 收 BCP 47 的主要語言子標籤。規則是 ISO 639-1 有兩字母代碼就用兩字母，沒有才用 ISO 639-3 的三字母。
各模型在內部轉成自己的標籤。

這張票只收下表的子標籤。帶地區的完整標籤，例如 `zh-TW`，報 `ValueError`。

Qwen3-ASR 的對應表。名稱清單來自 qwen_asr 的 `SUPPORTED_LANGUAGES`，30 種全收：

| 標籤 | 名稱 | 標籤 | 名稱 | 標籤 | 名稱 | 標籤 | 名稱 |
|---|---|---|---|---|---|---|---|
| zh | Chinese | en | English | yue | Cantonese | ar | Arabic |
| de | German | fr | French | es | Spanish | pt | Portuguese |
| id | Indonesian | it | Italian | ko | Korean | ru | Russian |
| th | Thai | vi | Vietnamese | ja | Japanese | tr | Turkish |
| hi | Hindi | ms | Malay | nl | Dutch | sv | Swedish |
| da | Danish | fi | Finnish | pl | Polish | cs | Czech |
| fil | Filipino | fa | Persian | el | Greek | ro | Romanian |
| hu | Hungarian | mk | Macedonian | | | | |

`zh` 以外的標籤都照非 zh 處理，`yue` 也是：每個 token 都可以下標點，也沒有預設標點。
`yue` 要不要套 #24 的單位規則，不在這張票。

### 2.3 音檔前處理

照 qwen_asr 的 `normalize_audio_input` 做，讓模型看到的輸入跟官方前處理一樣。

- 路徑：用 `soundfile.read(path, dtype="float32")` 讀。多聲道時，各聲道取平均。
- `(numpy 陣列, 取樣率)`：陣列必須是一維。qwen_asr 對二維陣列是用形狀去猜哪一維是聲道，這裡不猜。
- 取樣率不是 16000：用 `soxr.resample(..., quality="HQ")` 轉換。
- 峰值大於 1 時除以峰值，再 clip 到 [-1, 1]。這跟 qwen_asr 的 `float_range_normalize` 相同。

### 2.4 context

`context.replace("{text}", text)`。不用 `str.format`，所以 `{text}` 以外的大括號原樣保留。

填好的 context 放進 system 訊息，這跟 qwen_asr 的做法相同。ASR 沒有 system prompt 參數。

## 第 3 段：CLI 與驗證

### 3.1 example.py

`-m` 是 `ASRPunctuator` 支援的名稱時，走 ASR。其他名稱照舊走文字 LLM。

新增兩個參數：
- `--audio PATH`：音檔路徑。
- `--context TEMPLATE`：context 模板。沒給時用 `add_punctuation` 的預設值。

#26 Output (Failures) 列的 CLI 組合，一律用 `parser.error` 結束，結束碼 2。

`-c` 的預設值改成 `None`，才分得出使用者有沒有給。文字 LLM 沒給 `-c` 時照舊用 50。

不動的部分：
- `-l` 的選項維持 `zh`、`en`。CLI 要開放更多語言，不在這張票。
- `-b` 照舊只在 `--list` 時使用。ASR 不能用 `--list`，所以 `-b` 對 ASR 沒有作用，跟現在文字 LLM 不用 `--list` 時一樣。
- 文字照舊經過 `preprocess` 清理。

### 3.2 單元測試，不用 GPU

假模型照 `tests/test_punctuator.py` 的做法：允許下標點時，選標點清單的第一個。
tokenizer 用 `Qwen/Qwen3-ASR-1.7B` 的真 tokenizer。只下載 tokenizer 檔案。

- #24 的 zh 各條，`k=-1` 與 `k=1` 都跑，套在 `ASRPunctuator` 上。
  - 用的是 `test_zh_marks_never_split_a_unit` 裡 `chunk_size=50` 的各條。
  - chunk 切點的各條不適用，因為 ASR 不切 chunk。
- #26 Output (Failures) 的每一種錯誤各一個測試。CLI 的錯誤照 `tests/test_example.py` 的做法。
- 音檔前處理：
  - 同一個訊號用路徑與 `(陣列, 16000)` 輸入，前處理結果相同。
  - 48 kHz 輸入轉成 16 kHz，長度正確。
- `<|audio_pad|>` 換成音訊向量：用小型隨機初始化的真 `Qwen3ForCausalLM`。
  檢查換過的位置真的吃到音訊向量，個數不符時會 raise。
  這是為了不讓假模型去驗證自己對 transformers 的假設。

### 3.3 對照，在 exp/envs/asr 跑，不進 repo

- prompt 字串：跟 qwen_asr 的 `_build_text_prompt` 逐字相同，`context` 有內容與空字串各一次。
- 重取樣：同一個 48 kHz 訊號，`soxr` HQ 跟 `librosa.resample` 的輸出逐點比對。

### 3.4 真模型，用 GPU，結果由 PR 回報

跑之前先看 `nvidia-smi`。結果寫到 `exp/runs/20261014-gh26/`。

- AC 1：42 筆，兩個模型各跑一次。回報拿掉標點後跟原文相同的筆數。
- AC 2：同一個音檔用路徑與陣列各跑一次，輸出逐字比對。
- AC 3：把 001.wav 轉成 48 kHz 的檔案再跑，回報拿掉標點後是否跟原文相同。
- AC 5：照票上的指令跑 `example.py`，貼出輸出。
- 跟原型對照：1.7B 的輸出，跟 `exp/runs/20261003-v2/native.jsonl` 的 `ctx` 條件、`native_cached` 欄比對，回報逐字相同的筆數。
  只對兩邊文字相同的樣本比。原型有先拿掉空白。這一項只是參考，不是 gate。分歧預期來自 bf16 的平手。
- 文字 LLM 的回歸：在 main 與本分支各跑一次 `scripts/run_benchmark.py --k -1`，輸出要逐字相同。
  `ForcedPunctuator` 只搬程式碼，不改模型輸入，所以應該完全相同。
