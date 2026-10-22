# LLM Punctuator

Add punctuation to unpunctuated text (ASR outputs, transcripts) using LLM with constrained generation to preserve original content

## Table of Contents

- [Installation](#installation)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
- [Usage](#usage)
  - [Distance Bonus (ASR models)](#distance-bonus-asr-models)
  - [Supported Models](#supported-models)
- [Benchmark Results](#benchmark-results)
- [Development](#development)
  - [Install Git Hooks](#install-git-hooks)
  - [Conventional Commits Format](#conventional-commits-format)
  - [Code Quality Tools](#code-quality-tools)
- [TODO](#todo)

## Installation

Tested on python3.10

### Using uv (recommended)
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package and dependencies
uv pip install -e .
```

### Using pip
```bash
pip install -e .
```

## Usage

Inference using a text file:
```bash
python example.py --file data/zh_weather_forecast_1.txt
```

Or specify a different model:
```bash
python example.py \
    -m Qwen/Qwen2-1.5B-Instruct \
    --file data/zh_weather_forecast_1.txt
```

Or inference using texts directly:
```bash
python example.py \
    --text 關心天氣了今天封面上東遠離所以後方比較乾的冷空氣會南下那麼清晨到上午各地也會有一些降雨到是中午過後就會逐漸轉乾剩下的是東部還有大台北地區呢可能還會有一些局部的短暫雨其他地方都會轉為是多雲道型的天氣狀況了而至於在溫度方面呢今天清晨各地低溫大概是十七到二十度白天北台灣的高溫大概也就是在二十度上下所以整天來說還是偏涼的至於在東部是二十三度二十四度左右中南部地區甚至可以來到二十七度倒是今天入夜之後到明天清晨這段時間會變得比較冷一點中部以北跟東北部的地區氣溫大概就只有十五十六度沿海空曠地區或近山區的平地氣溫可能會稍微再更低一些南部跟花東是十八十九度我就提醒您如果今天平安夜明天的聖誕節想要出去過節的話務必要做好保暖工作
```

Or punctuate every text in a JSONL list, loading the model once:
```bash
python example.py --list texts.jsonl -o punctuated.jsonl
```
See `python example.py --help` for the list format.

### Distance Bonus (ASR models)

An ASR model may stop putting marks for hundreds of characters. `distance_bonus` makes a mark
more likely the further it is from the last mark:

```python
punctuator.add_punctuation(texts, audios, distance_bonus=0.2)
```
```bash
python example.py -m Qwen/Qwen3-ASR-1.7B --text ... --audio a.wav --distance-bonus 0.2
```

Where a mark may go, every mark's score gets `distance_bonus` times the number of characters
since the last mark. The count starts at the start of the text. It counts the non-whitespace
characters of the input text, in every language. The mark then competes with the next token,
which is text or the end of the text. The default `0.0` leaves the output as it was.
`distance_bonus` must be finite and `>= 0`.

#### Measured Effect

Measured in #43 with Qwen3-ASR-1.7B on 30 episodes of one Taiwanese news show. Each episode is
2.7 to 4.3 minutes long.

| b | 0 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| F1 against free transcription | 0.691 | 0.814 | 0.865 | 0.873 | 0.874 |
| Pause score S summed over 30 episodes | 482.7 | 586.1 | 617.1 | 609.2 | 594.2 |

- F1 compares where marks go, not which marks. The reference is the marks of the same model
  transcribing each segment freely. Only marks that map onto the input text count.
- S is defined below. Of the 10 values of b tried, S peaks at b=0.2.
- Read by hand, b=0.3 starts to break text in wrong places, such as 「告發自己，性騷也讓」.
  b=0 to 0.25 do not.
- So b=0.2 was chosen for this data.

#### Picking a Value for Your Data

1. Align each text to its audio with Qwen3-ForcedAligner to get the time of each word. A gap
   of at least 0.08 seconds between two words is a pause. That is one frame of the aligner.
2. Transcribe the audio freely with the same model. Its marks are where a mark should go.
   Per episode, q1 is the share of those positions with a pause. q0 is the share of the other
   positions with a pause. κ = (q1 + q0) / 2.
3. For each candidate b, S(b) = marks on a pause − κ × all marks. κ is the break-even point.
   It assumes an extra wrong mark costs as much as a missing one.
4. Sum S over many episodes, then take the b with the largest sum. Per episode, the best b
   ranged from 0.03 to 0.5 over the 30 episodes. The median gap in S between two neighboring
   values of b was only 0.64, and 46 of 270 pairs were equal. A little noise flips the choice.
5. To set the pause threshold, take the one with the largest q1 − q0. On this data it is
   0.595 at 1 frame. It falls to 0.434 at 3 frames, 0.24 seconds. At 4 frames, 0.32 seconds,
   it falls to 0.248, and the chosen b becomes 0.05. Speech research often uses 200 to 250 ms
   as the threshold. In read speech, short pauses cluster at 100 to 150 ms
   (Campione & Véronis 2002). A threshold of 200 ms or more drops the pauses at commas.

This method was checked on one news show, one model, and Chinese only.

### Supported Models

This package supports any HuggingFace model with chat template support. The following models have been tested:

**Recommended:**
* **Qwen/Qwen3-1.7B** (Default, best balance of speed and quality)

## Server

Run as an HTTP server:

### Direct
```bash
python -m llm_punctuator.server
```

### With custom model
```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B PORT=8000 python -m llm_punctuator.server
```

### Docker
```bash
docker build -t llm-punctuator .
docker run -p 8000:8000 -e MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B llm-punctuator
```

### Client Script
```bash
# 中文（預設）
python scripts/client.py "你好世界今天天氣真好"

# 英文
python scripts/client.py "hello world how are you" --language en

# 指定 server
python scripts/client.py "你好世界" --base-url http://localhost:9000
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```
Returns `200` when model is ready, `503` when loading or failed.

#### Server Info
```bash
curl http://localhost:8000/info
```

#### Punctuate
```bash
curl -X POST http://localhost:8000/api/v1/punctuate \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天氣很好出門記得帶傘", "language": "zh"}'
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to punctuate (min 1 char) |
| `language` | string | No | `"zh"` or `"en"` (default from settings) |
| `chunk_size` | int | No | Chunk size for long text (default from settings) |
| `punctuations` | string | No | Allowed punctuation characters (e.g. `"，。？！"`) |
| `k` | int | No | Most mark positions decided per forward pass; `-1` decides the whole chunk (default `-1`) |

### Environment Variables

See [.env.example](.env.example) for a template. Copy it and modify as needed:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-1.7B` | HuggingFace model name or path |
| `DEFAULT_LANGUAGE` | `zh` | Default language |
| `DEFAULT_CHUNK_SIZE` | `200` | Default chunk size |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `info` | Log level |

## Benchmark Results

Reference data is organized by category under `data/benchmark/`:

```
data/benchmark/
├── asr/    # ASR transcripts (speech, conversation)
├── news/   # News articles
└── wiki/   # Wikipedia-style encyclopedia entries
```

Each category contains at least 5 reference samples. Input text is derived automatically by stripping punctuation from references.

### Step 1: Generate Model Output

```bash
python scripts/run_benchmark.py --output-dir results/my_model
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | (required) | Directory to save output files |
| `-m, --model` | `Qwen/Qwen3-1.7B` | Model name or path |
| `-l, --language` | `zh` | Language (`zh` or `en`) |
| `-c, --chunk-size` | `50` | Chunk size for processing |
| `--k` | `-1` | Most mark positions decided per forward pass; `-1` decides the whole chunk |

Output mirrors the category structure: `results/my_model/{asr,news,wiki}/`.

### Step 2: Evaluate

```bash
python scripts/evaluate_benchmark.py --model-output results/my_model
```

Outputs per-category metrics tables followed by an overall summary.

## Development

### Install Git Hooks
This project uses git hooks to enforce code quality and commit message standards.

```bash
# Install git hooks (run this once after cloning the repository)
./bin/install-hooks.sh
```

The following hooks will be installed:
- **commit-msg**: Validates that commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) format
- **pre-push**: Runs linting checks (`./bin/lint.sh`) before allowing push

#### Conventional Commits Format
Commit messages must follow this pattern:
```
<type>(<scope>): <subject>
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, etc)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Changes to build system or dependencies
- `ci`: Changes to CI configuration
- `chore`: Other changes

**Examples:**
```
feat(auth): add login functionality
fix: resolve memory leak in data processing
docs(readme): update installation instructions
```

### Code Quality Tools
```bash
# Check code formatting and linting (without auto-fix)
./bin/lint.sh

# Auto-fix code formatting and linting issues
./bin/fix.sh
```

## TODO

- [ ] version control
- [ ] benchmark result
