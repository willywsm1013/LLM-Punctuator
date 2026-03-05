# LLM Punctuator

Add punctuation to unpunctuated text (ASR outputs, transcripts) using LLM with constrained generation to preserve original content

## Table of Contents

- [Installation](#installation)
  - [Using uv (recommended)](#using-uv-recommended)
  - [Using pip](#using-pip)
- [Usage](#usage)
  - [Supported Models](#supported-models)
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

### Supported Models

This package supports any HuggingFace model with chat template support. The following models have been tested:

**Recommended:**
* **Qwen/Qwen3-1.7B** (Default, best balance of speed and quality)

## Server

Run as an HTTP server:

### Direct
```bash
python -m llm_punctuator
```

### With custom model
```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B PORT=8000 python -m llm_punctuator
```

### Docker
```bash
docker build -t llm-punctuator .
docker run -p 8000:8000 -e MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B llm-punctuator
```

### API Example
```bash
curl -X POST http://localhost:8000/api/v1/punctuate \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天氣很好出門記得帶傘", "language": "zh"}'
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-1.7B` | HuggingFace model name or path |
| `DEFAULT_LANGUAGE` | `zh` | Default language |
| `DEFAULT_CHUNK_SIZE` | `50` | Default chunk size |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `LOG_LEVEL` | `info` | Log level |

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
