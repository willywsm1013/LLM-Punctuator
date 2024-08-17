# LLM Punctuator
This repo tries to use LLM as a punctuator by constraining the next predicting tokens.

## Install
Tested on python3.10
```
pip install -r requirements.txt
```

## Usage
### Greedy Search
Inference using a text file:
```
python example.py \
    -m Qwen/Qwen2-1.5B-Instruct \
    --file  data/zh_weather_forecast_1.txt
```

Or inference using texts directly:
```
python example.py \
    -m Qwen/Qwen2-1.5B-Instruct \
    --text 關心天氣了今天封面上東遠離所以後方比較乾的冷空氣會南下那麼清晨到上午各地也會有一些降雨到是中午過後就會逐漸轉乾剩下的是東部還有大台北地區呢可能還會有一些局部的短暫雨其他地方都會轉為是多雲道型的天氣狀況了而至於在溫度方面呢今天清晨各地低溫大概是十七到二十度白天北台灣的高溫大概也就是在二十度上下所以整天來說還是偏涼的至於在東部是二十三度二十四度左右中南部地區甚至可以來到二十七度倒是今天入夜之後到明天清晨這段時間會變得比較冷一點中部以北跟東北部的地區氣溫大概就只有十五十六度沿海空曠地區或近山區的平地氣溫可能會稍微再更低一些南部跟花東是十八十九度我就提醒您如果今天平安夜明天的聖誕節想要出去過節的話務必要做好保暖工作
```
### Beam Search
This repo also implemented the beam search algorithm:
```
python example.py \
    -m Qwen/Qwen2-1.5B-Instruct \
    --file  data/zh_weather_forecast_1.txt \
    --num_beams 5
```

### Support Models
Currently only support these models:

* Qwen/Qwen2-7B-Instruct
* Qwen/Qwen2-1.5B-Instruct
* Qwen/Qwen2-0.5B-Instruct
* taide/Llama3-TAIDE-LX-8B-Chat-Alpha1
* 01-ai/Yi-1.5-6B-Chat
* 01-ai/Yi-1.5-9B-Chat
* meta-llama/Meta-Llama-3.1-8B-Instruct
* google/gemma-2-2b-it

See `src/punctuator/transformers.py` for more model details.

## TODO
- [ ] version control
- [ ] benchmark result