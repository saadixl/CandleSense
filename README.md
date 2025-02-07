## What is this project?
This is a project to predict price movement of a stock based on it's candle stick chart. I have some example candle stick charts which denotes if the price should go up or down. The idea is to find the closest match for the sample candle stick chart images to predict what will happen.

## How to use the project
- Assuming you have Python 3+ installed
- Create a `.env` file in the root
- Add your openai api key under `OPENAI_API_KEY`
- Run `pip3 install -r src/requirements.txt` to install the python packages
- Run `python3 src/main.py`
Upon running the command, you will see a result like this:
```
Sample image: nvda_1.png
Closest match: d-shooting-star.png
Similarity score: 79.76%
```
