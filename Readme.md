# Telegram bot with Pollinations AI image generation

## Setup

1. Create a virtual environment and install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2. Configure environment:

```
cp .env.example .env
# put your Telegram bot token into TELEGRAM_BOT_TOKEN
```

3. Run the bot:

```
python -m bot.main
```

## Usage in Telegram

- Command:

```
/img <prompt> [--model flux|flux-realism|flux-anime] [--size 1024x1024] [--seed 123]
```

- Example:

```
/img cute corgi astronaut --size 768x768 --model flux --seed 42
```

Notes:
- The bot uses `https://image.pollinations.ai/prompt/<prompt>` with optional query parameters.
- If Telegram cannot fetch by URL, the bot falls back to downloading and uploading the image.
- Large sizes or certain models might respond slower or fail occasionally; try another size/model/seed.
