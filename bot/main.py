import asyncio
import logging
import os
import re
from typing import Dict, Optional, Tuple

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, BufferedInputFile
from aiogram.exceptions import TelegramBadRequest

import httpx
from dotenv import load_dotenv
from urllib.parse import quote

from .math_pretty import to_telegram_blocks


def load_config_from_env() -> Tuple[str, str, str, Optional[int]]:
    load_dotenv()
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    default_model = os.getenv("DEFAULT_MODEL", "flux").strip()
    default_size = os.getenv("DEFAULT_SIZE", "1024x1024").strip()
    default_seed_env = os.getenv("DEFAULT_SEED", "").strip()
    default_seed = int(default_seed_env) if default_seed_env.isdigit() else None

    if not telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Put it in .env or environment.")

    return telegram_bot_token, default_model, default_size, default_seed


def parse_flags_from_text(text: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """Parse flags like --size=1024x1024 --seed=123 --model=flux from the tail of the text.

    Returns tuple of (prompt_without_flags, flags_dict)
    """
    # Pattern matches --key=value or --key value
    flag_pattern = re.compile(r"--(?P<key>model|size|seed)\s*(=\s*(?P<val1>\S+)|\s+(?P<val2>\S+))?")

    flags: Dict[str, Optional[str]] = {}
    def replace_match(m: re.Match) -> str:
        key = m.group("key")
        value = m.group("val1") or m.group("val2") or None
        flags[key] = value
        return ""  # strip from text

    prompt = flag_pattern.sub(replace_match, text).strip()
    return prompt, flags


def build_pollinations_url(prompt: str, model: Optional[str], size: Optional[str], seed: Optional[int]) -> str:
    # Pollinations accepts direct prompt path and query params like size, seed, model
    # size is typically WIDTHxHEIGHT, e.g., 1024x1024
    encoded_prompt = quote(prompt, safe="")
    base_url = "https://image.pollinations.ai/prompt"

    query_params = []
    if size:
        query_params.append(f"size={size}")
    if seed is not None:
        query_params.append(f"seed={seed}")
    if model:
        query_params.append(f"model={quote(model)}")

    query = ("?" + "&".join(query_params)) if query_params else ""
    return f"{base_url}/{encoded_prompt}{query}"


async def try_send_photo_via_url(message: Message, url: str, caption: Optional[str]) -> bool:
    try:
        await message.answer_photo(photo=url, caption=caption)
        return True
    except TelegramBadRequest:
        return False


async def fetch_image_bytes(url: str) -> Optional[bytes]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(60.0)) as client:
        resp = await client.get(url)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
            return resp.content
        return None


def build_caption(prompt: str, model: Optional[str], size: Optional[str], seed: Optional[int]) -> str:
    parts = [f"🖼️ {prompt}"]
    meta = []
    if model:
        meta.append(f"model={model}")
    if size:
        meta.append(f"size={size}")
    if seed is not None:
        meta.append(f"seed={seed}")
    if meta:
        parts.append("\n" + " · ".join(meta))
    return "".join(parts)


def extract_args_after_command(text: str) -> str:
    # Remove leading command like "/img" possibly with bot username
    return re.sub(r"^\s*/\w+(?:@\w+)?\s*", "", text, count=1).strip()


def resolve_parameters(prompt_flags: Dict[str, Optional[str]], defaults: Tuple[str, str, Optional[int]]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    default_model, default_size, default_seed = defaults

    model = (prompt_flags.get("model") or default_model).strip() if (prompt_flags.get("model") or default_model) else None
    size = (prompt_flags.get("size") or default_size).strip() if (prompt_flags.get("size") or default_size) else None
    seed_str = prompt_flags.get("seed")
    seed = int(seed_str) if (seed_str and seed_str.isdigit()) else default_seed

    return model, size, seed


async def on_start(message: Message) -> None:
    text = (
        "Привет! Я генерирую изображения через Pollinations.\n\n"
        "Использование: /img <описание> [--model flux|flux-realism|flux-anime] [--size 1024x1024] [--seed 123]\n\n"
        "Пример: /img милый корги астронавт --size 768x768 --model flux --seed 42"
    )
    await message.answer(text)


async def on_img(message: Message, defaults: Tuple[str, str, Optional[int]]) -> None:
    raw_args = extract_args_after_command(message.text or "")
    if not raw_args:
        await message.answer("Добавьте описание после команды. Пример: /img синтвейв пейзаж")
        return

    prompt_text, flags = parse_flags_from_text(raw_args)
    if not prompt_text:
        await message.answer("Нужно текстовое описание после команды /img")
        return

    model, size, seed = resolve_parameters(flags, defaults)
    url = build_pollinations_url(prompt_text, model=model, size=size, seed=seed)

    caption = build_caption(prompt_text, model=model, size=size, seed=seed)

    # First try to let Telegram fetch by URL
    sent = await try_send_photo_via_url(message, url=url, caption=caption)
    if sent:
        return

    # Fallback: fetch bytes and upload
    await message.answer("Генерирую изображение, подождите…")
    image_bytes = await fetch_image_bytes(url)
    if not image_bytes:
        await message.answer("Не удалось получить изображение от Pollinations. Попробуйте изменить запрос или параметры.")
        return

    await message.answer_photo(photo=BufferedInputFile(image_bytes, filename="pollinations.jpg"), caption=caption)


async def on_math(message: Message) -> None:
    raw = extract_args_after_command(message.text or "")
    if not raw:
        await message.answer("Использование: /math <LaTeX или выражение>. Пример: /math \\int_0^1 x^2 dx")
        return
    blocks = to_telegram_blocks(raw)
    if not blocks:
        await message.answer("Не удалось обработать выражение.")
        return
    for kind, content in blocks:
        if kind == "text":
            await message.answer(text=content, parse_mode="HTML")
        else:  # code block
            await message.answer(text=f"<pre><code>{content}</code></pre>", parse_mode="HTML")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    token, default_model, default_size, default_seed = load_config_from_env()

    bot = Bot(token=token, parse_mode="HTML")
    dp = Dispatcher()

    # Handlers
    dp.message.register(on_start, Command("start"))
    dp.message.register(lambda m: on_img(m, (default_model, default_size, default_seed)), Command("img"))
    dp.message.register(on_math, Command("math"))

    logging.info("Bot is starting polling…")
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass