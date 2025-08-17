import os
import random
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def ensure_output_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def build_pipeline(model_id: str) -> StableDiffusionPipeline:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	dtype = torch.float16 if device == "cuda" else torch.float32
	pipe = StableDiffusionPipeline.from_pretrained(
		model_id,
		safety_checker=None,
		revision=None,
		torch_dtype=dtype,
	)
	# Turbo models come with their own scheduler, but keep fallback here
	try:
		pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
	except Exception:
		pass
	pipe = pipe.to(device)
	if device == "cpu":
		for _, module in pipe.components.items():
			try:
				module.to("cpu")
			except Exception:
				pass
	# Memory improvements
	try:
		pipe.enable_attention_slicing()
		if torch.cuda.is_available():
			pipe.enable_sequential_cpu_offload()
	except Exception:
		pass
	if device == "cuda":
		try:
			pipe.enable_xformers_memory_efficient_attention()
		except Exception:
			pass
	pipe.set_progress_bar_config(disable=True)
	return pipe


def apply_motion_blur(image: Image.Image, kernel_size: int = 7) -> Image.Image:
	# Fallback to a slight box blur to mimic gentle motion without kernel issues
	return image.filter(ImageFilter.BoxBlur(1.2))


def add_sensor_noise(image: Image.Image, sigma: float = 3.0) -> Image.Image:
	arr = np.array(image).astype(np.float32)
	noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
	noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
	return Image.fromarray(noisy)


def generate_image(
	output_dir: str = "/workspace/output",
	filename_prefix: str = "unintentional_selfie",
	seed: int | None = None,
) -> str:
	ensure_output_dir(output_dir)

	model_id = os.environ.get("MODEL_ID", "stabilityai/sd-turbo")
	pipe = build_pipeline(model_id)

	if seed is None:
		seed = random.randint(0, 2**32 - 1)
	generator_device = "cuda" if torch.cuda.is_available() else "cpu"
	generator = torch.Generator(device=generator_device).manual_seed(seed)

	# Use smaller base size on CPU, then upscale to 1080x1920
	base_width = int(os.environ.get("BASE_WIDTH", "640"))
	base_height = int(os.environ.get("BASE_HEIGHT", "1136"))  # ~9:16
	final_width, final_height = 1080, 1920

	prompt = (
		"accidental selfie, random unintentional shot, partial face out of frame, "
		"handheld iPhone photo, natural human skin, normal body proportions, casual street at night, "
		"even amber street lighting, slight motion blur, realistic, candid, unposed, "
		"subtle film grain, shallow depth of field, real photograph"
	)
	negative_prompt = (
		"cgi, 3d render, cartoon, anime, doll, mannequin, plastic, game character, "
		"deformed, extra fingers, extra limbs, watermark, text, logo, oversharp, "
		"ai generated artifacts, unrealistic skin, mask, face paint"
	)

	steps = int(os.environ.get("STEPS", "6")) if "sd-turbo" in model_id else int(os.environ.get("STEPS", "18"))
	guidance = 0.0 if "sd-turbo" in model_id else 5.0

	with torch.autocast("cuda", enabled=torch.cuda.is_available()):
		image = pipe(
			prompt=prompt,
			negative_prompt=negative_prompt,
			width=base_width,
			height=base_height,
			num_inference_steps=steps,
			guidance_scale=guidance,
			generator=generator,
		).images[0]

	image = apply_motion_blur(image, kernel_size=7)
	image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
	image = add_sensor_noise(image, sigma=2.0)
	image = image.resize((final_width, final_height), resample=Image.Resampling.BICUBIC)

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	outfile = os.path.join(output_dir, f"{filename_prefix}_{timestamp}_seed{seed}.jpg")
	image.save(outfile, format="JPEG", quality=92, subsampling=2, optimize=True)
	return outfile


if __name__ == "__main__":
	outfile = generate_image()
	print(f"Saved: {outfile}")