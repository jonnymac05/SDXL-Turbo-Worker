import runpod
from diffusers import AutoPipelineForText2Image
import torch
import base64
import io
import time
import os

# Load the model
try:
    token = os.getenv("HUGGINGFACE_TOKEN") #added env variable
    pipe = AutoPipelineForText2Image.from_pretrained(
        # "stabilityai/sdxl-turbo", #commenting out previous model
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float16,
        use_auth_token=token # added auth token for huggingface
       # variant="fp16" #commenting out unnecessary
    )
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that processes jobs. """
    job_input = job['input']
    prompt = job_input.get('prompt', '')

    # Extract height and width from input, defaulting to 512 if not provided
    height = job_input.get('height', 512)
    width = job_input.get('width', 512)

    # Ensure dimensions are multiples of 8
    height = (height // 8) * 8
    width = (width // 8) * 8

    time_start = time.time()
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=1,
        guidance_scale=0.0
    ).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')

runpod.serverless.start({"handler": handler})
