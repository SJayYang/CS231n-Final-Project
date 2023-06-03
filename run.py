from diffusers import StableDiffusionPipeline
import torch

model_path = "/home/ubuntu/roentgen"
device='cuda'  # or mps, cpu...
pipe = StableDiffusionPipeline.from_pretrained(model_path)
# pipe = StableDiffusionPipeline.from_pretrained(model_path).to(torch.float32).to(device)
pipe = pipe.to(device)

prompt = "big right-sided pleural effusion"

# pipe([prompt], num_inference_steps=75, height=512, width=512, guidance_scale=4)
image = pipe(prompt).images[0]
