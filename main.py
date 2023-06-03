import os
import torch
from PIL import Image
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline


# Step 1: Define the output directory
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

# Step 2: Set the desired file format and compression level (for JPEG)
file_format = 'JPEG'
compression_level = 80

# Step 3: Generate the images using GaussianDiffusion
model_path = "/home/ubuntu/roentgen"
device='cuda'  # or mps, cpu...
pipe = StableDiffusionPipeline.from_pretrained(model_path)
# pipe = StableDiffusionPipeline.from_pretrained(model_path).to(torch.float32).to(device)
pipe = pipe.to(device)

prompt = "big right-sided pleural effusion"

# pipe([prompt], num_inference_steps=75, height=512, width=512, guidance_scale=4)
output = pipe(prompt)
images = output.images

# Step 3: Save the images
for i, generated_image in enumerate(generated_images):
    # Assuming 'generated_image' is a torch.Tensor or numpy.ndarray

    # Move the image to the CPU if using GPU
    generated_image = generated_image.cpu()

    # Normalize the image tensor to [0, 1] range
    generated_image = (generated_image + 1) / 2

    # Convert the tensor to a PIL Image object
    generated_image = encoders.pil_image(generated_image)

    # Save the image as an individual file
    image_name = f'image_{i}.png'
    output_path = os.path.join(output_dir, image_name)
    generated_image.save(output_path)

print("Image generation and saving completed!")