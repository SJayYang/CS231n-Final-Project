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
image = pipe(prompt).images[0]

# Step 4: Enable progressive loading (for JPEG)
image_options = {'progressive': True} if file_format == 'JPEG' else {}

# Step 5: Resize the image if necessary
max_width = 1000

# Step 6: Save the images
for i, generated_image in enumerate(generated_images):
    # Assuming 'generated_image' is a torch.Tensor

    # Move the image to the CPU if using GPU
    generated_image = generated_image.cpu()

    # Convert the generated image to a PIL Image object
    generated_image = transforms.ToPILImage()(generated_image)

    # Resize the image if necessary
    if generated_image.width > max_width:
        scale_factor = max_width / generated_image.width
        new_height = int(generated_image.height * scale_factor)
        generated_image = generated_image.resize((max_width, new_height), Image.LANCZOS)

    # Save the image
    image_name = f'image_{i}.{file_format.lower()}'
    output_path = os.path.join(output_dir, image_name)
    generated_image.save(output_path, format=file_format, optimize=True, quality=compression_level, **image_options)

    # Optionally display progress or other post-processing steps

print("Image generation and saving completed!")
