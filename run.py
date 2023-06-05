import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from itertools import chain


# Step 1: Define the output directory
output_dir = '/home/ubuntu/CS231n-Final-Project/generated_images'
os.makedirs(output_dir, exist_ok=True)

# Step 2: Set the desired file format and compression level (for JPEG)
file_format = 'JPEG'
compression_level = 80


# Step 3: Load the RoentGen model
model_path = "/home/ubuntu/roentgen"
device='cuda'  # or mps, cpu...
pipe = StableDiffusionPipeline.from_pretrained(model_path)
pipe = pipe.to(device)


# Step 5: Enable progressive loading (for JPEG)
image_options = {'progressive': True} if file_format == 'JPEG' else {}

def save_images(generated_images, disease_type): 
    # Step 6: Save the images
    for i, generated_image in enumerate(tqdm(generated_images)):

        # Save the image
        num = i + 64
        image_name = f'{disease_type}_{num}.{file_format.lower()}'
        output_path = os.path.join(output_dir, image_name)
        generated_image.save(output_path, format=file_format, optimize=True, quality=compression_level, **image_options)

# Step 4: Run stable diffusion image generation
def generate_dataset_prompt(disease_type, num_data = 50):
    prompts = [disease_type] * num_data
    return prompts

diseases = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

print("Start inference loop")
num_examples = 256
batch_size = 8
def split_list_into_batches(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

for disease in diseases: 
    print(disease)
    total_images = []
    prompts =  generate_dataset_prompt(disease, num_data = num_examples)
    prompts_split = split_list_into_batches(prompts, batch_size=batch_size)
    for prompt_batch in tqdm(prompts_split):
        output = pipe(prompt_batch, num_inference_steps=75, height=512, width=512, guidance_scale=4)
        generated_images_batch = output.images
        total_images.append(generated_images_batch)
    total_images = list(chain.from_iterable(total_images))
    save_images(generated_images=total_images, disease_type=disease)




