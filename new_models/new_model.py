import torchxrayvision as xrv
import skimage, torch, torchvision

# Prepare the image:
img_path = "/home/ubuntu/CS231n-Final-Project/generated_images/Fibrosis_0.jpeg"
img = skimage.io.imread(img_path)
img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
img = img.mean(2)[None, ...] # Make single color channel

transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])

img = transform(img)
img = torch.from_numpy(img)

# Load model and process image
model = xrv.models.DenseNet(weights="densenet121-res224-all")
outputs = model(img[None,...]) # or model.features(img[None,...]) 

# Print results
print(dict(zip(model.pathologies,outputs[0].detach().numpy())))

