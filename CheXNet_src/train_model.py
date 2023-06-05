# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


NEW_CKPT_PATH = "/home/ubuntu/CheXNet/new_model.pth.tar"
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR_TEST = '/home/ubuntu/CheXNet/ChestX-ray14/images'
TEST_IMAGE_LIST = '/home/ubuntu/CheXNet/ChestX-ray14/labels/test_list.txt'
DATA_DIR_TRAIN = '/home/ubuntu/CS231n-Final-Project/generated_images'
TRAIN_IMAGE_LIST = '/home/ubuntu/CS231n-Final-Project/CheXNet_src/generated_images.txt'
BATCH_SIZE = 8

def replaceLayers(): 
	new_model = DenseNet121(N_CLASSES).cuda()
	# Get the state_dict of the existing model
	existing_state_dict = new_model.state_dict()

	# Load the state_dict into the new model
	checkpoint = torch.load(NEW_CKPT_PATH)
	new_state_dict = checkpoint['state_dict']

	# Filter and update the existing state_dict with compatible layers from the loaded state_dict
	updated_state_dict = {}

	for key in existing_state_dict:
		if key in new_state_dict:
			updated_state_dict[key] = new_state_dict[key]
		else:
			updated_state_dict[key] = existing_state_dict[key]

	# Load the updated state_dict into the new model
	new_model.load_state_dict(updated_state_dict)
	return new_model

def main():

    cudnn.benchmark = True

    if os.path.isfile(NEW_CKPT_PATH):
        print("=> loading checkpoint")
        model = replaceLayers()
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR_TRAIN,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR_TEST,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    train(train_dataset)

    for i, (inp, target) in enumerate(tqdm(test_loader)):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

def train(dataset):
     # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameters
    num_epochs = 1
    #batch_size = 8

    # Prepare your dataset and create data loaders
    train_dataset = dataset
    #valid_dataset = ...
    train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    #valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # Initialize the model
    model = DenseNet121(N_CLASSES)
    model = model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        #total = 0
        #correct = 0

        for i, (inp, target) in enumerate(tqdm(train_loader)):
            target = target.to(device)
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            with torch.no_grad():
                input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device))
            #input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device), volatile=True)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

            # Calculate loss
            loss = nn.MSELoss()(output_mean, target)
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item() * input_var.size(0)

            #predicted_labels = torch.round(output_mean)
            #correct += (predicted_labels == target).sum().item()
            #total += target.size(0)
            #correct += (predicted_labels == target.view_as(predicted_labels)).sum().item()
            #total += target.size(0) * target.size(1)  # Multiply by n_crops
    
        # Calculate average training loss
        train_loss = train_loss / len(train_dataset)
        print("Train Loss:", train_loss)
        #print("Train Accuracy", 100 * correct / total)
    
        # Validation
#        model.eval()
#        valid_loss = 0.0
#        correct = 0
    
#        with torch.no_grad():
#            for inputs, labels in valid_loader:
#                inputs = inputs.to(device)
#                labels = labels.to(device)
            
                # Forward pass
#                outputs = model(inputs)
            
                # Calculate loss
#                loss = loss_function(outputs, labels)
#                valid_loss += loss.item() * inputs.size(0)
            
                # Calculate accuracy
#                _, predicted = torch.max(outputs.data, 1)
#                correct += (predicted == labels).sum().item()
    
        # Calculate average validation loss and accuracy
#        valid_loss = valid_loss / len(valid_dataset)
#        accuracy = correct / len(valid_dataset)
    
        # Print training and validation metrics for each epoch
#        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}')

     

if __name__ == '__main__':
    main()