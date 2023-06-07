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
from torch.utils.data import random_split
from read_data import ChestXrayDataSet
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import wandb
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default=None,
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    args = parser.parse_args()
    return args

args = parse_args()

wandb.init(
    # set the wandb project where this run will be logged
    project="cs231n_final_project",
    name = "cs231n_lr_" + str(args.learning_rate) + "_epochs_" + str(args.epochs) + "_" + args.data_type, 
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": args.learning_rate,
    "architecture": "CheXNet",
    "dataset": args.data_type,
    "epochs": args.epochs,
    }
)
NEW_CKPT_PATH = "/home/ubuntu/CheXNet/new_model.pth.tar"
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR_TEST = '/home/ubuntu/CheXNet/ChestX-ray14/images'
TEST_IMAGE_LIST = '/home/ubuntu/CheXNet/ChestX-ray14/labels/test_list.txt'
if args.data_type == "generated": 
    DATA_DIR_TRAIN = '/home/ubuntu/CS231n-Final-Project/generated_images'
    TRAIN_IMAGE_LIST = '/home/ubuntu/CS231n-Final-Project/CheXNet_src/generated_images.txt'
elif args.data_type == "mixed":
    DATA_DIR_TRAIN = '/home/ubuntu/CheXNet/ChestX-ray14/images'
    TRAIN_IMAGE_LIST = '/home/ubuntu/CS231n-Final-Project/CheXNet_src/mixed_train.txt'
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
    length = len(train_dataset[0])
    total_length = len(train_dataset)
    train_length = int(0.75 * total_length)
    valid_length = total_length - train_length

# Split the dataset
    train_dataset, validation_dataset = random_split(train_dataset, [train_length, valid_length])

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

    train(train_dataset, validation_dataset, model)

    # switch to evaluate mode
    model.eval()

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
    CLASS_NAMES.append("AVG")
    AUROCs.append(AUROC_avg)
    import pandas as pd

    list1 = CLASS_NAMES
    list2 = AUROCs

    data = {col_name: [row_value] for col_name, row_value in zip(list1, list2)}
    df = pd.DataFrame(data)
    my_table = wandb.Table(dataframe=df)
    wandb.log({"AUROCs": my_table})
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

def train(train_dataset, validation_dataset, model):
     # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameters
    num_epochs = wandb.config.epochs

    # Prepare your dataset and create data loaders

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=8, pin_memory=True)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        num_batches = 0

        for i, (inp, target) in enumerate(tqdm(train_loader)):
            num_batches += 1
            target = target.to(device)
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()

            with torch.no_grad():
                input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device))

            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

            # Calculate loss
            loss = nn.MSELoss()(output_mean, target)
        
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss_individual': loss})
        
            train_loss += loss.item() * input_var.size(0)
    
        # Calculate average training loss
        train_loss = train_loss / num_batches
        wandb.log({'train_loss_batch': train_loss})
        print("Train Loss:", train_loss)
    

    model.eval()  # Set model to evaluation mode

    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    #model.train()

    # Training loop
    
        # Training
    val_loss = 0.0
    num_batches = 0

    for i, (inp, target) in enumerate(tqdm(val_loader)):
        target = target.to(device)
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()

        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device))

        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

        # Calculate loss
        loss = nn.MSELoss()(output_mean, target)
        wandb.log({'val_loss_individual': loss})
        val_loss += loss.item() * input_var.size(0)
        num_batches += 1

    val_loss /= num_batches
    wandb.log({'val_loss_batch': train_loss})
    f1_scores = f1_score(gt.cpu().numpy(), torch.round(pred).cpu().numpy(), average=None)
    #accuracies = accuracy_score(gt.cpu().numpy(), pred.cpu().numpy())
    #accuracies = accuracy_score(gt.cpu().numpy(), torch.round(pred).cpu().numpy(), multioutput='raw_values')
    # Calculate other metrics if needed

    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", f1_scores)
"""
    gt = torch.FloatTensor().to(device)
    pred = torch.FloatTensor().to(device)
    val_loss = 0.0

    with torch.no_grad():
        for inp, target in tqdm(val_loader):
            inp = inp.to(device)
            target = target.to(device)
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).to(device))
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)
            loss = nn.MSELoss()(output_mean, target)
            val_loss += loss.item() * input_var[0]

    # Calculate evaluation metrics
    #mse_loss = nn.MSELoss()(pred, gt).item()
    f1_scores = f1_score(gt.cpu().numpy(), torch.round(pred).cpu().numpy(), average=None)
    #accuracies = accuracy_score(gt.cpu().numpy(), pred.cpu().numpy())
    #accuracies = accuracy_score(gt.cpu().numpy(), torch.round(pred).cpu().numpy(), multioutput='raw_values')
    # Calculate other metrics if needed

    print("Validation Loss:", val_loss)
    print("Validation Accuracy:", f1_scores)
        #print("Train Accuracy", 100 * correct / total)
"""    


     

if __name__ == '__main__':
    main()