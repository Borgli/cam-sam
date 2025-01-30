import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from mobile_sam import SamPredictor as MobileSamPredictor, sam_model_registry as mobile_sam_model_registry
from transformers.models.auto.image_processing_auto import model_type
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator  import SAM2AutomaticMaskGenerator

# Define your model
class PolypClassificationModel(nn.Module):
    def __init__(self):
        super(PolypClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def run_experiment(config):
    print('Running experiment with config:', config)

    # Load your data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.ImageFolder(root='path_to_your_train_data', transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize your model
    model = PolypClassificationModel()

    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train your model
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def load_segment_anything(model_type, checkpoint_folder, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def load_mobile_segment_anything_predictor(model_type, checkpoint_folder, device):
    sam = mobile_sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    mask_generator = MobileSamPredictor(sam)
    return mask_generator


def load_segment_anything_predictor(model_type, checkpoint_folder, device):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_folder)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def load_segment_anything_2_predictor():
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    return predictor


def load_segment_anything_2():
    predictor = SAM2AutomaticMaskGenerator.from_pretrained("facebook/sam2-hiera-large")
    return predictor


def convert_to_black_white(segmentation):
    """
    Converts a 3-channel boolean mask into a single-channel black and white mask.

    Parameters:
    mask (np.ndarray): A 3-dimensional numpy array of shape (3, height, width) where each slice represents a channel mask.

    Returns:
    np.ndarray: A 2-dimensional numpy array where each element is True if any corresponding element in the input channels is True.
    """
    # Combine the three channels using logical OR
    combined_mask = np.logical_or(np.logical_or(segmentation[0], segmentation[1]), segmentation[2])
    return combined_mask
