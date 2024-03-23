#imports here
import torch
import json
from torch import nn
from torch import optim
import numpy as np
from torchvision import datasets,transforms,models


model = models.vgg16(pretrained=True)
#freeze model's parameters
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                            ('fc1',nn.Linear(25088,12544)),
                            ('Relu',nn.ReLU()),
                            ('dropout',nn.Dropout(0.2)),
                            ('fc2',nn.Linear(12544,7)),
                            ('output',nn.LogSoftmax(dim=1))]))
model.classifier = classifier
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(),lr=0.001)

# TODO: Write a function that loads a checkpoint and rebuilds the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=device)
    input_size = checkpoint['input_size']
    output_size=checkpoint['output_size']
    epoch = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['learning_rate']
    model.load_state_dict(checkpoint['state_dict'])

    return model,class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    MAX_SIZE=(256,256)
    image.thumbnail(MAX_SIZE)

    #center crop to 224X224
    left_margin = (image.width - 224) / 2
    top_margin = (image.height - 224) / 2
    right_margin = left_margin + 224
    bottom_margin = top_margin + 224

    image = image.crop((left_margin, top_margin, right_margin, bottom_margin))

    #converts 0-255 to 0-1
    np_image = np.array(image)

    np_image = np_image/255.0

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    np_image = (np_image-means)/std

    formatted_img = np_image.transpose((2,0,1))
    tensor_image  = torch.from_numpy(formatted_img)
    return tensor_image


def load_model():
    model,class_to_idx= load_checkpoint(r"D:\checkpoint_new.pth")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model
    

def load_class_mapping(json_file_path):
    with open(json_file_path, 'r') as json_file:
        class_mapping = json.load(json_file)
    return class_mapping

def predict(image_tensor, model, class_mapping, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # TODO: Implement the code to predict the class from an image file
    input = image_tensor.float()
    input = input.unsqueeze(0)
    model.eval()

    with torch.no_grad():
        input = input.to(device)
        logps = model(input)

    ps = torch.exp(logps)

    top_p, top_indices = ps.topk(topk, dim=1)

    # Convert indices to actual class names using the class mapping
    top_classes = [class_mapping[str(idx.item())] for idx in top_indices[0]]

    return top_classes
