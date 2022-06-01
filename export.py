import torch 
import torch.nn as nn

cwd = 'C:/Users/info/Desktop/Ear_Landmarks/ear_detection_rcnn/model_parameters/'
from torchvision.models.vgg import vgg16_bn


model = vgg16_bn()
model.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=1000, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=1000, out_features=500, bias=True),
    nn.Linear(in_features=500, out_features=224, bias=True),
    nn.Linear(in_features=224, out_features=14, bias=True),
    )

model.load_state_dict(torch.load(cwd + 'model9.pt') )


model.eval()


dummy_input = torch.zeros(1,3,224,224)

torch.onnx.export(model, dummy_input,'ear_landmarks.onnx', verbose=True)