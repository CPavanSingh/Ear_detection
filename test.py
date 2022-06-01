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






import matplotlib.pyplot as plt
import os
from skimage import io
import torchvision.transforms as transforms

cwd = 'C:/Users/info/Desktop/Ear_Landmarks/ear_detection_rcnn/'


#img_path = cwd + 'data/test/images/test_1.png'
img_path = cwd + 'data/myear4resized.jpg'
label_path = cwd + 'data/train/lables/train_1.txt'
#label_path = cwd + 'data/train/landmarks/train_0.txt'
img_result_path = cwd + 'data/myear4'

image = io.imread(img_path)
transform = transforms.ToTensor()

image = transform(image)
image = torch.unsqueeze(image, 0)
predictions = model(image)

img_original = plt.imread(img_path)


print(predictions)

predictions = predictions.tolist()

print(predictions)

# with open(label_path, 'r') as f:
#     lines_list = f.readlines()         # all lines as list

for j in range(len(predictions[0])): 
        
    if j%2 == 0: 
         x_ = predictions[0][j]
         y_ = predictions[0][j+1]
         plt.scatter([x_ * 224], [y_ * 224])
    

plt.imshow(img_original)
plt.savefig(img_result_path)
plt.close()





