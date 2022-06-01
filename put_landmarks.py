import matplotlib.pyplot as plt
import os

cwd = 'C:/Users/info/Desktop/Ear_Landmarks/ear_detection_rcnn/'


img_path = cwd + 'data/train/images/train_1.png'
label_path = cwd + 'data/train/lables/train_1.txt'
#label_path = cwd + 'data/train/landmarks/train_0.txt'
img_result_path = cwd + 'data/train'


img_original = plt.imread(img_path)




with open(label_path, 'r') as f:
    lines_list = f.readlines()         # all lines as list

    for j in range(3, 10): 
        string = lines_list[j]
        str1, str2 = string.split(' ')
        x_ = float(str1)
        y_ = float(str2)
        plt.scatter([x_ * 224], [y_ * 224])

plt.imshow(img_original)
plt.savefig(img_result_path)
plt.close()


# def put_landmarks(i, pred, single_img=False, myDir = 'your ciurrent dir'):

#     img_path = myDir + '/myear' + str(i+1) + 'resized.jpg'
#     img_result_path = myDir + '/myear' + str(i+1) + 'resized_landmarks.jpg'

#     if(single_img):      # if the case is single sample, not a whole set
#         #img_path = 'data/single/sampleimage.png'
#         #img_result_path = 'data/single/result/result.png'
#         img_path = os.path.join(myDir, 'single_img.png')
#         img_result_path = os.path.join(myDir, 'single_img_result.png')

#     img_original = plt.imread(img_path)

#     for j in range(0,55):  # drop the landmark points on the image
#         plt.scatter([pred[j]], [pred[j+55]])


#     plt.imshow(img_original)
#     plt.savefig(img_result_path)
#     plt.close()