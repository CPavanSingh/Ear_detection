import numpy as np


cwd = 'C:/Users/info/Desktop/Ear_Landmarks/ear_detection_rcnn/'

size = 3000
ctr = -1

for i in range(size):


    ctr += 1
    landmark_path = cwd + 'data/train/landmarks/train_' + str(ctr) + '.txt'
    lables = cwd + 'data/train/lables/train_' + str(ctr) + '.txt'


    with open(landmark_path, 'r') as f:
                lines_list = f.readlines()         # all lines as list

                for j in range(3, 58):          # in landmark text files, landmarks start at 3rd line end in 57th
                    string = lines_list[j]
                    str1, str2 = string.split(' ')
                    x_ = float(str1)
                    x_ = round(x_, 3)
                    y_ = float(str2)
                    y_ = round(y_, 3)


                    if (j == 3):                # if first landmark point
                        temp_x = np.array(x_)
                        temp_y = np.array(y_)

                    elif (j == 39 or j == 22 or j == 19 or j == 12 or j == 7 or j == 47):                          # if not first landmark point
                        temp_x = np.hstack((temp_x, x_))
                        temp_y = np.hstack((temp_y, y_))

                # if i == 0 :
                #     #print(temp_x, temp_y)
                #     for p in range(7):
                #         print(temp_x[p] , temp_y[p])
                
                with open(lables, 'w') as f:
                
                    f.write("version: 3\n")
                    f.write("n_points: 7\n")
                    f.write("{\n")
                    for p in range(7):
                        line = str(temp_x[p]) + " " + str(temp_y[p]) + "\n"
                        f.write(line)
                    f.write("}\n")
                    f.close()