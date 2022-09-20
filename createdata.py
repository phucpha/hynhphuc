import numpy as np
from os import listdir
import cv2
from sklearn.preprocessing import LabelBinarizer
import pickle
'''nơi lấy dữ liệu hình'''
raw_path_nomal = "D:/cac_ki_hoc/04_CD2/link2_archive/TB_Chest_Radiography_Database/"

def createFileData(raw_path= raw_path_nomal):
    images = []
    labels = []

    for folder in listdir(raw_path):
        if folder != '.DS_Store':
            print("folder",folder)
            for file in listdir(raw_path + folder ):
                if file != '.DS_Store':
                    print(file)
                    '''đọc ảnh và chuyển về ảnh xám và resise ảnh'''
                    img = cv2.resize(cv2.imread(raw_path + folder + "/" + file), dsize=(512, 512))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    '''nếu thêm xl blur'''
                    '''img = cv2.GaussianBlur(img, (3, 3), 0)'''

                    '''tạo biên biên theo 2 chiều x, y'''
                    imgx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1, delta=0)
                    imgy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1, delta=0)
                    imgx = cv2.convertScaleAbs(imgx)
                    imgy = cv2.convertScaleAbs(imgy)
                    img = cv2.addWeighted(imgx, 0.5, imgy, 0.5, 0)
                    images.append(img)
                    labels.append(folder)
    '''chuyển ảnh sang mảng numpy'''
    images = np.array(images)
    images = images.reshape(images.shape[0], 512, 512, 1)
    print(type(images))
    print(images.shape)
    labels = np.array(labels)
    print(type(labels))
    print(labels.shape)

    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)
    print(labels.shape)
    file = open('pix_blur.data', 'wb')
    # dump information to that file
    pickle.dump((images, labels), file)
    # close the file
    file.close()

createFileData()


