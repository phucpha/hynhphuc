from sklearn.preprocessing import LabelBinarizer

from keras import models
import numpy as np
import cv2
from os import listdir

model = models.load_model("modelsaved_5_07.h5")

path1 = "D:/cac_ki_hoc/04_CD2/link2_archive/TB_Chest_Radiography_Database/"

"""path1 source train """



font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)

fontScale = 2
color = (0, 255, 0)
thickness = 2

def createFileData(raw_path=path1):
    '''đọc dữ 400 hình trong dữ liệu train để test gồm 200 hình class 0 và 200 class 1'''
    images = []
    img_original = []
    lables= []
    names=[]


    i = 0
    for folder in listdir(raw_path):
        for file in listdir(raw_path+folder):
             if file != '.DS_Store':
                print(file)
                i=i+1
                if i > 200:
                    i=0
                    break
                '''đọc ảnh , resize,chuyển xám, tạo biên, '''
                img = cv2.resize(cv2.imread(raw_path  +folder +"/" + file), dsize=(512, 512))
                img_original.append(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imgx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3, scale=1, delta=0)
                imgy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3, scale=1, delta=0)
                imgx = cv2.convertScaleAbs(imgx)
                imgy = cv2.convertScaleAbs(imgy)
                img = cv2.addWeighted(imgx, 0.5, imgy, 0.5, 0)
                images.append(img)
                lables.append(file[0])
                names.append(file)
    '''mã hóa nhãn về [0] và [1]'''
    encoder = LabelBinarizer()
    labels_encode = encoder.fit_transform(lables)
    images = np.array(images)
    images = images.reshape(images.shape[0], 512, 512, 1)
    ''' :return images: mảng dữ liệu 400 ảnh(400, 512,512 ,1)
                img_original : mảng ảnh gốc chưa xữ lí dùng để hiển thị
                lables: mảng của nhãn (400,)
                names: mảng của tên file ảnh
                lables_encode: mảng của nhãn sau khi encode'''
    return images, img_original, lables, names , labels_encode


pictures, image_ori, lables, names , lables_encode = createFileData()
print(pictures.shape)
while (True):
    picture_numbber = int(input("nhap số thứ tự hình:"))
    print(names[picture_numbber])
    print(lables_encode[picture_numbber])

    y_predict = model.predict(pictures[picture_numbber].reshape(1, 512, 512, 1 ))
    print('Gia tri du doan: ', y_predict)

    '''nếu giá trị dự đoán nhỏ hơn 0.01 thì làm tròn =0'''
    if y_predict < 0.01:
        y_predict = 0

    y_predict = str(y_predict)



    picture = image_ori[picture_numbber]
    '''hình để hiển thị'''
    cv2.putText(picture,str(lables[picture_numbber]) + y_predict, org, font,
                fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("predict", picture)

    cv2.waitKey()

