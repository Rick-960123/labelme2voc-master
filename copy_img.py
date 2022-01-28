from sklearn.model_selection import train_test_split
import os ,cv2

imagedir = './out/'
outdir = './datasets_seg/'

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    if len(file.split('_'))<2 or file.split('_')[1]!='label.png':
        continue
    images.append(filename)
for i in images:
    img=cv2.imread("./labelme-1024/"+i.split('_')[0]+'.jpg')
    cv2.imwrite(outdir+'img/'+i+'.jpg',img)
    img2 = cv2.imread(imagedir + i + '.png')
    cv2.imwrite(outdir + 'labels/' + i + '.png', img2)
    print("wangcheng")