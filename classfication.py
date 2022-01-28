from sklearn.model_selection import train_test_split
import os

imagedir = './out/'
outdir = './out_label/'

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    if len(file.split('_'))<2 or file.split('_')[1]!='label.png':
        continue
    images.append(filename)

train, test = train_test_split(images, train_size=0.6, random_state=0)
val, test = train_test_split(test, train_size=0.5, random_state=0)

with open(outdir + "train.txt", 'w') as f:
    f.write('\n'.join(train))

with open(outdir + "val.txt", 'w') as f:
    f.write('\n'.join(val))

with open(outdir + "test.txt", 'w') as f:
    f.write('\n'.join(test))
