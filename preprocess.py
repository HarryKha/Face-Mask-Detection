import xmltodict
import cv2
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join


def getImageNames():
    image_names = []
    for dirname, _, filenames in os.walk('./dataset'):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            extension = fullpath[len(fullpath) - 4:]
            if extension != '.xml':
                image_names.append(filename)
    
    # print(image_names)
    return image_names


def get_path(image_name):
    
    #CREDIT: kaggle.com/dohunkim
    
    home_path = './dataset/'
    image_path = home_path + 'images/1/' + image_name
    
    if image_name[-4:] == 'jpeg':
        label_name = image_name[:-5] + '.xml'
    else:
        label_name = image_name[:-4] + '.xml'
    
    label_path = home_path + 'annotations/' + label_name

    print(f"Image path: {image_path} , Label path: {label_path}")

    return  image_path, label_path

def parse_xml(label_path):
    
    #CREDIT: kaggle.com/dohunkim
    
    x = xmltodict.parse(open(label_path , 'rb'))
    item_list = x['annotation']['object']

    print(item_list)
    
    # when image has only one bounding box
    if not isinstance(item_list, list):
        item_list = [item_list]
        
    result = []
    
    for item in item_list:
        name = item['name']
        bndbox = [(int(item['bndbox']['xmin']), int(item['bndbox']['ymin'])),
                  (int(item['bndbox']['xmax']), int(item['bndbox']['ymax']))]       
        result.append((name, bndbox))
    
    size = [int(x['annotation']['size']['width']), 
            int(x['annotation']['size']['height'])]
    
    return result, size

def visualize_image(image_name, bndbox=True):
    
    #CREDIT: kaggle.com/dohunkim
    
    image_path, label_path = get_path(image_name)
    
    image = cv2.imread(image_path)
    print(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if bndbox:
        labels, size = parse_xml(label_path)
        thickness = int(sum(size)/400.)
        
        for label in labels:
            name, bndbox = label
            
            if name == 'with_mask':
                cv2.rectangle(image, bndbox[0], bndbox[1], (0, 255, 0), thickness)
            elif name == 'without_mask':
                cv2.rectangle(image, bndbox[0], bndbox[1], (255, 0, 0), thickness)
            elif name == 'mask_weared_incorrect':
                cv2.rectangle(image, bndbox[0], bndbox[1], (0, 0, 255), thickness)
    
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title(image_name)
    plt.imshow(image)
    plt.show()

image_names = getImageNames()

NUM_OF_IMGS_TO_VISUALIZE = 5

for i in range(NUM_OF_IMGS_TO_VISUALIZE):
    visualize_image(image_names[i])

def cropImage(image_name):
    image_path, label_path = get_path(image_name)
    
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    labels, size = parse_xml(label_path)
    
    cropedImgLabels = []

    for label in labels:
        name, bndbox = label
        
        
        croped_image = image[bndbox[0][1]:bndbox[1][1], bndbox[0][0]:bndbox[1][0]]
        
        label_num = 0
        
        if name == "with_mask":
            label_num = 0
        elif name == "without_mask":
            label_num = 1
        elif name == 'mask_weared_incorrect':
            label_num = 2
        
        cropedImgLabel = [croped_image, label_num]
        
        cropedImgLabels.append(cropedImgLabel)
        
    return cropedImgLabels

def createDirectory(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        print("Directory " + dirname + " already exists.")

dir_name = 'train/'
label_0_dir = dir_name + "0/"
label_1_dir = dir_name + "1/"
label_2_dir = dir_name + "2/"
models_dir = "models/"


createDirectory(dir_name)
createDirectory(label_0_dir)
createDirectory(label_1_dir)
createDirectory(label_2_dir)
createDirectory(models_dir)

label_0_counter = 0
label_1_counter = 0
label_2_counter = 0

for image_name in image_names:
    cropedImgLabels = cropImage(image_name)
    
    for cropedImgLabel in cropedImgLabels:
        
        label = cropedImgLabel[1]
        img = cropedImgLabel[0]
        
        if label == 0:
            croped_img_name = str(label_0_counter) + ".jpg"
            cv2.imwrite(label_0_dir + croped_img_name, img)
            label_0_counter += 1
        elif label == 1:
            croped_img_name = str(label_1_counter) + ".jpg"
            cv2.imwrite(label_1_dir + croped_img_name, img)
            label_1_counter += 1
        elif label == 2:
            croped_img_name = str(label_2_counter) + ".jpg"
            cv2.imwrite(label_2_dir + croped_img_name, img)
            label_2_counter += 1

filenames_label_0 = [f for f in listdir(label_0_dir) if isfile(join(label_0_dir, f))]
filenames_label_1 = [f for f in listdir(label_1_dir) if isfile(join(label_1_dir, f))]
filenames_label_2 = [f for f in listdir(label_2_dir) if isfile(join(label_2_dir, f))]

print("Total number of images: " + str(len(filenames_label_0) + len(filenames_label_1)))
print("Number of images labeled 0: " + str(len(filenames_label_0)))
print("Number of images labeled 1: " + str(len(filenames_label_1)))
print("Number of images labeled 2: " + str(len(filenames_label_2)))
