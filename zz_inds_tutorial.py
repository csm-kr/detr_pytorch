from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image

# 1.coco dataset load! 
dataDir='/usr/src/data/industrial/DataSet/Test_Dataset'
annFile='{}/coco_annotations.json'.format(dataDir)

# initialize COCO api for instance annotations
coco=COCO(annFile)
print(coco)

# display COCO categories and supercategories

# coco dataset categories 출력하기 
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# coco dataset 음식 categories 출력하기 
cat_food = set([cat['name'] for cat in cats if cat['supercategory'] == 'coco_annotations'])
print() # for blank
print(f"food category where supercategory is coco_annotations: \n {' '.join(cat_food)}")

# get all images containing given categories, select one at random - 도넛의 category id를 얻는다. 
catIds = coco.getCatIds(catNms=['star']);
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# print(catIds)
imgIds = coco.getImgIds(catIds=catIds);
# print(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])
# print(imgIds)
print(imgIds[np.random.randint(0,len(imgIds))])
img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
# print("coco_url :", img['coco_url'])

# load industrial image
ind_image = Image.open(dataDir + '/' + img['file_name'])
ind_image = np.array(ind_image)

# # load and display
# plt.imshow(I); plt.axis('off')

# get annotation 
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
# for ann in anns:
#     print(ann)
# # print(anns)
# coco.showAnns(anns)

print(catIds)
print(annIds)
print(anns)
anns_img = np.zeros((img['height'], img['width']))
for ann in anns:
#     print(ann)
    anns_img = np.maximum(anns_img, coco.annToMask(ann) )
#     print(coco.annToMask(ann))
#     print(ann['category_id'])
    # plt.imshow(anns_img)
#     print(anns_img)

print(anns_img)
print('a')

bimask_img = anns_img.astype(np.bool8)

# 1. load background image
# plt.figure('background')
# B = io.imread(dataDir + '/background.jpg')
# plt.imshow(B)
# plt.show()

# 1. load background image from PIL

plt.figure('cropped image')
bg_im = Image.open(dataDir + '/background.jpg')

# 2. random crop
w, h = bg_im.size
import random 
new_x1 = random.randint(0, w-540)
new_y1 = random.randint(0, h-540)
new_x2 = new_x1 + 540
new_y2 = new_y1 + 540

bg_im = bg_im.crop((new_x1, new_y1, new_x2, new_y2))

# 224, 224 로 resize 하는부분 
bg_im = bg_im.resize((224, 224))
bg_im_np = np.array(bg_im)


hyp_im = np.zeros_like(bg_im_np)
hyp_im[~bimask_img] = bg_im_np[~bimask_img]


hyp_im[bimask_img] = ind_image[bimask_img]

plt.imshow(hyp_im)
plt.show()

# from skimage.util import crop
# C = crop(B, ((50, 100), (50, 50), (0,0)), copy=False)
# plt.figure('cropped image')
# plt.imshow(C)
# plt.show()
# 2. crop the random 


# plt.show()