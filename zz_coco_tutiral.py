from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


# 1.coco dataset load! 
dataDir='/usr/src/data/coco'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories

# coco dataset categories 출력하기 
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# coco dataset 음식 categories 출력하기 
cat_food = set([cat['name'] for cat in cats if cat['supercategory'] == 'food'])
print()
print(f"food category where supercategory is food: \n {' '.join(cat_food)}")

# get all images containing given categories, select one at random - 도넛의 category id를 얻는다. 
catIds = coco.getCatIds(catNms=['donut']);
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# print(catIds)
imgIds = coco.getImgIds(catIds=catIds );
# print(imgIds)
# imgIds = coco.getImgIds(imgIds = [324158])
# print(imgIds)
print(imgIds[np.random.randint(0,len(imgIds))])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)


# load and display
plt.imshow(I); plt.axis('off')

# get annotation 
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
for ann in anns:
    print(ann)
# print(anns)
coco.showAnns(anns)
plt.show()

# print(catIds)
# print(annIds)
# print(anns)
# anns_img = np.zeros((img['height'], img['width']))
# for ann in anns:
# #     print(ann)
#     anns_img = np.maximum(anns_img, coco.annToMask(ann) )
# #     print(coco.annToMask(ann))
# #     print(ann['category_id'])
#     plt.imshow(anns_img)
# #     print(anns_img)

