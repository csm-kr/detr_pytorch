import os
import wget
import glob
from utils.util import bar_custom, xy_to_cxcy

# coco
import zipfile
# voc
import tarfile


def download_coco(root_dir='D:\data\\coco', remove_compressed_file=True):
    # for coco 2017
    coco_2017_train_url = 'http://images.cocodataset.org/zips/train2017.zip'
    coco_2017_val_url = 'http://images.cocodataset.org/zips/val2017.zip'
    coco_2017_test_url = 'http://images.cocodataset.org/zips/test2017.zip'
    coco_2017_trainval_anno_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

    os.makedirs(root_dir, exist_ok=True)
    anno_dir = os.path.join(root_dir, 'annotations')
    os.makedirs(anno_dir, exist_ok=True)

    """Download the COCO data if it doesn't exit in processed_folder already."""
    if (os.path.exists(os.path.join(root_dir, 'train2017')) and
            os.path.exists(os.path.join(root_dir, 'val2017'))):

        print("data already exist!")
        return

    print("Download...")

    # image download
    wget.download(url=coco_2017_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=coco_2017_val_url, out=root_dir, bar=bar_custom)
    print('')

    # annotation download
    wget.download(coco_2017_trainval_anno_url, out=root_dir, bar=bar_custom)
    print('')

    print("Extract...")
    # image extract
    with zipfile.ZipFile(os.path.join(root_dir, 'train2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))
    with zipfile.ZipFile(os.path.join(root_dir, 'val2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))

    # annotation extract
    with zipfile.ZipFile(os.path.join(root_dir, 'annotations_trainval2017.zip')) as unzip:
        unzip.extractall(os.path.join(root_dir))

    # remove zips
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))  # in root_dir remove *.zip
        for anno_zip in root_zip_list:
            os.remove(anno_zip)

        img_zip_list = glob.glob(os.path.join(root_dir, '*.zip'))  # in img_dir remove *.zip
        for img_zip in img_zip_list:
            os.remove(img_zip)
        print("Remove *.zips")

    print("Done!")


def download_voc(root_dir='D:\data\\voc', remove_compressed_file=True):

    voc_2012_train_url = 'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
    voc_2007_train_url = 'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
    voc_2007_test_url = 'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'

    os.makedirs(root_dir, exist_ok=True)

    """Download the VOC data if it doesn't exist in processed_folder already."""
    if (os.path.exists(os.path.join(root_dir, 'VOCtrainval_11-May-2012')) and
        os.path.exists(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007')) and
        os.path.exists(os.path.join(root_dir, 'VOCtest_06-Nov-2007'))):
        print("data already exist!")
        return

    print("Download...")

    wget.download(url=voc_2012_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_train_url, out=root_dir, bar=bar_custom)
    print('')
    wget.download(url=voc_2007_test_url, out=root_dir, bar=bar_custom)
    print('')

    os.makedirs(os.path.join(root_dir, 'VOCtrainval_11-May-2012'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'VOCtest_06-Nov-2007'), exist_ok=True)

    print("Extract...")

    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_11-May-2012.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtrainval_11-May-2012'))
    with tarfile.open(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtrainval_06-Nov-2007'))
    with tarfile.open(os.path.join(root_dir, 'VOCtest_06-Nov-2007.tar')) as tar:
        tar.extractall(os.path.join(root_dir, 'VOCtest_06-Nov-2007'))

    # remove tars
    if remove_compressed_file:
        root_zip_list = glob.glob(os.path.join(root_dir, '*.tar'))  # in root_dir remove *.zip
        for root_zip in root_zip_list:
            os.remove(root_zip)
        print("Remove *.tars")

    print("Done!")