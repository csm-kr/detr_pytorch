import os
import time
import glob
import torch
from PIL import Image
from models.model import DETR
from torchvision import transforms as tfs
from models.postprocessor import PostProcess
from utils.download_url_pretrained import download_pretrained_model
from utils.visualize_boxes import visualize_boxes_labels, visualize_cv2

def demo_image_transforms(demo_image):

    transform_demo = tfs.Compose([tfs.Resize((1024, 1024)),
                                  tfs.ToTensor(),
                                  tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    demo_image = transform_demo(demo_image)
    demo_image = demo_image.unsqueeze(0)      # 1, 3, H, W
    return demo_image


@ torch.no_grad()
def demo(demo_root, device=None, model=None):

    # 1. set post processors
    postprocessors = {'bbox': PostProcess()}

    # 2. make list of images
    demo_image_list = glob.glob(os.path.join(demo_root, '*' + '.jpg'))
    total_time = 0

    # 2. download and .pth file
    pth_name = "detr_coco_best"
    download_pretrained_model(pth_name, '1BfgWrkkX2v_d3sbLtIrguZTtIy-MRA5K')
    checkpoint = torch.load(f=os.path.join(torch.hub.get_dir(), 'checkpoints', pth_name),
                            map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tic = time.time()

    for idx, img_path in enumerate(demo_image_list):

        # --------------------- img load ---------------------
        demo_image_pil = Image.open(img_path).convert('RGB')
        demo_image = demo_image_transforms(demo_image_pil).to(device)

        w, h = demo_image_pil.size
        orig_target_sizes = torch.tensor([h, w]).unsqueeze(0).to(device)

        outputs = model(demo_image)
        pred_boxes, pred_labels, pred_scores = postprocessors['bbox'](outputs, orig_target_sizes, is_demo=True)

        keep = pred_scores[0] > 0.7

        visualize_cv2(image=demo_image.squeeze(0),
                      boxes=pred_boxes[0][keep],
                      labels=pred_labels[0][keep],
                      data_type='coco',
                      num_labels=91,
                      scores=pred_scores[0][keep],
                      original_h=h,
                      original_w=w,
                      name=os.path.basename(img_path))

        toc = time.time()
    
        inference_time = toc - tic
        total_time += inference_time
    
        if idx % 100 == 0 or idx == len(demo_image_list) - 1:
            # ------------------- check fps -------------------
            print('Step: [{}/{}]'.format(idx, len(demo_image_list)))
            print("fps : {:.4f}".format((idx + 1) / total_time))

    print("complete detection...!")
    return


def demo_worker():
    # 2. device
    device = torch.device('cuda:0')

    # 3. model
    model = DETR(num_classes=91, num_queries=100, d_model=256).cuda()
    model = torch.nn.DataParallel(module=model, device_ids=[0])

    demo(demo_root='/usr/src/data/cvml',
         device=device,
         model=model)

if __name__ == '__main__':
    demo_worker()