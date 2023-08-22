import os
import torch
import gdown


def download_pretrained_model(pth_name, file_id=None):
    google_path = 'https://drive.google.com/uc?id='
    if file_id is None:
        return None
    torch_dir = torch.hub.get_dir()
    output_name = pth_name
    if os.path.exists(os.path.join(torch_dir, 'checkpoints', output_name)):
        print("Already downloads!")
    else:
        gdown.download(google_path+file_id,
                       os.path.join(torch_dir, 'checkpoints', output_name),
                       quiet=False)