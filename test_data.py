import os
from options.test_options import TestOptions
# from data import CreateDataLoader
from models import create_model
# from util.visualizer import save_images
from util import html
import glob
from PIL import Image
import torchvision.transforms as transforms
from util.util import tensor2im
from tqdm import tqdm
import torch
if __name__ == '__main__':
    ### folder to save gen images
    save_folder = 'test_data_hand'
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
   
    #### folder frames
    img_folder = '/home/ubuntu/openpose/build/examples/tutorial_api_python/draw_kp/test_data_hand_76/train_img'
    img_path = glob.glob(img_folder+ '/*.png')
    ### folder hand images
    hand_folder = '/home/ubuntu/openpose/build/examples/tutorial_api_python/draw_kp/test_data_hand_76/train_left'
    hand_path = [i.replace(img_folder,hand_folder) for i in img_path]
    model = create_model(opt)
    model.setup(opt)
   
    if opt.eval:
        model.eval()

    for i in tqdm(range(len(img_path))):
        if i >= opt.how_many:
            break
        img_raw = Image.open(img_path[i]).convert('RGB')
        kp_raw = Image.open(img_path[i]).convert('RGB')
        img_raw = transforms.ToTensor()(img_raw)
        kp_raw = transforms.ToTensor()(kp_raw)
        data = torch.cat((img_raw.unsqueeze(0), kp_raw.unsqueeze(0)), 1)

        with torch.no_grad():
           out_put = model.netG(data)
        trans_img = tensor2im(out_put)
        trans_img = Image.fromarray(trans_img)
        name = img_path[i].split('/')[-1]
        trans_img.save(os.path.join(save_folder,name))
      