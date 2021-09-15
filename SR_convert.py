import os
import sys
import glob
from tqdm import tqdm

# os.chdir('/home/rain/ffmpeg/')
# 원본 이미지들이 들어 있는 폴더
GT_path = '/home/super/chloe_sr/test_img2/'

# SR 된 이미지들이 들어갈 폴더( 없을 경우 알아서 만듬)
LR_path = '/home/super/chloe_sr/sr_img/'

# 사용할 모델 
model_nm = 'wdsr_div2k_test_x2_1c'

img_paths=[]
img_list = os.listdir(GT_path)

for extension in img_list:
    img_paths_ext = glob.glob(os.path.join(GT_path, '**', f'{extension}'), recursive=True)
    img_paths.extend(img_paths_ext)

for img_path in tqdm(img_paths):
        img_dir, img_file = os.path.split(img_path)
        img_id, img_ext = os.path.splitext(img_file)
        out_dir = os.path.join(LR_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        output_dir = os.path.join(out_dir, f'{img_id}.png')
        command = "ffmpeg_opencv -i {} -vf sr=dnn_backend=tensorflow:model=/home/joeng/amu/wdsr-model/{} {}".format(img_path,model_nm, output_dir)

        os.system(command)
