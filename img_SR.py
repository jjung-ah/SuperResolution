import os, sys

img_id = sys.argv[1] 


### SR for Lenna
Lenna_command1 = 'ffmpeg_opencv -i /home/super/chloe_sr/chloe_1c/image/Lenna.png '
Lenna_command1 = Lenna_command1.replace('\n', ' ').replace('\r', '')
Lenna_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_1c/chloe_reshape/{}.pb /home/super/chloe_sr/chloe_1c/chloe_reshape/Lenna_{}.png -y'.format(img_id, img_id)
Lenna_command2 = Lenna_command2.replace('\n', ' ').replace('\r', '')
Lenna_command = Lenna_command1 + Lenna_command2

# print(Lenna_command)
os.system(Lenna_command)

### SR for bird
bird_command1 = 'ffmpeg_opencv -i /home/super/chloe_sr/chloe_1c/image/bird.png '
bird_command1 = bird_command1.replace('\n', ' ').replace('\r', '')
bird_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_1c/chloe_reshape/{}.pb /home/super/chloe_sr/chloe_1c/chloe_reshape/bird_{}.png -y'.format(img_id, img_id)
bird_command2 = bird_command2.replace('\n', ' ').replace('\r', '')
bird_command = bird_command1 + bird_command2

# print(bird_command)
os.system(bird_command)




### SR for tiger
tiger_command1 = 'ffmpeg -i /home/chloe/amu/wdsr-model/pixel_tiger.png '
tiger_command1 = tiger_command1.replace('\n', ' ').replace('\r', '')
tiger_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/chloe/amu/wdsr-model/{}.pb /home/chloe/amu/model_test/tiger/tiger_{}.png -y'.format(img_id, img_id)
tiger_command2 = tiger_command2.replace('\n', ' ').replace('\r', '')
tiger_command = tiger_command1 + tiger_command2

# print(tiger_command)
#os.system(tiger_command)



### SR for cat
cat_command1 = 'ffmpeg -i /home/chloe/amu/wdsr-model/pixel_cat.png '
cat_command1 = cat_command1.replace('\n', ' ').replace('\r', '')
cat_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/chloe/amu/wdsr-model/{}.pb /home/chloe/amu/model_test/cat/cat_{}.png -y'.format(img_id, img_id)
cat_command2 = cat_command2.replace('\n', ' ').replace('\r', '')
cat_command = cat_command1 + cat_command2

# print(cat_command)
#os.system(cat_command)



### SR for effel
eiffel_command1 = 'ffmpeg -i /home/super/chloe_sr/test_img/eiffel1.jpg '
eiffel_command1 = eiffel_command1.replace('\n', ' ').replace('\r', '')
eiffel_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_1c/chloe_reshape/{}.pb /home/super/chloe_sr/chloe_1c/chloe_reshape/eiffel_{}.png -y'.format(img_id, img_id)
eiffel_command2 = eiffel_command2.replace('\n', ' ').replace('\r', '')
eiffel_command = eiffel_command1 + eiffel_command2

# print(eye_command)
os.system(eiffel_command)



### SR for night
night_command1 = 'ffmpeg_opencv -i /home/super/chloe_sr/test_img/nightview2.png '
night_command1 = night_command1.replace('\n', ' ').replace('\r', '')
night_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_1c/chloe_reshape/{}.pb /home/super/chloe_sr/chloe_1c/chloe_reshape/nightview2_{}.png -y'.format(img_id, img_id)
night_command2 = night_command2.replace('\n', ' ').replace('\r', '')
night_command = night_command1 + night_command2

# print(Lenna_command)
os.system(night_command)


### SR for chameleon
chamel_command1 = 'ffmpeg_opencv -i /home/super/chloe_sr/test_img/chameleon.png '
chamel_command1 = chamel_command1.replace('\n', ' ').replace('\r', '')
chamel_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_1c/chloe_reshape/{}.pb /home/super/chloe_sr/chloe_1c/chloe_reshape/chameleon_{}.png -y'.format(img_id, img_id)
chamel_command2 = chamel_command2.replace('\n', ' ').replace('\r', '')
chamel_command = chamel_command1 + chamel_command2

# print(Lenna_command)
os.system(chamel_command)



### SR for italy
italy_command1 = 'ffmpeg_opencv -i /home/super/chloe_sr/test_img/italy.png '
italy_command1 = italy_command1.replace('\n', ' ').replace('\r', '')
italy_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/chloe_sr/chloe_test/{}.pb /home/super/chloe_sr/wdsr_img/italy_{}.png -y'.format(img_id, img_id)
italy_command2 = italy_command2.replace('\n', ' ').replace('\r', '')
italy_command = italy_command1 + italy_command2

# print(Lenna_command)
# os.system(italy_command)


### SR for xray
xray_command1 = 'ffmpeg -i /home/super/chloe_sr/chloe_1c/image/avg_diff2.png '
xray_command1 = xray_command1.replace('\n', ' ').replace('\r', '')
xray_command2 = '-vf sr=dnn_backend=tensorflow:model=/home/super/wdsr-model/{}.pb /home/super/chloe_sr/chloe_1c/avg_diff2_{}.png -y'.format(img_id, img_id)
xray_command2 = xray_command2.replace('\n', ' ').replace('\r', '')
xray_command = xray_command1 + xray_command2

# print(xray_command)
#os.system(xray_command)

#################### Not enough memory TT #####################
### SR for video(Mcountdown)
# mc1 = 'ffmpeg -hwaccel nvdec -hwaccel_device 0 -i /home/joeng/amu/model_test/McoundDown/Mcountdown_EPI0642_1080.mp4 -ss 1800 -t 10 -vsync 0 '
# mc1 = mc1.replace('\n', ' ').replace('\r', '')
# mc2 = '-vf sr=dnn_backend=tensorflow:model=/home/joeng/amu/wdsr-model/{}.pb -c:v h264_nvenc -gpu 0 -preset slow -acodec copy -minrate 44M -bufsize 50M -maxrate 50M -b:v 44M -pix_fmt yuv444p16le /home/joeng/amu/model_test/McoundDown/MC_{}.mp4 -y'.format(img_id, img_id)
# mc2 = mc2.replace('\n', ' ').replace('\r', '')
# mc = mc1 + mc2

# os.system(mc)



### SR for video(1987)
# mv1 = 'ffmpeg -hwaccel nvdec -hwaccel_device 0 -i /home/joeng/amu/model_test/m1987/1987_1920x1080_5000k_PC.mp4 -ss 7338 -t 10 -vsync 0 '
# mv1 = mv1.replace('\n', ' ').replace('\r', '')
# mv2 = '-vf sr=dnn_backend=tensorflow:model=/home/joeng/amu/wdsr-model/{}.pb -c:v h264_nvenc -gpu 0 -preset slow -acodec copy -minrate 44M -bufsize 50M -maxrate 50M -b:v 44M -pix_fmt yuv444p16le /home/joeng/amu/model_test/m1987/m1987_{}.mp4 -y'.format(img_id, img_id)
# mv2 = mv2.replace('\n', ' ').replace('\r', '')
# mv = mv1 + mv2

# os.system(mv)
