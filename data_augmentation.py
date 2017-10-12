# -*- coding: utf-8 -*-
# Data Augmentation : Augmentor library 사용.
# Augmentor : Image augmentation library in python for machine learning.
# https://github.com/mdbloice/Augmentor
import Augmentor

p = Augmentor.Pipeline("./aug_test")  # image file 들이 저장된 directory 를 넘겨준다.
num_images = len(p.augmentor_images)

# 90도 회전한 이미지
p.rotate90(probability=1)
p.sample(num_images)
# 180도 회전한 이미지
p.rotate180(probability=1)
p.sample(num_images)
# 270도 회전한 이미지
p.rotate270(probability=1)
p.sample(num_images)
# 좌우 뒤집은 이미지
p.flip_left_right(probability=1)
p.sample(num_images)
# 상하 뒤집은 이미지
p.flip_top_bottom(probability=1)
p.sample(num_images)
# random distortion 을 적용한 이미지
p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=1)
p.sample(num_images*3)


# 이미지 한 장당 6 장의 augmented images 가 생성되어 원래 data set 의 7배로 부풀림.
