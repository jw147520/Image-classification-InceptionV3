import Augmentor

p = Augmentor.Pipeline("./aug_test")
num_images = len(p.augmentor_images)


p.rotate90(probability=1)
p.sample(num_images)

p.rotate180(probability=1)
p.sample(num_images)

p.rotate270(probability=1)
p.sample(num_images)

p.flip_left_right(probability=1)
p.sample(num_images)


p.flip_top_bottom(probability=1)
p.sample(num_images)

p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=1)
p.sample(num_images*3)
