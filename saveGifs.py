import glob
import os 
from PIL import Image

import cv2

# filepaths
fp_in = "./envImages/obsMap_*.png"
fp_out = "./envImages/explore.gif"
fp_out_vid = "./envImages/explore.mp4"


# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif

image_list = []
#img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
for count, f in enumerate(sorted(glob.glob(fp_in))):
    
    if count % 100 == 0:
        img = Image.open(f)
        image_list.append(img)

        # img = cv2.imread(f)
        # height, width, layers = img.shape
        # size = (width,height)
        # image_list.append(img)
    #img.close()

# out = cv2.VideoWriter(fp_out_vid,cv2.VideoWriter_fourcc(*'MP4V'), 100, size)
 
# for i in range(len(image_list)):
#     out.write(image_list[i])
# out.release()


image_list[0].save(fp_out,
               save_all=True,
               append_images=image_list[1:],
               duration=300,
               loop=0)
print(len(image_list))

# imageio.mimsave("./envImages/explore.gif", image_list)


# png_dir = "./envImages/"
# images = []

# for count, file_name in enumerate(sorted(os.listdir(png_dir))):
#     #print(count)

#     if (file_name.endswith('.png') and count % 10 == 0):
#         file_path = os.path.join(png_dir, file_name)
#         images.append(imageio.imread(file_path))

# #kargs = { 'duration': 0.005 }
# imageio.mimsave("./envImages/explore.gif", images)