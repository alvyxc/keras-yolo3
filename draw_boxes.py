import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import random

#image_path = "/Users/xc573a/personws/keras-yolo3/open-image-dataset/test/8d7334f97cfced13.jpg"
#boxes = [9,179,16,213,338,14,165,22,202,338,15,155,24,174,338,20,188,31,229,338,28,155,34,187,338,34,188,
#         45,227,338,35,155,43,189,338,44,142,99,215,338,54,191,64,229,338,59,105,68,137,338,79,181,90,235,338,
#         86,103,95,141,338,92,140,100,172,338,92,175,106,241,338,100,144,156,192,338,107,171,116,201,338,110,178,
#         119,218,338,119,170,131,235,338,130,187,137,217,338,140,138,150,165,338,142,183,155,238,338,149,143,157,164,
#         338,157,136,166,174,338,159,102,168,142,338,163,190,176,235,338,167,151,174,183,338,175,145,182,176,338,183,104,
#         203,138,338,183,187,197,235,338,184,146,191,174,338,195,148,205,186,338,205,187,222,242,338,207,147,215,190,338,211,
#         110,221,145,338,217,155,225,191,338,221,178,232,221,338,228,158,236,191,338]

#image_path = "/Users/xc573a/personws/keras-yolo3/open-image-dataset/test/8db756648142a44a.jpg"
#boxes = [98,155,132,225,371,18,37,33,71,371,31,94,49,138,371,229,109,255,159,371,0,148,43,255,63,22,14,244,228,63,175,23,255,161,63,205,25,255,62,63]
image_path = "/Users/xc573a/personws/keras-yolo3/open-image-dataset/test/8da4eaf31e51d314.jpg"
boxes = [0,125,30,203,488,52,19,220,251,488]


axis_font = {'fontname':'Arial', 'size':'8'}

lines = [line.rstrip('\n') for line in open('model_data/openimgs_classes.txt')]
im = np.array(Image.open(image_path), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

color=['r', 'b', 'y', 'w', "y"]
# Create a Rectangle patch
for i in range(0, len(boxes), 5):
    c = random.choice (color)
    rect = patches.Rectangle((boxes[i],boxes[i+1]),boxes[i+2]-boxes[i],boxes[i+3]-boxes[i+1],linewidth=2,edgecolor=c,facecolor='none',
                             label='label')
    # Add the patch to the Axes
    plt.text(boxes[i], boxes[i+1], lines[boxes[i+4]], color=c, **axis_font)
    ax.add_patch(rect)

plt.show()

