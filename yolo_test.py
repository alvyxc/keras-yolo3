import os
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


FLAGS = None
#TEST_DIR = 'open-image-dataset/kaggle-2019-object-detection/test_challenge_2019'
TEST_DIR = 'open-image-dataset/test'
OUTPUT_CSV = 'submit/output.csv'


def detect_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        sys.exit('Cannot open image file: {}'.format(img_path))
    else:
        r_image = yolo.detect_image(image)
        r_image.show()


def detect_test_imgs(yolo):
    global FLAGS
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    if FLAGS.shuffle:
        from random import shuffle
        shuffle(jpgs)
    for jpg in jpgs:
        img_path = os.path.join(TEST_DIR, jpg)
        detect_img(yolo, img_path)
        str_in = input('{}, <ENTER> for next or "q" to quit: '.format(img_path))
        if str_in.lower() == 'q':
            break
    yolo.close_session()


def infer_img(yolo, img_path):
    try:
        image = Image.open(img_path)
    except:
        #sys.exit('Cannot open image file: {}'.format(img_path))
        print('!!! Cannot open image file: {}'.format(img_path))
        return []
    else:
        return yolo.infer_image(image)


def submit_test_imgs(yolo):
    jpgs = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
    os.makedirs(os.path.split(OUTPUT_CSV)[0], exist_ok=True)
    with open(OUTPUT_CSV, 'w') as f:
        f.write('ImageId,PredictionString\n')
        for jpg in jpgs:
            img_path = os.path.join(TEST_DIR, jpg)
            boxes = infer_img(yolo, img_path)
            f.write('{},'.format(os.path.splitext(jpg)[0]))
            # 1 record: [label, confidence, x_min, y_min, x_max, y_max]
            box_strings = ['{:s} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(b[0], b[1], b[2], b[3], b[4], b[5]) for b in boxes]
            if box_strings:
                f.write(' '.join(box_strings))
            f.write('\n')
    yolo.close_session()


def detect_single_img(yolo, img_path):
    detect_img(yolo, img_path)
    boxes = infer_img(yolo, img_path)
    print('ImageId,PredictionString\n')
    print('{},'.format(img_path))
    # 1 record: [label, confidence, x_min, y_min, x_max, y_max]
    box_strings = ['{:s} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}'.format(b[0], b[1], b[2], b[3], b[4], b[5]) for b in boxes]
    if box_strings:
        print(' '.join(box_strings))
    print('\n')


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--score', type=float,
        help='score (confidence) threshold, default ' + str(YOLO.get_defaults("score"))
    )

    parser.add_argument(
        '--shuffle', default=False, action="store_true",
        help='shuffle images for display mode'
    )

    parser.add_argument(
        '--display', default=False, action="store_true",
        help='display mode, to show inferred images with bounding box overlays'
    )

    parser.add_argument(
        '--submit', default=False, action="store_true",
        help='submit mode, to generate "output.csv" for Kaggle submission'
    )

    parser.add_argument(
        '--image', default="", action='store', type=str, help='single image path')


    FLAGS = parser.parse_args()

    if FLAGS.display:
        print("Display mode")
        if (FLAGS.image is not None) and (FLAGS.image != ""):
            detect_single_img(YOLO(**vars(FLAGS)), FLAGS.image)
        else:
            detect_test_imgs(YOLO(**vars(FLAGS)))
    elif FLAGS.submit:
        print("Submit mode: writing to output.csv")
        submit_test_imgs(YOLO(**vars(FLAGS)))
    else:
        print("Please specify either Display or Submit mode.")
