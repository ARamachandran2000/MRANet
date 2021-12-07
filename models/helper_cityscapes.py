from glob import glob
import os
import random
import scipy.misc
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import time
import tensorflow as tf
# tf.disable_v2_behavior()
from urllib.request import urlretrieve
from tqdm import tqdm
import os.path
import shutil
import zipfile
import cv2



# Assuming RGB Format

Label = namedtuple('Label', ['name', 'color'])
# label_classes = [
# 	Label('car', (0,  0,142)),
# 	Label('person', (220, 20, 60))]
label_classes = [
    Label('unlabelled', (0, 0, 0)),
    Label('undamaged', (0, 255, 0)),
    Label('medium_damage', (0, 0, 255)),
    Label('high_damage', (255, 0, 0))

]


def load_data(image_paths, label_paths):
    image_files = glob(image_paths +  '*.png')
    label_files = glob(label_paths + '*.png')

    gt_images = []
    for img in image_files:
        img_base = os.path.basename(img)
        label_base = img_base
        label = label_paths + label_base

        gt_images.append(label)

    return image_files, gt_images


def gen_batches_fn(img_shape, image_paths, label_paths):
    def get_batches_fn(batch_size):

        image_files = glob(image_paths + '*.png')
        label_files = glob(label_paths + '*.png')
        print(image_files)

        gt_images, train_images = [], []
        for img in image_files:  # [0:79]
            img_base = os.path.basename(img)
            img_city = os.path.basename(os.path.dirname(img))
            label_base = img_base
            # Changing the last term in the training images to the label base because they have the same name up to that point.
            label = label_paths + label_base

            train_images.append(img)
            gt_images.append(label)

        train_image_paths, gt_image_paths = shuffle(train_images, gt_images)

        for batch_i in range(0, len(train_image_paths), batch_size):

            train_images, gt_images = [], []

            for img, label in zip(train_image_paths[batch_i:batch_i + batch_size],
                                  gt_image_paths[batch_i:batch_i + batch_size]):
                # print(img, label)
                image = scipy.misc.imresize(scipy.misc.imread(img), img_shape)
                #cv2.imwrite("/content/inp.png",image)
                gt_image = scipy.misc.imresize(scipy.misc.imread(label, mode='RGB'), img_shape)
                #cv2.imwrite("/content/gt.png",gt_image)
                label_bg = np.zeros([img_shape[0], img_shape[1]], dtype=bool)
                # plt.imshow(image)
                # plt.show()
                # plt.imshow(gt_image)
                # plt.show()
                label_list = []
                for l in label_classes[1:]:
                    current_class = np.all(gt_image == np.array(l.color), axis=2)
                    label_bg = current_class | label_bg
                    label_list.append(current_class)
                # plt.imshow(label_bg)
                # plt.show()

                # ~ changes 0 to 1 and 1 to 0 so we find everything else not considered a class and stack this
                # onto the label_list.
                label_bg = ~label_bg
                # Now we stack labels depth wise. For example, 2 classes would result in a shape (256, 512, 3) where
                # each depth slice (pixel) might look like [False, False, True] or [0, 0, 1].
                label_all = np.dstack([label_bg, *label_list])

                train_images.append(image)
                gt_images.append(label_all)
                #print(train_images[0].shape,gt_images[0].shape)

            yield np.array(train_images), np.array(gt_images)

    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, image_test, gt_test, image_shape, label_colors):
    for f in image_test:
        image_file = f
        print("image_file= ",image_file)
        gt_image_file = gt_test[0]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        labels = sess.run([tf.argmax(tf.nn.softmax(logits), axis=-1)], {keep_prob: 1.0, image_pl: [image]})
        #print("Labels_shape = ", labels[0].shape, len(labels),gt_image.shape)
        labels = labels[0].reshape(image_shape[0], image_shape[1])
        #print("Labels_shape = ", labels.shape)
        labels_colored = np.zeros((128,128,4))
        #print(labels_colored.shape)
        for lab in label_colors:
            label_mask = labels == lab
            #print(label_mask)
            #print(*label_colors[lab])
            labels_colored[label_mask] = np.array([*label_colors[lab],255])

        mask = scipy.misc.toimage(labels_colored, mode="RGBA")
        #print(labels_colored.shape)
        #cv2.imwrite("check.png",labels_colored)
#         init_img = scipy.misc.toimage(image)
#         init_img.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(mask)


def save_inference_samples(runs_dir, image_test, gt_test, sess, image_shape, logits, keep_prob, input_image,
                           label_colors):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, image_test, gt_test, image_shape,
                                    label_colors)
    for name, image in image_outputs:
        print(name)
        scipy.misc.imsave(os.path.join(output_dir, name), image)


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
