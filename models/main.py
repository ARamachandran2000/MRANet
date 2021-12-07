from helper import *
import tensorflow as tf
# tf.disable_v2_behavior()
import time
import numpy as np
import cv2

## Global variables ##
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4


def load_vgg(sess, vgg_path):
    # Load vgg from path
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    # Extract graph and tensors that we need to manipulate for deconvolutions
    # We also want input, keepprob, layer3, 4 and 7 outputs for this particular FCN.
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3_out = graph.get_tensor_by_name('layer3_out:0')
    layer4_out = graph.get_tensor_by_name('layer4_out:0')
    layer7_out = graph.get_tensor_by_name('layer7_out:0')

    # To find and print tensor names:
    # for op in graph.get_operations():
    # 	print(str(op.name))

    return input_image, keep_prob, layer3_out, layer4_out, layer7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network. We already have the encoder part based on vgg.
    Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 convolution for layer 7 from vgg
    layer7_conv_1x1 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # First deconvolution layer with layer7 (after 1x1 convolution) as input
    deconv_layer1 = tf.layers.conv2d_transpose(
        layer7_conv_1x1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))  # Stride amount is cause upsampling by 2

    # 1x1 convolution for layer 4 from vgg
    layer4_conv_1x1 = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Adding deconvolved layer 1 and layer 4 for first skip connection.
    skip_connection1 = tf.add(layer4_conv_1x1, deconv_layer1)

    # 1x1 convolution for layer 3 for skip connection 2
    layer3_conv_1x1 = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Second deconvolution layer
    deconv_layer2 = tf.layers.conv2d_transpose(
        skip_connection1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # Second skip connection made up of second deconvolution layer and 1x1 convolution of layer 3
    skip_connection2 = tf.add(deconv_layer2, layer3_conv_1x1)

    # Final deconvolution layer to reconstruct image
    deconv_output_layer = tf.layers.conv2d_transpose(
        skip_connection2,
        num_classes,
        kernel_size=16,
        strides=8,
        padding="same",
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    return deconv_output_layer


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # logits and labels are now 2D tensors where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # Computes softmax cross entropy between logits and labels
    cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)

    # Computes the mean of elements across dimensions of a tensor
    cross_entropy_loss = tf.reduce_mean(cross_entropy_logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Minimizes loss by combining calls compute_gradients() and apply_gradients().
    train_op = optimizer.minimize(cross_entropy_loss, name='train_op')

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    total_loss = []
    for epoch in range(epochs):

        start_time = time.time()
        loss = None
        batch_num = 0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: image,
                           correct_label: label,
                           keep_prob: KEEP_PROB,
                           learning_rate: LEARNING_RATE})
            batch_num += 1
            total_loss.append(loss)
            print("Batch {0} Loss {1} Time {2}".format(batch_num, loss, time.time() - start_time))

        print("[Epoch: {0}/{1} Loss: {2} Time: {3}]".format(epoch + 1, epochs, loss, time.time() - start_time))
        if epoch % 1 == 0:
            save_file = 'saved_models/model_latest.ckpt'
            saver = tf.train.Saver()
            saver.save(sess, save_file)

    end_time = time.time() - start_time


def run():
    # Load Data
    data_folder = 'dataset/'
    image_paths = data_folder + 'Final_Images_Final/'
    label_paths = data_folder + 'Final_Labels_Final_2/'
    vgg_path = './data/vgg/'
    runs_dir = './runs_city'
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_classes)}
    print("Labels = ",label_colors)
    maybe_download_pretrained_vgg('./data')
    # train_image_paths, gt_image_paths = load_data(image_paths, label_paths, data_type='train')

    # Training Paramaters
    batch_size = 32
    img_shape = (128, 128)
    num_classes = 4
    learning_rate = 1e-4
    epochs = 51  #Change this

    with tf.Session() as sess:
        ## Construct Network ##
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        print("Starting Training...")
        get_batches_fn = gen_batches_fn(img_shape, image_paths, label_paths)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate)
        
        save_file = './model1.ckpt'
        saver = tf.train.Saver()
        saver.save(sess, save_file)

        ### Testing ###
        image_test, gt_test = load_data(image_paths, label_paths)
        data_dir = data_folder + 'Final_Images_Final/' + '*.png'
        # save_inference_samples(runs_dir, data_dir, sess, img_shape, logits, keep_prob, input_image, label_colors)
        save_inference_samples(runs_dir, image_test, gt_test, sess, img_shape, logits, keep_prob, input_image,
                               label_colors)


# +
def continue_training():
    # Load Data
    data_folder = 'dataset/'
    image_paths = data_folder + 'Final_Images_Final/'
    label_paths = data_folder + 'Final_Labels_Final_2/'
    vgg_path = './data/vgg/'
    runs_dir = './runs_city'
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_classes)}

    # Training Paramaters
    batch_size = 4
    img_shape = (128, 128)
    num_classes = 4
    learning_rate = 1e-4
    epochs = 4

    with tf.Session() as sess:

        ## Construct Network ##
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

#         get_batches_fn = gen_batches_fn(img_shape, image_paths, label_paths)

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./saved_models/'))
        graph = tf.get_default_graph()

#         for epoch in range(epochs):

#             total_loss = []
#             start_time = time.time()
#             loss = None
#             batch_num = 0
#             for image, label in get_batches_fn(batch_size):
#                 _, loss = sess.run(
#                     [train_op, cross_entropy_loss],
#                     feed_dict={input_image: image,
#                                correct_label: label,
#                                keep_prob: KEEP_PROB,
#                                learning_rate: LEARNING_RATE})
#                 batch_num += 1
#                 total_loss.append(loss)
#                 print("Batch {0} Loss {1} Avg.loss {2} Time {3}".format(batch_num, loss, np.mean(total_loss),
#                                                                         time.time() - start_time))

#             print("[Epoch: {0}/{1} Loss: {2} Time: {3}]".format(epoch + 1, epochs, np.mean(total_loss),
#                                                                       time.time() - start_time))

#         save_file = './model10.ckpt'
#         saver = tf.train.Saver()
#         saver.save(sess, save_file)

        ### Testing ###
        image_test, gt_test = load_data(image_paths, label_paths)
        data_dir = data_folder + 'Final_Images/' + '*.png'
        save_inference_samples(runs_dir, image_test, gt_test, sess, img_shape, logits, keep_prob, input_image,
                               label_colors)


# -

if __name__ == '__main__':
#     run()
    continue_training()


