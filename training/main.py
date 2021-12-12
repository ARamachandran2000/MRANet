from helper import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import numpy as np
import cv2
from MRA_Model import *

## Global variables ##
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4


# Used to Load KEEP_PROB and LEARNING_RATE
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


# Activation Functions Used in Model : ReLu and Softmax
def create_activation():
    return layers.ReLU()
def sigmoid_activation():
    return layers.Softmax()

def create_model():

    # Placeholder for input image of size (128,128,3)
    input_1 = tf.placeholder(tf.float32, [1, 128, 128, 3], name='input')
    
    # Encoder block of MRANet
    enc_1 = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(input_1)
    enc_1 = create_activation()(enc_1)
    enc_1 = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(enc_1)
    enc_1 = create_activation()(enc_1)    #This will be used in skip connection 1

    maxpool_1 = layers.MaxPooling2D((2,2))(enc_1)


    ######################################
    '''
    Take the DL1 block here and concatenate with the downstream layer
    '''

    # Calculate Level-1 MRA Decomposition
    w = WaveTFFactory().build('db2', dim=2)

    dl1_input = w.call(tf.expand_dims(input_1[:,:,:,0],axis=-1)) #Calculate Wavelet Decomposition on only gray scale image

    dl1_enc = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(dl1_input)
    dl1_enc = create_activation()(dl1_enc)
    dl1_enc = layers.Conv2D(64,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(dl1_enc)
    dl1_enc = create_activation()(dl1_enc)


    #Concatenate with main mode i.e output of maxpool_1

    concat_1 = tf.concat([maxpool_1,dl1_enc],axis=-1)

    enc_2 = layers.Conv2D(128,kernel_size=(3,3),padding="same",kernel_initializer = tf.random_normal_initializer(stddev=0.01))(concat_1)
    enc_2 = create_activation()(enc_2)
    enc_2 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_2)
    enc_2 = create_activation()(enc_2) #This will be used in skip connection 2

    maxpool_2 = layers.MaxPooling2D((2,2))(enc_2)

    ######################################
    '''
    Take the DL2 block here and concatenate with the downstream layer
    '''
    # Calculate Level-2 MRA Decomposition

    dl2_input = w.call(tf.expand_dims(dl1_input[:, :, :, 0], axis=-1))  # Calculate Wavelet Decomposition on only channel 0 of dl1_input
    dl2_enc = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dl2_input)
    dl2_enc = create_activation()(dl2_enc)
    dl2_enc = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dl2_enc)
    dl2_enc = create_activation()(dl2_enc)



    concat_2 = tf.concat([maxpool_2,dl2_enc],axis=-1)

    enc_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_2)
    enc_3 = create_activation()(enc_3)
    enc_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_3)
    enc_3 = create_activation()(enc_3)  # This will be used in skip connection 3
    maxpool_3 = layers.MaxPooling2D((2, 2))(enc_3)

    enc_4 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(maxpool_3)
    enc_4 = create_activation()(enc_4)
    enc_4 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_4)
    enc_4 = create_activation()(enc_4)  # This will be used in skip connection 4
    maxpool_4 = layers.MaxPooling2D((2, 2))(enc_4)

    enc_5 = layers.Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(maxpool_4)
    enc_5 = create_activation()(enc_5)
    enc_5 = layers.Conv2D(1024, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(enc_5)
    enc_5 = create_activation()(enc_5)

    #####-> Upsampling Blocks , MRANet Decoder

    upsampling_1 = layers.UpSampling2D((2,2))(enc_5)
    dec_1 = layers.Conv2D(512,kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_1)
    dec_1 = create_activation()(dec_1)

    concat_3 = tf.concat([enc_4,dec_1],axis = -1)
    dec_2 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_3)
    dec_2 = create_activation()(dec_2)
    dec_2 = layers.Conv2D(512, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_2)
    dec_2 = create_activation()(dec_2)

    upsampling_2 = layers.UpSampling2D((2,2))(dec_2)
    dec_3 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_2)
    dec_3 = create_activation()(dec_3)

    concat_4 = tf.concat([enc_3,dec_3],axis=-1)
    dec_4 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_4)
    dec_4 = create_activation()(dec_4)
    dec_4 = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_4)
    dec_4 = create_activation()(dec_4)

    upsampling_3 = layers.UpSampling2D((2, 2))(dec_4)
    dec_5 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_3)
    dec_5 = create_activation()(dec_5)

    concat_5 = tf.concat([enc_2,dec_5],axis=-1)
    dec_6 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_5)
    dec_6 = create_activation()(dec_6)
    dec_6 = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_6)
    dec_6 = create_activation()(dec_6)

    upsampling_4 = layers.UpSampling2D((2, 2))(dec_6)
    dec_7 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(upsampling_4)
    dec_7 = create_activation()(dec_7)

    concat_6 = tf.concat([enc_1, dec_7], axis=-1)
    dec_8 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(concat_6)
    dec_8 = create_activation()(dec_8)
    dec_8 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_8)
    dec_8 = create_activation()(dec_8)

    dec_9 = layers.Conv2D(4, kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_8)
    dec_9 = create_activation()(dec_9)
    final_output = layers.Conv2D(4 , kernel_size=(3, 3), padding="same", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(dec_9)
    #final_output = sigmoid_activation()(final_output)


    #model = models.Model(input_1,final_output)

    return input_1,final_output

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
    print(logits.shape)
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
    Trains the  neural network and prints out the loss during training.
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
    print('Done')

def run():
    '''
        Top-Level Model
        Creates the session variable and calls train

    '''
    # Load Data
    data_folder = '../dataset/'
    image_paths = data_folder + 'Final_Images_Final/'
    label_paths = data_folder + 'Final_Labels_Final_2/'
    vgg_path = '../data/vgg/'
    runs_dir = './runs_city'
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_classes)}
    print("Labels = ",label_colors)
    maybe_download_pretrained_vgg('../data')
    # train_image_paths, gt_image_paths = load_data(image_paths, label_paths, data_type='train')

    # Training Paramaters
    batch_size = 1
    img_shape = (128, 128)
    num_classes = 4
    learning_rate = 1e-4
    epochs = 51  #Change this

    with tf.Session() as sess:
        ## Construct Network ##
        _, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        #deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        input_1,deconv_output = create_model()
        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

        print("Starting Training...")
        get_batches_fn = gen_batches_fn(img_shape, image_paths, label_paths)

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_1, correct_label,
                 keep_prob, learning_rate)
        
        save_file = './model1.ckpt'
        saver = tf.train.Saver()
        saver.save(sess, save_file)

        ### Testing ###
        image_test, gt_test = load_data(image_paths, label_paths)
        data_dir = data_folder + 'Final_Images_Final/' + '*.png'
        print("Here")
        # save_inference_samples(runs_dir, data_dir, sess, img_shape, logits, keep_prob, input_image, label_colors)
        save_inference_samples(runs_dir, image_test, gt_test, sess, img_shape, logits, keep_prob, input_1,
                               label_colors)

def continue_training():
    # Loads Data from checkpoint and continues training , Also used to run inference
    data_folder = '../dataset/'
    image_paths = data_folder + 'Final_Images_Final/'
    label_paths = data_folder + 'Final_Labels_Final_2/'
    vgg_path = '../data/vgg/'
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
        _, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        #deconv_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        input_1,deconv_output = create_model()
        # placeholder for labels, shape=(128, 256, 512, 30). X placeholder is input_image found above.
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        ## Optimize Network ##
        logits, train_op, cross_entropy_loss = optimize(deconv_output, correct_label, learning_rate, num_classes)

        # Initialize variables
        sess.run(tf.global_variables_initializer())

#         get_batches_fn = gen_batches_fn(img_shape, image_paths, label_paths, data_type='train')

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
        save_inference_samples(runs_dir, image_test, gt_test, sess, img_shape, logits, keep_prob, input_1,
                               label_colors)

# Main Block
if __name__ == '__main__':
#     run()
    continue_training()
