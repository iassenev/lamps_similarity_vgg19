#-----------------------------------------------------------------------------------------------------------------------
#   Date:           22.01.2019
#   Description:    Lamps distance generator on top of VGG-19 model
#-----------------------------------------------------------------------------------------------------------------------

print                   ( "initializing pathlib, urllib3, progressbar, scipy.io, matplotlib.pyplot, numpy and tensorflow" )

from pathlib import Path
import urllib3
import urllib3.contrib.pyopenssl
import progressbar
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

urllib3.contrib.pyopenssl.inject_into_urllib3( )

#-----------------------------------------------------------------------------------------------------------------------
# class CONFIG as a properties holder
#-----------------------------------------------------------------------------------------------------------------------
class CONFIG:
    # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition".
    MEANS               = np.array( [ 123.68, 116.779, 103.939 ] ).reshape( ( 1, 1, 1, 3 ) )
    IMAGE_WIDTH         = 400
    IMAGE_HEIGHT        = 300
    COLOR_CHANNELS      = 3

#-----------------------------------------------------------------------------------------------------------------------
# VGG-19 model loader
#-----------------------------------------------------------------------------------------------------------------------
def initialize_vgg_model( path ):
    """
    Returns a model for the purpose of 'painting' the picture.
    Takes only the convolution layer weights and wrap using the TensorFlow
    Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
    the paper indicates that using AveragePooling yields better results.
    The last few fully connected layers are not used.
    Here is the detailed configuration of the VGG model:
        0 is conv1_1 (3, 3, 3, 64)
        1 is relu
        2 is conv1_2 (3, 3, 64, 64)
        3 is relu    
        4 is maxpool
        5 is conv2_1 (3, 3, 64, 128)
        6 is relu
        7 is conv2_2 (3, 3, 128, 128)
        8 is relu
        9 is maxpool
        10 is conv3_1 (3, 3, 128, 256)
        11 is relu
        12 is conv3_2 (3, 3, 256, 256)
        13 is relu
        14 is conv3_3 (3, 3, 256, 256)
        15 is relu
        16 is conv3_4 (3, 3, 256, 256)
        17 is relu
        18 is maxpool
        19 is conv4_1 (3, 3, 256, 512)
        20 is relu
        21 is conv4_2 (3, 3, 512, 512)
        22 is relu
        23 is conv4_3 (3, 3, 512, 512)
        24 is relu
        25 is conv4_4 (3, 3, 512, 512)
        26 is relu
        27 is maxpool
        28 is conv5_1 (3, 3, 512, 512)
        29 is relu
        30 is conv5_2 (3, 3, 512, 512)
        31 is relu
        32 is conv5_3 (3, 3, 512, 512)
        33 is relu
        34 is conv5_4 (3, 3, 512, 512)
        35 is relu
        36 is maxpool
        37 is fullyconnected (7, 7, 512, 4096)
        38 is relu
        39 is fullyconnected (1, 1, 4096, 4096)
        40 is relu
        41 is fullyconnected (1, 1, 4096, 1000)
        42 is softmax
    """
    
    vgg = scipy.io.loadmat(path)

    vgg_layers = vgg['layers']
    
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for a given layer.
        """
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a TensorFlow layer. Expects a
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from the VGG
        model at 'layer'.
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases from the VGG
        model at 'layer'.
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool2d(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph                       = { }
    graph['input']              = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']            = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']            = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1']           = _avgpool(graph['conv1_2'])
    graph['conv2_1']            = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']            = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2']           = _avgpool(graph['conv2_2'])
    graph['conv3_1']            = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']            = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']            = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']            = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3']           = _avgpool(graph['conv3_4'])
    graph['conv4_1']            = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']            = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']            = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']            = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4']           = _avgpool(graph['conv4_4'])
    graph['conv5_1']            = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']            = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']            = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']            = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5']           = _avgpool(graph['conv5_4'])
    
    return                      graph

def download                ( url, file, caption ):
    urllib3.disable_warnings( urllib3.exceptions.InsecureRequestWarning )
    http                    = urllib3.PoolManager()
    response                = http.request( "GET", url, preload_content=False )
    file_size               = int( response.headers.get( "Content-Length" ) )

    widgets                 = [
        caption + " : ", progressbar.RotatingMarker( ),
        " ", progressbar.Percentage( ),
        " ", progressbar.Bar( marker = "=", left = "[", right = "]", fill_left = True ),
        " ", progressbar.ETA( ),
        " ", progressbar.FileTransferSpeed( ),
    ]
    progress_bar            = progressbar.ProgressBar( maxval = file_size, widgets = widgets )
    progress_bar.start      ( )

    downloading_file        = Path( str( file ) + ".downloading" )
    with open( downloading_file, "wb" ) as out:
        downloaded_size     = 0
        while True:
            chunk_size      = 64 * 1024
            data            = response.read( chunk_size )
            if not data:
                break

            out.write       ( data )
            downloaded_size += len( data )
            progress_bar.update( downloaded_size )

    Path.rename             ( downloading_file, file )
    progress_bar.finish     ( )
    response.release_conn   ( )

def initialize_pretrained_model( pretrained_model_folder ):
    Path.mkdir              ( pretrained_model_folder, parents = True, exist_ok = True )
    
    pretrained_model_url    = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"
    pretrained_model_name   = "imagenet-vgg-verydeep-19.mat"
    pretrained_model_path   = Path.joinpath( pretrained_model_folder, pretrained_model_name )
    if not Path.exists( pretrained_model_path ):
        download            ( pretrained_model_url, pretrained_model_path, "downloading " + pretrained_model_name )

    return                  pretrained_model_path

g_images_database_version   = 0

def initialize_images_database( images_folder, urls ):
    global g_images_database_version
    new_images_database_version = g_images_database_version + 1

    result                  = []

    Path.mkdir              ( images_folder, parents = True, exist_ok = True )
    for url in urls:
        image_name          = Path( url ).parts[ -1 ]
        image_path          = Path.joinpath( images_folder, image_name )
        if not Path.exists( image_path ):
            g_images_database_version = new_images_database_version
            download        ( url, image_path, "downloading " + image_name )
        result.append       ( image_path )

    return                  result, g_images_database_version

#-----------------------------------------------------------------------------------------------------------------------
# set an already loaded image as an input for VGG-19 model
#-----------------------------------------------------------------------------------------------------------------------
def set_input                   ( image, model, session ):
    # Reshape image to mach expected input of VGG-19
    # consider to use another algorithm for resizing
    image                       = scipy.misc.imresize( image, (300, 400) )

    # Substract the mean to match the expected input of VGG-19
    image                       = image - CONFIG.MEANS

    x_train                     = np.array( image )

    # Assign the image to be the input of the VGG model.  
    session.run                 ( model['input'].assign(x_train) )

#-----------------------------------------------------------------------------------------------------------------------
# retrieves a specified layer encoding
#-----------------------------------------------------------------------------------------------------------------------
def layer_encoding              ( session, layer ):
    return                      session.run( layer )

#-----------------------------------------------------------------------------------------------------------------------
# returns encoding for an already loaded image
#-----------------------------------------------------------------------------------------------------------------------
def image_encoding              ( image, model, session, features_layer ):
    set_input                   ( image, model, session )
    return                      layer_encoding( session, features_layer )

#-----------------------------------------------------------------------------------------------------------------------
# returns a loaded image in the CONFIG.IMAGES_FOLDER with the specified name
#-----------------------------------------------------------------------------------------------------------------------
def encoding_file_name( image_path, layer ):
    return                      Path.joinpath( image_path.parent, "encodings", layer, image_path.name + ".npy" )

#-----------------------------------------------------------------------------------------------------------------------
# returns a loaded image in the CONFIG.IMAGES_FOLDER with the specified name
#-----------------------------------------------------------------------------------------------------------------------
def new_image( name ):
    return                      scipy.misc.imread( name )

g_encodings_layer               = ""
g_encodings                     = { }
g_processed_images_database_version = -1

#-----------------------------------------------------------------------------------------------------------------------
# returns a dictionary of the encodings from preprocessed encodings if exist or from the raw images
#-----------------------------------------------------------------------------------------------------------------------
def layer_encodings             ( images_database_version, images_file_names, encodings_layer, model, session ):
    global g_encodings_layer, g_encodings, g_processed_images_database_version

    if ( g_processed_images_database_version == images_database_version and g_encodings_layer == encodings_layer and len(images_file_names) == len(g_encodings) ):
        found                   = False
        for i in images_file_names:
            if not (i in g_encodings):
                found           = True
                break

        if not found:
            return              g_encodings

    result                      = { }
    for image_path in images_file_names:
        if image_path in g_encodings:
            result[ image_path ] = g_encodings[ image_path ]
            continue

        encoding_path           = encoding_file_name( image_path, encodings_layer )
        if Path.exists( encoding_path ):
            result[ image_path ] = np.load( encoding_path )
            continue

        Path.mkdir              ( encoding_path.parent, parents = True, exist_ok = True )
        print                   ( "computing encoding for " + image_path.name )
        image                   = new_image( image_path )
        set_input               ( image, model, session )
        encoding                = layer_encoding( session, model[ encodings_layer ] )
        np.save                 ( encoding_path, encoding )
        result[ image_path ]    = encoding

    g_processed_images_database_version = images_database_version
    g_encodings_layer           = encodings_layer
    g_encodings                 = result
    return                      result

#-----------------------------------------------------------------------------------------------------------------------
# adds a specified image to the output
#-----------------------------------------------------------------------------------------------------------------------
def show_image                  ( image ):
    subplot                     = plt.imshow( image, interpolation = "nearest" )
    subplot.set_cmap            ( "hot" )
    axes                        = subplot.axes
    axes.get_xaxis().set_visible( False )
    axes.get_yaxis().set_visible( False )

def add_output_image            ( figure, image, index ):
    figure.add_subplot          ( 1, 5, index )
    show_image                  ( image )

def show_similar                ( image, encodings_layer, model, images_file_names, images_database_version ):
    session                     = tf.InteractiveSession( )

    encodings                   = layer_encodings( images_database_version, images_file_names, encodings_layer, model, session )

    target_image                = new_image( image )
    target_encoding             = image_encoding( target_image, model, session, model[ encodings_layer ] )

    session.close               ( )

    distances                   = []
    for i in encodings:
        distances.append        ( ( np.linalg.norm( target_encoding - encodings[ i ] ), i ) )

    distances.sort              ( )

    show_image                  ( target_image )
    plt.show                    ( )

    figure                      = plt.figure( figsize = (1200, 400), dpi=1, frameon = False )
    for i in range( 0, 5 ):
        add_output_image        ( figure, new_image( distances[i][1] ), i+1 )

    # set output window position to (0, 0)
    # output_window               = plt.get_current_fig_manager().window
    # output_window.wm_geometry   ( "+0+0" )

    plt.show                    ( )
