import cv2
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.vgg16 import VGG16
from os import listdir
from os.path import isfile, isdir, join
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras.preprocessing import image
import numpy as np
import random
import h5py
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.models import Model
from keras.layers import Flatten
from keras import backend as K
import tensorflow as tf
import os
from keras.regularizers import l1
from keras.layers import Dropout
from tensorflow.keras import optimizers
from linebot import (LineBotApi, WebhookHandler)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage,
    SourceUser, SourceGroup, SourceRoom,
    TemplateSendMessage, ConfirmTemplate, MessageTemplateAction,
    ButtonsTemplate, URITemplateAction, PostbackTemplateAction,
    CarouselTemplate, CarouselColumn, PostbackEvent,
    StickerMessage, StickerSendMessage, LocationMessage, LocationSendMessage,
    ImageMessage, VideoMessage, AudioMessage,
    UnfollowEvent, FollowEvent, JoinEvent, LeaveEvent, BeaconEvent
)
import mysql.connector
# for now()
import datetime
# for timezone()
import pytz
import time


def insert_fire_detection(mydb, detection, detection_prob, detection_time, img_path, cam_id, lat, lon, address):
    detection_image = convertToBinaryData(img_path)
    mycursor = mydb.cursor()

    sql = "INSERT INTO fire_detections VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s)"
    val = (None, detection, detection_prob, detection_time,
           detection_image, cam_id, lat, lon, address)
    mycursor.execute(sql, val)
    mydb.commit()


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData


def loadtrainmodel(model_save_file, weight_save_file):
    json_file = open(model_save_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_save_file)
    print("Loaded model from disk")

    loaded_model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                         loss='categorical_crossentropy',
                         metrics=['acc'])
    return loaded_model


def getCustomModel(train_model):

    vgg_model = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(224, 224, 3))

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

    # Getting output tensor of the last VGG layer that we want to include
    x = layer_dict['block5_pool'].output
    x = Flatten()(x)
    for layer in train_model.layers:
        x = layer(x)

    custom_model = Model(inputs=vgg_model.input, outputs=x)
    return custom_model


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(
            last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def makeHeatmapFrame(img_array, model, last_conv_layer_name, frame, display_text, pred_index=None):
    heatmap = make_gradcam_heatmap(
        img_array, model, last_conv_layer_name, pred_index)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.putText(heatmap, display_text, text_location, font,
                          fontScale, font_color, thickness, cv2.LINE_AA)
    return heatmap


frame_skip = 60
text_location = (100, 100)

model_json = "model.json"
model_h5 = "model.h5"
last_layers = loadtrainmodel(model_json, model_h5)
last_layers.summary()
fire_detector_model = getCustomModel(last_layers)
fire_detector_model.summary()
# dummy_image = cv2.imread('smoke.jpg')
# dummy_image = cv2.resize(dummy_image, None, fx=1, fy=1)
# Choose Video Source option

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("fire.mp4")
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
frame_count = 120
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
font_color = (0, 0, 255)
status = ""

class_list = ["fire", "non_fire",  "smoke"]
class_prediction = class_list[0]
probs = []
CHANNEL_ACCESS_TOKEN = 't8OgaTeF5Z6LfMWngJKlflxfSPYZ+BVfDKY9V81pqPshq488c+hpBnvelMDjzMzgdb6QQgmnDu28dEDEsr8ybXd0f6FIfx/i34rAScywLMzoeNbmorIp4XKb/gJ9lV74AwHH4WQHD76pT2w0lxZ5xgdB04t89/1O/w1cDnyilFU='
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
APACHE_SERVER_FOLDER = 'C:\\AppServ\\www\\vermillion\\'
NGROK_URL = 'https://f4ba-171-99-154-51.ngrok.io'
SERVER_FOLDER = 'vermillion'

MESSSAGE_DELAY = 30  # in seconds
# dummy detection location
cam_id = 1
lat = 19.2116027
lon = 97.9801175
address = 'จุดชมวิว ต.ผาบ่อง อ.เมืองแม่ฮ่องสอน'

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="12345678",
    database="vermillion"
)


send_time = 0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # frame = dummy_image
    if ret == True:

        if frame_count > frame_skip:
            original = frame.copy()
            frame_count = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = cv2.resize(frame, (224, 224))
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            probs = fire_detector_model.predict([x])

            class_prediction = class_list[np.argmax(probs, axis=1)[0]]
            prediction_prob = float(probs[0][np.argmax(probs, axis=1)[0]])

            fire_text = "fire probability : " + str(probs[0][0])
            fire_heatmap = makeHeatmapFrame(
                x, fire_detector_model, 'block5_conv3', frame, fire_text, pred_index=0)

            nonfire_text = "nonfire probability : " + str(probs[0][1])
            nonfire_heatmap = makeHeatmapFrame(
                x, fire_detector_model, 'block5_conv3', frame, nonfire_text, pred_index=1)

            smoke_text = "smoke probability : " + str(probs[0][2])
            smoke_heatmap = makeHeatmapFrame(
                x, fire_detector_model, 'block5_conv3', frame, smoke_text, pred_index=2)

            send_message_elapse = time.time() - send_time
            if (class_prediction == "fire" or class_prediction == "smoke") and send_message_elapse > MESSSAGE_DELAY:
                current_time = datetime.datetime.now(
                    pytz.timezone('Asia/Bangkok'))
                filename = str(current_time).replace('.', '').replace(
                    '-', '').replace('+', '').replace(':', '').replace(' ', '') + '.jpg'

                file_path = APACHE_SERVER_FOLDER + filename
                IMAGE_URL = NGROK_URL + '/' + SERVER_FOLDER + '/' + filename

                print(file_path, IMAGE_URL)
                cv2.imwrite(file_path, original)
                event = 'เหตุการณ์'
                if class_prediction == "fire":
                    event = 'ไฟ'
                elif class_prediction == "smoke":
                    event = 'ควัน'

                message = "ตรวจสอบพบ{event}ที่สถานี{address} ความน่าจะเป็นจากเอไอ = {prob}".format(
                    event=event, address=address, prob=prediction_prob)
                text_message = TextSendMessage(text=message)
                image_message = ImageSendMessage(
                    original_content_url=IMAGE_URL,
                    preview_image_url=IMAGE_URL
                )
                location_title = ''
                if len(message) > 100:
                    location_title = message[0:99]
                else:
                    location_title = message

                location_message = LocationSendMessage(
                    title=location_title, address=address, latitude=lat, longitude=lon)

                line_bot_api.broadcast(text_message)
                line_bot_api.broadcast(image_message)
                line_bot_api.broadcast(location_message)
                send_time = time.time()
                detection_time = time.strftime('%Y-%m-%d %H:%M:%S')

                insert_fire_detection(mydb, class_prediction, prediction_prob,
                                      detection_time, file_path, cam_id, lat, lon, address)

        frame = cv2.putText(frame, "status : " + class_prediction, text_location, font,
                            fontScale, font_color, thickness, cv2.LINE_AA)
        fourchannel_display = np.vstack(
            (np.hstack((frame, nonfire_heatmap)), np.hstack((fire_heatmap, smoke_heatmap))))

        # Choose Display option
        cv2.imshow('Vermillion Wildfire Detection Version 1.0',
                   fourchannel_display)
        # cv2.imshow('Vermillion Wildfire Detection Version 1.0',
        #           frame)
        frame_count = frame_count + 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
