from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model, load_model
from keras.layers import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import numpy as np
import os
import random
import tensorflow as tf

main = tkinter.Tk()
main.title("Gray Scale Video & Image Colorization")
main.geometry("1300x1200")

global filename, model, Xtrain
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.compat.v1.get_default_graph()

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')

def preprocess():
    global filename, Xtrain
    text.delete('1.0', END)
    X = []
    for name in os.listdir(filename):
        X.append(img_to_array(load_img(filename+'/'+name, target_size=(256, 256))))
        print(filename+'/'+name)
    X = np.array(X, dtype=float)
    Xtrain = 1.0/255*X
    text.insert(END,'Dataset Processing & Normalization Completed\n\n')
    text.insert(END,"Total images = "+str(Xtrain.shape[0]))

def create_inception_embedding(grayscaled_rgb):
    global inception
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    embed = inception.predict(grayscaled_rgb_resized)
    return embed

def image_a_b_gen(batch_size):
    global inception
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

def trainCNN():
    global Xtrain, model
    text.delete('1.0', END)
    embed_input = Input(shape=(1000,))
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True)
    batch_size = 10
    model.compile(optimizer='rmsprop', loss='mse')
    if os.path.exists("model/color_weights.h5") == False:
        model.fit_generator(image_a_b_gen(batch_size), epochs=5000, steps_per_epoch=1)
        model.save_weights('model/color_weights.h5')
    else:
        model.load_weights('model/color_weights.h5')
    text.insert(END,"Custom CNN Model Loaded. Summary can be viewed in Console Screen")
    print(model.summary())

def imageColoring():
    global model
    filename = filedialog.askopenfilename(initialdir = "testImages")
    color_me = []
    color_me.append(img_to_array(load_img(filename, target_size=(256, 256))))
    color_me = np.array(color_me, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
    color_me_embed = create_inception_embedding(gray_me)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))
    original = cv2.imread(filename)
    output = model.predict([color_me, color_me_embed])
    output = output * 128
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[0][:,:,0]
    cur[:,:,1:] = output[0]
    img = lab2rgb(cur)
    '''
    print(type(img))
    print(img.shape)
    im = (img * 255).astype(np.uint8)
    imsave("im.png", im)
    '''
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("original Gray Image", original)
    cv2.imshow("Colored Image", img)
    cv2.waitKey(0)
    
def getColored():
    color_me = []
    color_me.append(img_to_array(load_img("input.png", target_size=(256, 256))))
    color_me = np.array(color_me, dtype=float)
    gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
    color_me_embed = create_inception_embedding(gray_me)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))
    output = model.predict([color_me, color_me_embed])
    output = output * 128
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[0][:,:,0]
    cur[:,:,1:] = output[0]
    img = lab2rgb(cur)
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img   


def videoColoring():
    global model
    filename = filedialog.askopenfilename(initialdir = "Videos")
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.resize(frame, (500, 500))
            w, h, c = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("input.png", frame)
            colored = getColored()
            colored = cv2.resize(colored, (w, h))
            cv2.imshow("Original Gray Video", frame)
            cv2.imshow("Colored Video", colored)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()                
    

font = ('times', 16, 'bold')
title = Label(main, text='Gray Scale Video & Image Colorization')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Color Images Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=700,y=200)
preprocessButton.config(font=font1) 

trainButton = Button(main, text="Train Custom CNN Model", command=trainCNN)
trainButton.place(x=700,y=250)
trainButton.config(font=font1) 

imageButton = Button(main, text="Colorized Test Image", command=imageColoring)
imageButton.place(x=700,y=300)
imageButton.config(font=font1)

videoButton = Button(main, text="Colorized Video", command=videoColoring)
videoButton.place(x=700,y=350)
videoButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
