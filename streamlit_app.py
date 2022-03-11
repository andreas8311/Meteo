import requests
from datetime import date, timedelta, datetime
import streamlit as st
import pandas as pd

from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import matplotlib.image as mpimg
from copy import copy
#import tensorflow as tf
import cv2 as cv


from tensorflow.keras import models


carte = mpimg.imread("carte_test.png")
cols = {1:[209,251,252],    ## Couleurs de l'echelle d'intensite de pluie (mm/h)
       2:[97,219,241],
       3:[76,147,240],
       4:[23,38,192],
       5:[0,141,3],
       6:[12,255,0],
       7:[255,249,0],
       8:[255,145,0],
       9:[232,0,0],
       10:[232,0,230],
       11:[255,175,254]}






st.title("Predicting Weather over North of France")



def retirer_carte_fond (img, carte):
    # Calcul de la diff entre l'image radar et la carte
    im_diff= np.asarray(img)- np.asarray(carte)

    # Restitution de leur vrai valeur aux pixels non proches de 0
    M =np.ones((866, 900, 3)) # M =np.ones((img.shape[0], img.shape[1], 3))
    M[im_diff<0.1]=0
    img_radar = M*img

    return img_radar

def retirer_txt (img):
    """Mettre zone de txt en haut a gauche de l'image a 0"""
    img[0:100,0:200,:] = 0
    return img

def colors2grays (img):
    """Transforme les vraies couleurs en niveau de gris"""
    gray_image_3c = copy(img)
    if np.max(img) <= 1. :
        gray_image_3c = copy(img)*255

    gray_image_1c = copy(gray_image_3c[:,:,0])
    gray_image_1c[:,:] = 0

    tolerance = 65  ## l

    for i in  range(1,12):
        col_lo=np.array([x-tolerance for x in cols[i]])
        col_hi=np.array([x+tolerance for x in cols[i]])

        mask=cv.inRange(gray_image_3c,col_lo,col_hi)
        gray_image_1c[mask>0]=i/11

    return gray_image_1c

def lissage_image(img):
    img = img.astype('float32')
    img = cv.medianBlur(img, 5)
    return img

def crop_image (img, zone) :
    ## Zoom sur la zone d'interet
    if zone == 'France_Nord' :
        limite = [30,450,100,750]    ## Limites : [H_min, H_max, L_min, L_max]
    elif zone == 'IDF':
        limite = [190,265,400,510]    ## Limites : [H_min, H_max, L_min, L_max]
    else :
        print("Unknown area : Area should be in ('France_Nord', 'IDF')")
    img_zoom = img[limite[0]:limite[1],limite[2]:limite[3]]
    return img_zoom


def iteration_15min(start, finish):
    ## Generateur de (an, mois, jour, heure, minute)
     while finish > start:
        start = start + timedelta(minutes=15)
        yield (start.strftime("%Y"),
               start.strftime("%m"),
               start.strftime("%d"),
               start.strftime("%H"),
               start.strftime("%M")
               )

def open_save_data(url, date_save):
    ## Ouvre l'image pointee par url
    ## Enregistre l'image avec l'extention date_save

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))
    st.image(img) # This is showing the image on the screen
    img = retirer_carte_fond(img, carte)
    img = retirer_txt(img)
    img_gray = colors2grays(img)
    img_gray = lissage_image(img_gray)
    img_zoomX = crop_image(img_gray, 'France_Nord')
    img_zoomX = img_zoomX[::5, ::5]
    #st.image(img_zoomX, clamp=True)

    return np.array(img_zoomX)

def scrapping_images (start, finish) :
    """Scrape images radar en ligne toutes les 15 min
    entre deux dates donnees sous forme de datetime.datetime
    Sauvegarde les dates pour lesquelles la page n'existe pas.  """

    saved_images = []
    for (an, mois, jour, heure, minute) in iteration_15min(start, finish):
        ## url scrapping :
        url = (f"https://static.infoclimat.net/cartes/compo/{an}/{mois}/{jour}/color_{jour}{heure}{minute}.jpg")
        date_save = f'{an}_{mois}_{jour}_{heure}{minute}'

        try :
            tmp = open_save_data(url, date_save)
            st.image(tmp)
            saved_images.append(tmp)
            #if len(saved_images)>=10:
                #break


        except UnidentifiedImageError :
            print (date_save, ' --> Missing data')
            break

    saved_images = saved_images[-10:]
    return saved_images

if st.button('Predict Weather'):

    date_only = datetime.now().date()
    time_only = datetime.now().time()
    start = datetime(date_only.year, date_only.month, date_only.day, (time_only.hour)-3,00)
    finish = datetime(date_only.year, datetime.now().date().month, date_only.day, (time_only.hour)+1, 00)

    st.write('Initial Images over France')
    frames = scrapping_images(start, finish)
    model = models.load_model("AJ_my_model_mse_long_11")
    st.header('And predicted images..')


    new_prediction = model.predict(np.expand_dims(frames, axis=0))
    new_predictions = np.squeeze(new_prediction, axis=0)
    #new_predictions = np.squeeze(new_predictions, axis=-1) # Should normally not be there
    st.write(new_predictions.shape)


    new_predictions2 = np.zeros(shape=(10, *new_predictions[0].shape))
    for i in range(10):
        one_frame = new_predictions[i]

        one_frame = np.where(one_frame < 0.02, 0, one_frame*3.5)  # one_frame*2.5
        # print("fra2 shape : ",one_frame.shape)
        new_predictions2[i] = one_frame


    img_back_gif = crop_image (carte, 'France_Nord')

    st.image(img_back_gif)

    #### GIF GENERATION ####
    #from PIL import Image
    st.write("shape frames : ", np.array(frames).shape)
    frames = [Image.fromarray(np.uint8((frame * 255).astype(int))) for frame in frames]
    frame_one = frames[0]
    frame_one.save('gif_1.gif', format="GIF", append_images=frames,
                   save_all=True, duration=10, loop=0)
    st.image('gif_1.gif',use_column_width='always')

    new_predictions_gif = np.squeeze(new_predictions2, axis=-1)
    st.write("shape new predictions2 : ", new_predictions_gif .shape)
    new_predictions_gif = [Image.fromarray(np.uint8((frame * 255).astype(int))) for frame in new_predictions_gif]
    frame_one_pred = new_predictions_gif[0]
    frame_one_pred.save('gif_2.gif', format="GIF", append_images=new_predictions_gif,
                   save_all=True, duration=10, loop=0)
    st.image('gif_2.gif', )



    for i in range(10):
        current_frames = new_predictions2[i]
        current_frames= np.squeeze(current_frames, axis=-1) # Should normally not be there
        st.write("Prediction number : ",i)
        new_im = (current_frames * 255).astype(np.uint8)

        new_im = cv.pyrUp(new_im)
        st.write(new_im.shape)

        st.image(new_im)
