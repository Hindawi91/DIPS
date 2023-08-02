import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import imageio
from functools import reduce
import os
from tensorflow.keras.preprocessing import image
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score

pseudo_metric = "Balanced Accuracy"  # Choose the pseudo metric "Balanced Accuracy" or "AUC"
CNN_classifier = "./CNN_Training/CNN DS1 Base Model.hdf5" # Choose base model depending on experiment

model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet')
layer_name = "conv2d_93"
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

# Path for the original target dataset (to get features)
path1 = "/data/DS1_2_DS2_Data/Original_target_ds2/"
# Path for the folder that contatins the translated dataset from each checkpoint model 
path2 = "/data/DS1_2_DS2_Data/Translated_target_ds2/"


def get_activations(intermediate_layer_model,folder):

    images = []

    im_files = glob(f'{folder}/*.j*')

    for i,im_file in enumerate (im_files):
        

        img1 = image.load_img(im_file,target_size=(299, 299))
        img1 = image.img_to_array(img1)
        img1 = np.expand_dims(img1, axis=0)
        img1 /= 255.
        images.append(img1)

    imagesNP = np.vstack(images)
    # print(imagesNP.shape)


    pred = intermediate_layer_model.predict(imagesNP)

    # print(pred.shape)

    pred = pred.reshape(pred.shape[0],-1)

    # print(pred.shape)

    return pred

# --------------- Get Features -------------------------
features = get_activations(intermediate_layer_model,folder=path1)
gm = GaussianMixture(n_components=2, random_state=0).fit(features)

y_pseudo_true = gm.predict(features)

y_pseudo_true_flipped = []
y_pseudo_true_flipped = [int(not i) for i in y_pseudo_true]


# # ============================================= GET y_pred for 30 models ========================================================


y_pseudo_lst = ['y_pseudo_true','y_pseudo_true_flipped']
model = tf.keras.models.load_model(CNN_classifier)

for y in y_pseudo_lst:

    if y == 'y_pseudo_true':

        y_true = y_pseudo_true

    else:
        y_true = y_pseudo_true_flipped


    All_Metrics = []
    All_CMs = []

    for exp in range(10000,300001,10000):

        # model_name = CNN_classifier

        images = []

        im_files = glob(f'{path2}results_{exp}/*.j*')

        for i,im_file in enumerate (im_files):
            

            img1 = image.load_img(im_file,target_size=(128, 72))
            # img1 = image.load_img(im_file)
            img1 = image.img_to_array(img1)
            img1 = np.expand_dims(img1, axis=0)
            img1 /= 255.
            images.append(img1)
                
        # model = tf.keras.models.load_model(model_name)
        

        imagesNP = np.vstack(images)
        y_pred = model.predict(imagesNP)
        # y_pred_prob = model.predict_proba(imagesNP)[:,1]
        y_pred_prob = y_pred[:,1]
        y_pred = np.argmax(y_pred,axis=1)
        
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        AUC = roc_auc_score(y_true, y_pred_prob)

        
        metrics = [exp,balanced_acc,AUC]
        metrics_names = ["Checkpoint Model","Balanced Accuracy","AUC"]
        
        All_Metrics.append(metrics)


    # In[24]:

    if y == 'y_pseudo_true':

        option1_df = pd.DataFrame(All_Metrics,columns=metrics_names)
        option1_avg = option1_df[pseudo_metric].mean()
        print(f"{option1_avg =}")

    elif y == 'y_pseudo_true_flipped':

        option2_df = pd.DataFrame(All_Metrics,columns=metrics_names)
        option2_avg = option2_df[pseudo_metric].mean()
        print(f"{option2_avg =}")

if option1_avg > option2_avg:

    print(f"Selected clusering choice is option 1")

    # Find best model
    best_PSM = option1_df[pseudo_metric].max()
    index = option1_df[pseudo_metric].idxmax()
    best_chk_model = option1_df.loc[index, 'Checkpoint Model']
    print("Best Checkpoint Model:", best_chk_model)
    print("Best PSM:", best_PSM)

    # save the results of the 30 models to excel

    option1_df.to_excel (f'PSM_Results.xlsx', index = False, header=True)

else:
    print(f"Selected clusering choice is option 2")

    # Find best model
    best_PSM = option2_df[pseudo_metric].max()
    index = option2_df[pseudo_metric].idxmax()
    best_chk_model = option2_df.loc[index, 'Checkpoint Model']
    print("Best Checkpoint Model:", best_chk_model)
    print("Best PSM:", best_PSM)

    # save the results of the 30 models to excel

    option2_df.to_excel (f'PSM_Results.xlsx', index = False, header=True)


