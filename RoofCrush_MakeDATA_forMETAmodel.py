####################################################
####                                            ####
#### 2018.06.19 SUNGHUN, CHANG                  ####
####                                            ####
#### For BIW Roofcrush Performance Prediction   ####
####                                            ####
####################################################

import tensorflow as tf
import os
import time
import sys
import argparse
import Model_Definitions_Expand_Features_v2 as models

cnt = 0
for modeldirfind in os.listdir("./"):
    if os.path.isdir(modeldirfind) == True :
        if modeldirfind.startswith("model_") == True :
            cnt = cnt + 1
            print(modeldirfind)
print(cnt)

os.system('cls')
tf_version = tf.__version__

log_dir = "./02_DATA_for_META_Regression"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
DATA_File = open(log_dir + "/DATA_for_META_Regression.sv", 'w')
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=2, help='Number of Trained Model. Default : 2')
args = parser.parse_args()
Num_Of_Models = args.model
'''
Num_Of_Models = cnt

if Num_Of_Models == 1:
    print("최소 2개 이상의 모델이 필요함")
    #quit()
    sys.exit(1)

PATH_DATASET = "./dataset"
FILE_PRACTICE = PATH_DATASET + os.sep + "20180614_Roof_Rev05_Shuffle_for_Train_482EA.csv"
#FILE_PRACTICE = PATH_DATASET + os.sep + "20180614_Roof_Rev05_Shuffle_for_Test_98EA.csv"

print()
print("        ************************************************** ")
print("        *****    DATA Population for BIW RoofCrush   ***** ")
print("        *****             SUNGHUN, CHANG             ***** ")
print("        *****       TensorFlow version : {}       ***** ".format(tf_version))
print("        ************************************************** ")
print()

Regressor_list = []

for k in range(1, Num_Of_Models + 1):
    tf.logging.info("MODEL #" + str(k) + " CONSTRUCTION")
    Regressor_list.append(tf.estimator.Estimator(model_fn=models.my_model_fn,
                                                  model_dir="./model_" + str(k).rjust(3,'0'),
                                                  params=
                                                  {
                                                      "feature_columns": models.feature_columns,
                                                      "model_identifier": str(k).rjust(3,'0')
                                                  }
                                                  )
                           )

total_prediction_list = []
one_model_prediction_list = []
feature_list = []

print("   ##  Restore and Build Neural Network Model from CheckPoint  ##\n")

_, tmp_labels = models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=500, shuffle_count=1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels) #Extract Tensor to List

for k in range(0,Num_Of_Models):
    predict_results = Regressor_list[k].predict(input_fn=lambda:models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1))
    i = 0
    one_model_prediction_list = []
    feature_list = []
    for prediction in predict_results:
        one_model_prediction_list.append(prediction["Squeeze"])
    #one_model_prediction_list.append(labels[i])
    i = i + 1
    total_prediction_list.append(one_model_prediction_list)
total_prediction_list.append(list(labels)) # tuple to list

temp_txt = ""
for l in range(0,Num_Of_Models) :
    temp_txt = temp_txt + "Model_" + str(l).rjust(3,'0') + ','
temp_txt = temp_txt + "Labels"
DATA_File.write(temp_txt+"\n")

temp_txt = ""
for j in range(0,len(total_prediction_list[0])):
    temp_txt = str(total_prediction_list[0][j])
    for k in range(1,Num_Of_Models+1) : # Label 때문에 한개가 더 필요함
        temp_txt = temp_txt + ',' + str(total_prediction_list[k][j])
    DATA_File.write(temp_txt+"\n")

DATA_File.close()
