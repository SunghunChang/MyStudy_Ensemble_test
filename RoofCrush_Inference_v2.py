####################################################
####                                            ####
#### 2018.06.19 SUNGHUN, CHANG                  ####
####                                            ####
#### For BIW Roofcrush Performance Prediction   ####
####                                            ####
####################################################

import tensorflow as tf
import os
import argparse
import Model_Definitions_Expand_Features_v2 as models

'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=2, help='Number of Train Model. Default : 2')
args = parser.parse_args()
Num_Of_Models = args.model
'''
cnt = 0
for modeldirfind in os.listdir("./"):
    if os.path.isdir(modeldirfind) == True :
        if modeldirfind.startswith("model_") == True :
            cnt = cnt + 1
Num_Of_Models = cnt

#tf.logging.set_verbosity(tf.logging.INFO)
tf_version = tf.__version__
tf.logging.info("TensorFlow version: {}".format(tf_version))

PATH_DATASET = "./dataset"
FILE_PRACTICE = PATH_DATASET + os.sep + "20180614_Roof_Rev05_Shuffle_for_Test_98EA.csv"

print()
print("        ************************************************** ")
print("        *****      Prediction for BIW RoofCrush      ***** ")
print("        *****             SUNGHUN, CHANG             ***** ")
print("        *****       TensorFlow version : {}       ***** ".format(tf_version))
print("        ************************************************** ")
print()

classifier_list = []

for k in range(1, Num_Of_Models + 1):
    tf.logging.info("MODEL #" + str(k) + " CONSTRUCTION")
    classifier_list.append(tf.estimator.Estimator(model_fn=models.my_model_fn,
                                                  model_dir="./model_" + str(k).rjust(3,'0'),
                                                  params=
                                                  {
                                                      "feature_columns": models.feature_columns,
                                                      "model_identifier": str(k).rjust(3,'0')
                                                  }
                                                  )
                           )

'''
_, tmp_labels = models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels)
'''

total_prediction_list = []
one_model_prediction_list = []
feature_list = []

print("   ##  Restore and Build Neural Network Model from CheckPoint  ##\n")

for k in range(0,Num_Of_Models):
    predict_results = classifier_list[k].predict(input_fn=lambda:models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1))
    i = 1
    one_model_prediction_list = []
    feature_list = []
    for prediction in predict_results:
        '''
        prediction_print = "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}".format(
            i,
            prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
            prediction["MP01"], prediction["MP02"], prediction["MP03"], prediction["MP04"], prediction["MP05"],
            prediction["MP06"], prediction["MP07"], prediction["MP08"], prediction["MP09"], prediction["MP10"],
            prediction["MP11"], prediction["MP12"],
            prediction["Squeeze"], "Predict : ")
        prediction_print_shape = "\t{0:0.2f}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}\t{5:0.2f}\t{6:0.2f}\t{7:0.2f}\t{8:0.2f}\t{9:0.2f}\t{10:0.2f}\t{11:0.2f}\t{12:0.2f}\t{13:0.2f}\t{14:0.2f}\t{15:0.2f}".format(
            prediction['A'], prediction['A_Ang_X'], prediction['A_Ang_Z'],
            prediction['S1'], prediction['S1_Ang_Y'], prediction['S1_Ang_Z'],
            prediction['S2'], prediction['S2_Ang_Y'], prediction['S2_Ang_Z'],
            prediction['B'], prediction['B_Ang_X'], prediction['B_Ang_Y'],
            prediction['R'], prediction['R_Ang_X'], prediction['R_Ang_Z'], prediction['BU'])
        '''
        feature_txt = "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}".format(
            i,
            prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
            prediction["MP01"], prediction["MP02"], prediction["MP03"], prediction["MP04"], prediction["MP05"],
            prediction["MP06"], prediction["MP07"], prediction["MP08"], prediction["MP09"], prediction["MP10"],
            prediction["MP11"], prediction["MP12"])
        feature_txt_shape = "\t{0:0.2f}\t{1:0.2f}\t{2:0.2f}\t{3:0.2f}\t{4:0.2f}\t{5:0.2f}\t{6:0.2f}\t{7:0.2f}\t{8:0.2f}\t{9:0.2f}\t{10:0.2f}\t{11:0.2f}\t{12:0.2f}\t{13:0.2f}\t{14:0.2f}\t{15:0.2f}".format(
            prediction['A'], prediction['A_Ang_X'], prediction['A_Ang_Z'],
            prediction['S1'], prediction['S1_Ang_Y'], prediction['S1_Ang_Z'],
            prediction['S2'], prediction['S2_Ang_Y'], prediction['S2_Ang_Z'],
            prediction['B'], prediction['B_Ang_X'], prediction['B_Ang_Y'],
            prediction['R'], prediction['R_Ang_X'], prediction['R_Ang_Z'], prediction['BU'])
        one_model_prediction_list.append(prediction["Squeeze"])
        feature_list.append(feature_txt + "\n\t" + feature_txt_shape)
        if Num_Of_Models == 1:
            print(feature_txt + "\n\t" + feature_txt_shape)  # 개별 결과는 모델이 1개일 때만 출력함
            print("\t        --> Predicted Value " + str(prediction["Squeeze"]).rjust(6,'0'))
        i = i + 1
    total_prediction_list.append(one_model_prediction_list)

average_prediction_file = []
for j in range(0,len(total_prediction_list[0])):
    temp_prediction = 0.0
    for k in range(0,Num_Of_Models) : temp_prediction = temp_prediction + total_prediction_list[k][j]
    average_prediction_file.append(temp_prediction / float(Num_Of_Models))

os.system('cls')

print("")
print("               *********************************")
print("               **   ENSEMBLE AVERAGE RESULTs  **")
print("               *********************************")
print("")
print("   ## Source file : {0}\n".format(FILE_PRACTICE))
for i in range(0,len(feature_list)) :
    print(feature_list[i] + "\n  Case #{0:3d} --> Predicted Value : {1:0.2f}\n".format(i+1,average_prediction_file[i]))