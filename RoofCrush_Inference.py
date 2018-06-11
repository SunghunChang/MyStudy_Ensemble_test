####################################################
####                                            ####
#### 2018.06.11 SUNGHUN, CHANG                  ####
####                                            ####
#### For BIW Roofcrush Performance Prediction   ####
####                                            ####
####################################################

import tensorflow as tf
import os
import argparse
import Model_Definitions as models
import Memory_Prediction_DATA as mpd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=2, help='Number of Train Model. Default : 2')
args = parser.parse_args()

Num_Of_Models = args.model

#tf.logging.set_verbosity(tf.logging.INFO)
tf_version = tf.__version__
tf.logging.info("TensorFlow version: {}".format(tf_version))

PATH_DATASET = "./dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "20180430_RC_Populated_DATA_Isolated_Train_Random.csv"
FILE_TEST = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"
FILE_PRACTICE = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"

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

_, tmp_labels = models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels)

for k in range(0,Num_Of_Models):
    predict_results = classifier_list[k].predict(input_fn=lambda:models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1))
    i = 1
    for prediction, expec in zip(predict_results, labels):
        prediction_print = "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}\t{16}{17:0.2f}\t{18:0.2f}%".format(
            i,
            prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
            prediction["MP01"], prediction["MP02"], prediction["MP03"], prediction["MP04"], prediction["MP05"],
            prediction["MP06"], prediction["MP07"], prediction["MP08"], prediction["MP09"], prediction["MP10"],
            prediction["MP11"], prediction["MP12"],
            prediction["Squeeze"], "Predict : ", "Expected : ", expec, (abs(prediction["Squeeze"] - expec)/expec) * 100)
        print(prediction_print)
        i = i + 1