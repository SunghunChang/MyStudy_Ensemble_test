####################################################
####                                            ####
#### 2018.05.25 SUNGHUN, CHANG                  ####
####                                            ####
#### For BIW Roofcrush Performance Prediction   ####
####                                            ####
####################################################

import argparse
import tensorflow as tf
import os
import Model_Definitions as models
import Model_Plot_WnB as varplots
import Model_Class
import sys
import six.moves.urllib.request as request

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=500, help='Number of Train Epoch. Default : 500')
parser.add_argument('--shuffle', type=int, default=256, help='Number of Train shuffle. Default : 256')
args = parser.parse_args()
# print(args.epoch)

Num_Of_Models = 2

tf.logging.set_verbosity(tf.logging.INFO)

print()
print()

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
tf.logging.info("TensorFlow version: {}".format(tf_version))

print()
print("        ************************************************** ")
print("        *****      Prediction for BIW RoofCrush      ***** ")
print("        *****             SUNGHUN, CHANG             ***** ")
print("        *****       TensorFlow version : {}       ***** ".format(tf_version))
print("        ************************************************** ")
print()

assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

# Windows users: You only need to change PATH, rest is platform independent
# PATH = 'S:' + os.sep + '_TF_virtualenv' + os.sep + '007_RoofCrush_Test'
PATH = "./"

# Fetch and store Training and Test dataset files
PATH_DATASET = "./dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "20180430_RC_Populated_DATA_Isolated_Train_Random.csv"
FILE_TEST = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"
FILE_PRACTICE = PATH_DATASET + os.sep + "IsolatedTEST_20180430_Random.csv"

print()
print("        ** Default PATH : {}".format(PATH))
print("        ** TRAIN : {}".format(FILE_TRAIN))
print("        ** TEST : {}".format(FILE_TEST))
print("         -- Number of Train Epoch : {}".format(args.epoch))
print("         -- Number of Train Shuffle : {}".format(args.shuffle))
print()

# Create a custom estimator using my_model_fn
tf.logging.info("Before classifier construction")

classifier_list = []
for k in range(1,Num_Of_Models + 1):
    tf.logging.info("MODEL #" + str(k) + " CONSTRUCTION")
    classifier_list.append(tf.estimator.Estimator(model_fn=models.my_model_fn,
                                                  model_dir="./model_" + str(k),
                                                  params=
                                                  {
                                                      "feature_columns": models.feature_columns,
                                                      "model_identifier": str(k)
                                                  }
                                                  )
                           )

tf.logging.info("...done constructing classifier")


# REFERENCE by Sunghun, Chang 2018.05.14
# 500 epochs = 500 * 120 records [60000] = (500 * 120) / 32 batches = 1875 batches
#
################################################################################################################
# Sunghun, Chang
# 390 records 32 batches ==> 500 epochs * 390 records [195000] = 195000/32 = 6093.75 batches
# 6000 steps
#
# So, if you use 64-batch size, the total steps per one RUN will represents about 3000 steps
# I think that if the batch size is too big, it will affect to accuracy.
# [But it can be solved by RE-training.]
################################################################################################################

# Train the model
# Input to training is a file with training example

train_result_list = []
for k in range(1, Num_Of_Models + 1):
    tf.logging.info("Train MODEL #" + str(k))
    train_result = classifier_list[k - 1].train(
        input_fn=lambda: models.my_input_fn(FILE_TRAIN, args.epoch, args.shuffle))  # file path, repeat, shuffle
    #train_result_list.append(train_result)
    tf.logging.info("TRAINING RESULT of MODEL #" + str(k))
    #tf.logging.info("{}".format(train_result))

# Display Weight and Bias
for k in range(0, Num_Of_Models):
    weight_layer_1 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/First_Hidden_Layer/kernel')
    bias_layer_1 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/First_Hidden_Layer/bias')

    weight_layer_2 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Second_Hidden_Layer/kernel')
    bias_layer_2 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Second_Hidden_Layer/bias')

    weight_layer_3 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Third_Hidden_Layer/kernel')
    bias_layer_3 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Third_Hidden_Layer/bias')

    varplots.PlotWeighNbias(weight_layer_1, weight_layer_2, weight_layer_3, bias_layer_1, bias_layer_2, bias_layer_3, k)

    os.system('cls')

# Evaluate the model using the data contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
for k in range(0,Num_Of_Models):
    tf.logging.info("Evaluation results of 1st MODEL")
    tf.logging.info(" ***** Evaluation results *****")
    tf.logging.info(" *****      #{} MODEL      *****".format(str(k + 1)))
    evaluate_result = classifier_list[k].evaluate(input_fn=lambda: models.my_input_fn(FILE_TEST, 1))
    for key in evaluate_result:
        tf.logging.info("   {} ---> {}".format(key, evaluate_result[key]))
    tf.logging.info(" ******************************")
    tf.logging.info(" ******************************")

# Predict in the data in FILE_TEST, repeat only once.
for k in range(0,Num_Of_Models):
    tf.logging.info(" ***** Prediction on test file ***** ")
    tf.logging.info(" *****        {}st MODEL        ***** ".format(str(k + 1)))
    predict_results = classifier_list[k].predict(input_fn=lambda: models.my_input_fn(FILE_PRACTICE, 1))
    i = 1
    for prediction in predict_results:
        print(
            "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}".format(
                i,
                prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
                prediction["MP01"], prediction["MP02"], prediction["MP03"], prediction["MP04"], prediction["MP05"],
                prediction["MP06"], prediction["MP07"], prediction["MP08"], prediction["MP09"], prediction["MP10"],
                prediction["MP11"], prediction["MP12"],
                prediction["Squeeze"], " Index ---> "))
        i = i + 1


# Let create a dataset for prediction
# It is from original file random()
prediction_input = {'vehicle': ["Small","Small","Small","Small","Mid","Mid","Mid","RV","RV","RV","RV"],
                    'Mp01': [3.2022093,3.1398065,3.2942017,6.5532746,3.8099214,2.3032416,7.3505315,4.8480404,5.047766,5.7952941,3.6771216],
                    'Mp02': [2.6485465,2.6650964,4.2525926,4.8693801,4.2538251,1.686367,5.5657804,3.2763543,4.6407705,6.0180962,2.8773542],
                    'Mp03': [4.480435,5.4473187,5.4135842,5.2599854,4.081369,1.9863109,5.5573553,4.471673,4.7784912,6.386367,2.9051203],
                    'Mp04': [4.5081212,3.8338425,5.9827548,5.6876672,3.6128938,1.9950785,5.7095601,3.4974604,4.5642672,8.1524887,3.8725659],
                    'Mp05': [4.3939385,4.2371326,6.1380491,5.059580,3.9146976,2.1221038,4.0639623,2.4475983,3.7237217,7.5504294,4.4166351],
                    'Mp06': [4.674751,10.0706279,7.9650342,5.1065518,4.0261236,2.6585385,6.7380661,3.0378084,3.9749315,13.1677902,6.6769886],
                    'Mp07': [5.3778967,6.0137504,7.0536957,20.4747898,4.4491594,2.6103166,5.8341228,4.3803837,5.7475077,14.3209523,8.0989227],
                    'Mp08': [5.475061,8.7609381,6.6416513,8.2920196,3.8133065,1.5197673,6.9709777,4.1229113,8.1771558,3.4223902,4.4379567],
                    'Mp09': [4.605879,3.087352,2.468536,3.1980702,3.6775033,2.3576642,1.7998759,4.6630176,3.9160913,2.6817308,2.5349934],
                    'Mp10': [4.8532357,6.2321219,4.9996724,8.0295619,5.8435842,4.3945977,5.1187231,4.1510166,6.3326299,3.7441892,4.0931377],
                    'Mp11': [7.0149714,7.1610133,11.0369594,10.5986617,6.1390255,3.5269413,6.7786751,5.3710064,10.2337745,4.2135063,4.4463526],
                    'Mp12': [12.7028538,6.686937,11.0733575,17.4228964,8.0387557,5.9244169,13.3056968,7.3354734,12.8865681,9.3598829,8.3120185]}
expected_value =[3.18,3.81,3.25,4.1,2.52,1.71,1.9,2.26,3.6,1.64,2.88]

tf.logging.info(" ***** Memory Base Predictions ***** ")

#
def new_input_fn():
    print(" -> Memory input acception ")
    def decode(x):
        print(tf.shape(x))
        x = tf.split(x, 13)  # Need to split into our 13 features
        return dict(zip(models.feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels

# First Model
# Predict all our prediction_input
new_predict_results = classifier_list[0].predict(input_fn=lambda:models.memory_input(prediction_input,None,1))
new_predict_results_2nd = classifier_list[1].predict(input_fn=lambda:models.memory_input(prediction_input,None,1))

# Print results
i = 1
aver_err = 0.0
tmp_values = []
for new_prediction in new_predict_results:
    print(
         "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}\t{17}{16:0.2f}\t{18:0.2f}%".format(
             i,
             new_prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
             new_prediction["MP01"], new_prediction["MP02"], new_prediction["MP03"], new_prediction["MP04"],
             new_prediction["MP05"], new_prediction["MP06"], new_prediction["MP07"], new_prediction["MP08"],
             new_prediction["MP09"], new_prediction["MP10"], new_prediction["MP11"], new_prediction["MP12"],
             new_prediction["Squeeze"],
             " Predict : ",
             expected_value[i-1],
             " Expected : ",
            ((abs(new_prediction["Squeeze"] -expected_value[i-1]))/expected_value[i-1])*100))
    aver_err = aver_err + ((abs(new_prediction["Squeeze"] -expected_value[i-1]))/expected_value[i-1])*100
    tmp_values.append(new_prediction["Squeeze"])
#    print("{0:0.2f}%".format(((abs(new_prediction["Squeeze"] / 1000000-expected_value[i-1]))/expected_value[i-1])*100))
    i = i + 1

# Print results : 2nd Model
i = 1
aver_err_2nd = 0.0
tmp_values_2nd = []
for new_prediction in new_predict_results_2nd:
    print(
         "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}\t{17}{16:0.2f}\t{18:0.2f}%".format(
             i,
             new_prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
             new_prediction["MP01"], new_prediction["MP02"], new_prediction["MP03"], new_prediction["MP04"],
             new_prediction["MP05"], new_prediction["MP06"], new_prediction["MP07"], new_prediction["MP08"],
             new_prediction["MP09"], new_prediction["MP10"], new_prediction["MP11"], new_prediction["MP12"],
             new_prediction["Squeeze"],
             " Predict : ",
             expected_value[i-1],
             " Expected : ",
            ((abs(new_prediction["Squeeze"] -expected_value[i-1]))/expected_value[i-1])*100))
    aver_err_2nd = aver_err_2nd + ((abs(new_prediction["Squeeze"] -expected_value[i-1]))/expected_value[i-1])*100
    tmp_values_2nd.append(new_prediction["Squeeze"])
#    print("{0:0.2f}%".format(((abs(new_prediction["Squeeze"] / 1000000-expected_value[i-1]))/expected_value[i-1])*100))
    i = i + 1

print("\n{0}\t{1:0.2f}%".format("1st Model Average Error : ", aver_err/len(tmp_values)))
print("{0}\t{1:0.2f}%".format("2nd Model Average Error : ", aver_err_2nd/len(tmp_values_2nd)))
#print(tmp_values)

average_prediction = []
average_err = []
average_err_total = 0.0
for j in range(len(tmp_values)):
    average_prediction.append((tmp_values[j]+tmp_values_2nd[j]) / 2.0)
    average_err.append((abs((tmp_values[j]+tmp_values_2nd[j]) / 2.0 - expected_value[j]) / expected_value[j]) * 100.0)
    average_err_total = average_err_total + (abs((tmp_values[j] + tmp_values_2nd[j]) / 2.0 - expected_value[j]) / expected_value[j]) * 100.0

print()
print("average_prediction : ")
print(average_prediction)
print("average_err : ")
print(average_err)

# total average error
print("\n{0}\t{1:0.2f}%".format("Total Model Average Error : ", average_err_total / len(average_err)))

tf.logging.info(" ***** Total End of Job ***** ")
