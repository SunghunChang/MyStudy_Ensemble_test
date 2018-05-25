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
import Memory_Prediction_DATA as mpd
import Model_Class
import sys
import six.moves.urllib.request as request

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=500, help='Number of Train Epoch. Default : 500')
parser.add_argument('--shuffle', type=int, default=256, help='Number of Train shuffle. Default : 256')
parser.add_argument('--model', type=int, default=2, help='Number of Train Model. Default : 2')
args = parser.parse_args()
# print(args.epoch)

Num_Of_Models = args.model

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
for k in range(1, Num_Of_Models + 1):
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

###################################################### Training ########################################################
train_result_list = []
for k in range(1, Num_Of_Models + 1):
    tf.logging.info("Train MODEL #" + str(k))
    train_result = classifier_list[k - 1].train(
        input_fn=lambda: models.my_input_fn(FILE_TRAIN, args.epoch, args.shuffle))  # file path, repeat, shuffle
    #train_result_list.append(train_result)
    #tf.logging.info("TRAINING RESULT of MODEL #" + str(k))
    #tf.logging.info("{}".format(train_result))
    tf.logging.info("END of MODEL #" + str(k) + " TRAINING")

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

##################################################### Evaluation #######################################################
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

##################################################### Prediction #######################################################
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


tf.logging.info(" ***** Memory Base Predictions ***** ")

#
def new_input_fn():
    print(" -> Memory input acception ")
    def decode(x):
        print(tf.shape(x))
        x = tf.split(x, 13)  # Need to split into our 13 features
        return dict(zip(models.feature_names, x))  # To build a dict of them

    dataset = tf.data.Dataset.from_tensor_slices(mpd.prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None  # In prediction, we have no labels

# Predict all our prediction_input
error_total_model = []
prediction_total = []
for k in range(0,Num_Of_Models):
    new_predict_results = classifier_list[k].predict(input_fn=lambda:models.memory_input(mpd.prediction_input,None,1))
    i = 1
    error_sum = 0.0
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
                mpd.expected_value[i - 1],
                " Expected : ",
                ((abs(new_prediction["Squeeze"] - mpd.expected_value[i - 1])) / mpd.expected_value[i - 1]) * 100))
        error_sum = error_sum + (
                    (abs(new_prediction["Squeeze"] - mpd.expected_value[i - 1])) / mpd.expected_value[i - 1]) * 100
        tmp_values.append(new_prediction["Squeeze"])
        #    print("{0:0.2f}%".format(((abs(new_prediction["Squeeze"] / 1000000-expected_value[i-1]))/expected_value[i-1])*100))
        i = i + 1
    error_total_model.append(error_sum/len(tmp_values))
    prediction_total.append(tmp_values)

print("\n")
for k in range(0,Num_Of_Models):
    print("#{2:d}{0}\t{1:0.2f}%".format(" Model Average Error : ", error_total_model[k], k))

#Calculate Average Prediction
average_prediction = []
average_err = []
for j in range(len(tmp_values)):
    temp_prediction = 0.0
    for k in range(0, Num_Of_Models):
        temp_prediction = temp_prediction + prediction_total[k][j]
    average_prediction.append(temp_prediction / Num_Of_Models)
    average_err.append(((abs(temp_prediction / Num_Of_Models - mpd.expected_value[j])) / mpd.expected_value[j]) * 100.0)

print()
print("average_prediction : ")
print(average_prediction)
print("average_err : ")
print(average_err)

average_err_total = 0.0
for j in range(len(average_err)):
    average_err_total = average_err_total + average_err[j]
# total average error
print("\n{0}\t{1:0.2f}%".format("Total Model Average Error : ", average_err_total / len(average_err)))

tf.logging.info(" ***** Total End of Job ***** ")
