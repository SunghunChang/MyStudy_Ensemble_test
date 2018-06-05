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
import logging
import logging.handlers
import time
now = time.localtime()
now_string = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

log_dir = "./00_Run_Logging"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
img_dir = "./01_Run_Weight_n_Bias"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
'''
eval_hook_dir = "./01_Eval_Hooks_etc"
if not os.path.exists(eval_hook_dir):
    os.makedirs(eval_hook_dir)
'''
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(funcName)s - %(asctime)s - %(message)s')
fh = logging.handlers.RotatingFileHandler('./00_Run_Logging/INFO_logging_CSH.log', maxBytes= 1024 * 1024 * 10, backupCount = 10) # 10MB - 10 files
fh.setFormatter(formatter)
log.addHandler(fh)
'''
fh = logging.FileHandler('./SunghunChang_Model.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)
'''

Result_File = open(log_dir + "/Run_Results.log", 'a')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=500, help='Number of Train Epoch. Default : 500')
parser.add_argument('--shuffle', type=int, default=256, help='Number of Train shuffle. Default : 256')
parser.add_argument('--batch', type=int, default=32, help='Number of Train batch size. Default : 32')
parser.add_argument('--model', type=int, default=2, help='Number of Train Model. Default : 2')
parser.add_argument('--chkpt', type=int, default=1000, help='Save Check Point Every N steps and Display Custom Logging Every N Iter. Default : 1000')
parser.add_argument('--chkptnum', type=int, default=10, help='Maximum Check Point Files that will be saved. Default : 10')
parser.add_argument('--wbplot', type=str2bool, default=True, help='Visualization of Weight and Bias at end of Training. Default : True')
args = parser.parse_args()
# print(args.epoch)

Num_Of_Models = args.model

tf.logging.set_verbosity(tf.logging.INFO)
#tf.logging.log_every_n(tf.logging.INFO, "100_steps_run", 100)

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

Result_File.write("\n")
Result_File.write("        ************************************************** \n")
Result_File.write("        *****      Prediction for BIW RoofCrush      ***** \n")
Result_File.write("        *****             SUNGHUN, CHANG             ***** \n")
Result_File.write("        *****       TensorFlow version : {}       ***** \n".format(tf_version))
Result_File.write("        ************************************************** \n")
Result_File.write("                  RUN DATE : {}\n".format(now_string))
Result_File.write("\n")

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

Result_File.write("\n")
Result_File.write("        ** Default PATH : {}\n".format(PATH))
Result_File.write("        ** TRAIN : {}\n".format(FILE_TRAIN))
Result_File.write("        ** TEST : {}\n".format(FILE_TEST))
Result_File.write("\n         -- Number of DNN Models     : {}\n".format(args.model))
Result_File.write("         -- Number of Train Epoch    : {}\n".format(args.epoch))
Result_File.write("         -- Train DATA Batch Size    : {}\n".format(args.batch))
Result_File.write("         -- Train DATA Shuffle Count : {}\n".format(args.shuffle))
Result_File.write("         -- Check Point File Count   : {}\n".format(args.chkptnum))
Result_File.write("         -- Check Point Saving / Custom Logging Every {} Steps\n".format(args.chkpt))
Result_File.write("         -- Display Weight and Bias  : {}\n\n".format(args.wbplot))
Result_File.write("\n")

# Create a custom estimator using my_model_fn
tf.logging.info("Before classifier construction")

validation_metrics = {"accuracy":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_accuracy,
            prediction_key="classes"
        ),
        "precision":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_precision,
            prediction_key="classes"
        ),
        "recall":
        tf.contrib.learn.MetricSpec(
            metric_fn=tf.contrib.metrics.streaming_recall,
            prediction_key="classes"
        )}

classifier_list = []
validation_monitor_list = []

#run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'cpu': 0}))
run_config = tf.estimator.RunConfig(session_config=tf.ConfigProto(device_count={'/GPU':0}),
                                    save_checkpoints_steps=args.chkpt,
                                    save_summary_steps=1, # Its default is 100
                                    keep_checkpoint_max= args.chkptnum)
for k in range(1, Num_Of_Models + 1):
    tf.logging.info("MODEL #" + str(k) + " CONSTRUCTION")
    classifier_list.append(tf.estimator.Estimator(model_fn=models.my_model_fn,
                                                  model_dir="./model_" + str(k),
                                                  #config=tf.contrib.learn.RunConfig(save_checkpoints_steps=100),
                                                  config=run_config,
                                                  params=
                                                  {
                                                      "feature_columns": models.feature_columns,
                                                      "model_identifier": str(k),
                                                      "train_logging" : args.chkpt
                                                      # "batch" : args.batch
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
    #summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir='./01_Eval_Hooks_etc', scaffold=tf.train.Scaffold(), summary_op=tf.summary.merge_all())
    train_result = classifier_list[k - 1].train(
        input_fn=lambda: models.my_input_fn(FILE_TRAIN, args.epoch, args.shuffle, args.batch))  # file path, repeat, shuffle, batch
        #hooks=[summary_hook])
    #train_result_list.append(train_result)
    #tf.logging.info("TRAINING RESULT of MODEL #" + str(k))
    #tf.logging.info("{}".format(train_result))
    tf.logging.info("END of MODEL #" + str(k) + " TRAINING")
    Result_File.write("MODEL #" + str(k) + " TRAINING IS DONE NORMALLY\n")

# below 3 lines are not work
#    for trained in train_result:
#        print("test_PRINT : " + trained["_total_loss"])
#    print("test_PRINT : " + train_result["training_hooks"])

# Display Weight and Bias
for k in range(0, Num_Of_Models):
    weight_layer_1 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/First_Hidden_Layer/kernel')
    bias_layer_1 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/First_Hidden_Layer/bias')

    weight_layer_2 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Second_Hidden_Layer/kernel')
    bias_layer_2 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Second_Hidden_Layer/bias')

    weight_layer_3 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Third_Hidden_Layer/kernel')
    bias_layer_3 = classifier_list[k].get_variable_value('Model_' + str(k + 1) + '_Layers/Third_Hidden_Layer/bias')

    varplots.PlotWeighNbias(weight_layer_1, weight_layer_2, weight_layer_3, bias_layer_1, bias_layer_2, bias_layer_3, k, args.wbplot)

    os.system('cls')

##################################################### Evaluation #######################################################
# Evaluate the model using the data contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
loss_values = []
rmse_values = []
for k in range(0,Num_Of_Models):
    tf.logging.info("Evaluation results of 1st MODEL")
    tf.logging.info(" ***** Evaluation results *****")
    tf.logging.info(" *****      #{} MODEL      *****".format(str(k + 1)))
    Result_File.write("\n*********************************\n")
    Result_File.write("Evaluation results of 1st MODEL\n")
    # summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir='./01_Eval_Hooks_etc', scaffold=tf.train.Scaffold(), summary_op=tf.summary.merge_all())
    evaluate_result = classifier_list[k].evaluate(input_fn=lambda: models.my_input_fn(FILE_TEST, 1)) #,hooks=[summary_hook])
    for key in evaluate_result:
        tf.logging.info("   {} ---> {}".format(key, evaluate_result[key]))
        Result_File.write("   {} ---> {}\n".format(key, evaluate_result[key]))
        if str(key) == 'rmse' :
            rmse_values.append(evaluate_result[key])
        if str(key) == 'loss' :
            loss_values.append(evaluate_result[key])
    tf.logging.info(" ******************************")
    tf.logging.info(" ******************************")
    Result_File.write("*********************************\n")

##################################################### Prediction #######################################################

# Prediction from file and Compare with Label - CSH made
#tmp_features, tmp_labels = models.my_input_fn(FILE_PRACTICE, 1)
_, tmp_labels = models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels) #Extract Tensor to List
    print(labels)

# Predict in the data in FILE_TEST, repeat only once.
error_file_total_model = []
tf.logging.info(" ***** Prediction on test file ***** ")
Result_File.write("\n ***** Prediction on test file ***** \n")

total_file_predictions = []
error_for_each_model = []
for k in range(0,Num_Of_Models):
    tf.logging.info(" *****        {}st MODEL        ***** ".format(str(k + 1)))
    Result_File.write(" *****        {}st MODEL        ***** \n".format(str(k + 1)))
    predict_results = classifier_list[k].predict(input_fn=lambda:models.my_input_fn(FILE_PRACTICE, repeat_count=1, batch_size=32, shuffle_count=1))
    i = 1
    error_file = []
    error_file_sum = 0.0
    temp_predict = []
    for prediction, expec in zip(predict_results, labels):
        error_file.append((abs(prediction["Squeeze"] - expec)/expec) * 100)
        prediction_print = "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}\t{16}{17:0.2f}\t{18:0.2f}%".format(
                i,
                prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
                prediction["MP01"], prediction["MP02"], prediction["MP03"], prediction["MP04"], prediction["MP05"],
                prediction["MP06"], prediction["MP07"], prediction["MP08"], prediction["MP09"], prediction["MP10"],
                prediction["MP11"], prediction["MP12"],
                prediction["Squeeze"], "Predict : ", "Expected : ", expec, (abs(prediction["Squeeze"] - expec)/expec) * 100)
        print(prediction_print)
        Result_File.write(prediction_print + "\n")
        temp_predict.append(prediction["Squeeze"])
        i = i + 1
    total_file_predictions.append(temp_predict)
    print("Model #{0} Average Error on File Inputs : {1:0.2f}%".format(str(k+1), sum(error_file)/len(error_file)))
    Result_File.write("\nModel #{0} Average Error on File Inputs : {1:0.2f}%".format(str(k+1), sum(error_file)/len(error_file)) + "\n\n")
    error_for_each_model.append(sum(error_file)/len(error_file))

# For Total Models
average_prediction_file = []
average_err_file = []
for j in range(0, len(temp_predict)):
    temp_prediction = 0.0
    for k in range(0, Num_Of_Models):
        temp_prediction = temp_prediction + total_file_predictions[k][j]
    average_prediction_file.append(temp_prediction / float(Num_Of_Models))
    average_err_file.append((abs((temp_prediction / float(Num_Of_Models)) - labels[j]) / labels[j]) * 100)

Result_File.write("\n")
print("")

for z in range(0,len(error_for_each_model)):
    Result_File.write("Model #{0} Error : {1:0.2f}%\n".format(str(z+1), error_for_each_model[z]))
    print("Model #{0} Error : {1:0.2f}%".format(str(z+1), error_for_each_model[z]))

print("***********************************")
print("Average ERROR for " + str(int(Num_Of_Models)) + " Models : " + "{0:0.2f}%".format(sum(average_err_file) / len(average_err_file)))
print("***********************************")
Result_File.write("\n************************************\n")
Result_File.write("Average ERROR for " + str(int(Num_Of_Models)) + " Models : " + "{0:0.2f}%\n".format(sum(average_err_file) / len(average_err_file)))
Result_File.write("\n************************************\n")


tf.logging.info(" ***** Memory Base Predictions ***** ")
Result_File.write("\n ***** Memory Base Predictions ***** \n")
#
def new_input_fn():
    print(" -> Memory input accepting ")
    def decode(x):
        #print(tf.shape(x))
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
    Result_File.write("\n ***** Using MODEL # {0:2d} Predictions ***** \n".format(k+1))
    for new_prediction in new_predict_results:
        prediction_print = "{0:d}\t{1}\t{2:0.1f}\t{3:0.1f}\t{4:0.1f}\t{5:0.1f}\t{6:0.1f}\t{7:0.1f}\t{8:0.1f}\t{9:0.1f}\t{10:0.1f}\t{11:0.1f}\t{12:0.1f}\t{13:0.1f}\t{15}{14:0.2f}\t{17}{16:0.2f}\t{18:0.2f}%".format(
                i,
                new_prediction["vehicle_type"].decode('utf-8').ljust(5).upper(),
                new_prediction["MP01"], new_prediction["MP02"], new_prediction["MP03"], new_prediction["MP04"],
                new_prediction["MP05"], new_prediction["MP06"], new_prediction["MP07"], new_prediction["MP08"],
                new_prediction["MP09"], new_prediction["MP10"], new_prediction["MP11"], new_prediction["MP12"],
                new_prediction["Squeeze"],
                " Predict : ",
                mpd.expected_value[i - 1],
                " Expected : ",
                ((abs(new_prediction["Squeeze"] - mpd.expected_value[i - 1])) / mpd.expected_value[i - 1]) * 100)
        print(prediction_print)
        Result_File.write(prediction_print + "\n")
        error_sum = error_sum + (
                    (abs(new_prediction["Squeeze"] - mpd.expected_value[i - 1])) / mpd.expected_value[i - 1]) * 100
        tmp_values.append(new_prediction["Squeeze"])
        #    print("{0:0.2f}%".format(((abs(new_prediction["Squeeze"] / 1000000-expected_value[i-1]))/expected_value[i-1])*100))
        i = i + 1
    error_total_model.append(error_sum/len(tmp_values))
    prediction_total.append(tmp_values)

print("\n")
Result_File.write("\n\n")
for k in range(0,Num_Of_Models):
    print("#{2:d}{0}\t{1:0.2f}%".format(" Model Average Error on Memory : ", error_total_model[k], k))
    Result_File.write("#{2:d}{0}\t{1:0.2f}%\n".format(" Model Average Error on Memory : ", error_total_model[k], k))

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

Result_File.write("\n")
Result_File.write("average_prediction : {}\n".format(average_prediction))
Result_File.write("average_err : {}\n".format(average_err))

average_err_total = 0.0
for j in range(len(average_err)):
    average_err_total = average_err_total + average_err[j]

tot_rmse = 0.0
tot_loss = 0.0
print()
Result_File.write("\n")

for j in range(0, Num_Of_Models):
    print("Model #" + str(int(j + 1)) + " : rmse [" + "{0:0.5f}".format(rmse_values[j]) + "]" + " // loss [" + "{0:0.5f}".format(loss_values[j]) + "]")
    Result_File.write("Model #" + str(int(j + 1)) + " : rmse [" + "{0:0.5f}".format(rmse_values[j]) + "]" + " // loss [" + "{0:0.5f}".format(loss_values[j]) + "]\n")
    tot_rmse = tot_rmse + rmse_values[j]
    tot_loss = tot_loss + loss_values[j]


print()
print("Average RMSE for " + str(int(Num_Of_Models)) + " Models : " + str("{0:0.5f}".format(tot_rmse/Num_Of_Models)))
print("Average LOSS for " + str(int(Num_Of_Models)) + " Models : " + str("{0:0.5f}".format(tot_loss/Num_Of_Models)))

Result_File.write("\n")
Result_File.write("Average RMSE for " + str(int(Num_Of_Models)) + " Models : " + str("{0:0.5f}".format(tot_rmse/Num_Of_Models)) + "\n")
Result_File.write("Average LOSS for " + str(int(Num_Of_Models)) + " Models : " + str("{0:0.5f}".format(tot_loss/Num_Of_Models)) + "\n")

# total average error
#print("\n{0}\t{1:0.2f}%".format("Total Model Average Error : ", average_err_total / len(average_err)))
print("Average ERROR for " + str(int(Num_Of_Models)) + " Models : " + "{0:0.3f}%".format(average_err_total / len(average_err)))
Result_File.write("Average ERROR for " + str(int(Num_Of_Models)) + " Models : " + "{0:0.3f}%\n".format(average_err_total / len(average_err)))

print("\n***************************************************")
print("** Number of Models : " + str(args.model))
print("** Number of Epoch : " + str(args.epoch))
print("** Number of Shuffle [for DATA] : " + str(args.shuffle))
print("** Save Check Point every " + str(args.chkpt) + " steps")
print("***************************************************\n")

tf.logging.info(" ***** Total End of Job ***** ")

Result_File.write("\n***************************************************\n")
Result_File.write("** Number of Models : " + str(args.model) + "\n")
Result_File.write("** Number of Epoch : " + str(args.epoch) + "\n")
Result_File.write("** Number of Shuffle [for DATA] : " + str(args.shuffle) + "\n")
Result_File.write("** Save Check Point every " + str(args.chkpt) + " steps\n")
Result_File.write("***************************************************\n")
Result_File.write("\n ******************************************************** Total End of Job ******************************************************** \n\n")

Result_File.close()

#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================

# To Display Tensor
#print(FILE_PRACTICE)
#tmp_features, tmp_labels = models.my_input_fn(FILE_PRACTICE, 1)
#with tf.Session() as sess:
#    #print(sess.run(tmp_features))
#    print(sess.run(tmp_labels))

#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================

'''
# Prediction from file and Compare with Label - CSH made...It Works
_, tmp_labels = models.my_input_fn(FILE_PRACTICE, 1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels)
file_predictions = classifier_list[0].predict(input_fn=lambda: models.my_input_fn(FILE_TEST, 1))

for pred_dict, expec in zip(file_predictions, labels):
    result_value = pred_dict['Squeeze']
    print("Prediction : {0:0.2f} / Expected {1:0.2f}".format(pred_dict['Squeeze'],expec))
'''


''' It is not work for Estimator
#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================

tmp_features, tmp_labels = models.my_input_fn(FILE_PRACTICE, 1)

#Rebuild Models
Tmp_Predictions = models.my_model_fn(tmp_features,tmp_labels,tf.estimator.ModeKeys.EVAL,params=
                                                  {
                                                      "feature_columns": models.feature_columns,
                                                      "model_identifier": str(1), #str(k),
                                                      "train_logging" : args.chkpt
                                                  }).predictions
saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model_1/model.ckpt-6522.data-00000-of-00001')
    saver.restore(sess, './model_1') #ckpt.model_checkpoint_path)

    tmp_prediction_values = []
    tmp_label_values = []
    while True:
        try:
            preds, lbls = sess.run([Tmp_Predictions, tmp_labels])
            tmp_prediction_values += preds
            tmp_label_values += lbls
        except tf.errors.OutOfRangeError:
            break
    print(tmp_prediction_values)
    print(tmp_label_values)

#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================
'''