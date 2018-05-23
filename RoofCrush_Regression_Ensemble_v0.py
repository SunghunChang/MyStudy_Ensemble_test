####################################################
####                                            ####
#### 2018.04.25 SUNGHUN, CHANG                  ####
####                                            ####
#### For BIW Roofcrush Performance Prediction   ####
####                                            ####
####################################################

import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
import os
import Model_Definitions as models
import Model_Class
import sys
import six.moves.urllib.request as request

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=500, help='Number of Train Epoch. Default : 500')
parser.add_argument('--shuffle', type=int, default=256, help='Number of Train shuffle. Default : 256')
args = parser.parse_args()
# print(args.epoch)

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

## Create features
#feature_names = ['vehicle', 'Mp01', 'Mp02', 'Mp03', 'Mp04', 'Mp05', 'Mp06', 'Mp07', 'Mp08', 'Mp09', 'Mp10', 'Mp11', 'Mp12']
#
# All our inputs are feature columns of type numeric_column
#vehicle = tf.feature_column.categorical_column_with_vocabulary_list('vehicle', ['Small', 'Mid', 'RV'])
#feature_columns = [
#    tf.feature_column.indicator_column(vehicle)]
#for k in range(1,13):
#    feature_columns.append(tf.feature_column.numeric_column(feature_names[k]))
'''
    tf.feature_column.numeric_column(feature_names[1]),
    tf.feature_column.numeric_column(feature_names[2]),
    tf.feature_column.numeric_column(feature_names[3]),
    tf.feature_column.numeric_column(feature_names[4]),
    tf.feature_column.numeric_column(feature_names[5]),
    tf.feature_column.numeric_column(feature_names[6]),
    tf.feature_column.numeric_column(feature_names[7]),
    tf.feature_column.numeric_column(feature_names[8]),
    tf.feature_column.numeric_column(feature_names[9]),
    tf.feature_column.numeric_column(feature_names[10]),
    tf.feature_column.numeric_column(feature_names[11]),
    tf.feature_column.numeric_column(feature_names[12])
]
'''

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    # print(" ***** My Input Function Calling ***** ")
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[""], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], field_delim=',')
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(models.feature_names, features)), label
        return d

    with tf.name_scope('DATA_Feeding'):
        dataset = (tf.data.TextLineDataset(file_path)  # Read text file
            .skip(1)  # Skip header row
            .map(decode_csv)  # Decode each line
            .cache() # Warning: Caches entire dataset, can cause out of memory
            .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
            .repeat(repeat_count)    # Repeats dataset this # times
            .batch(32)
            .prefetch(1)  # Make sure you always have 1 batch ready to serve
        )
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# Create a custom estimator using my_model_fn
tf.logging.info("Before classifier construction")

tf.logging.info("1st MODEL CONSTRUCTION")

classifier = tf.estimator.Estimator(model_fn=models.my_model_fn,
                                    model_dir="./model_" + "1",
                                    params={
                                        "feature_columns" : models.feature_columns
                                    }
                                    )  # Path to where checkpoints etc are stored

tf.logging.info("2nd MODEL CONSTRUCTION")
classifier_2nd = tf.estimator.Estimator(model_fn=models.my_model_fn_2,
                                        model_dir="./model_" + "2",
                                        params = {
                                            "feature_columns": models.feature_columns
                                        }
                                        )

classifier_list = [classifier, classifier_2nd]

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

# 1st MODEL TRAIN
tf.logging.info("Before Train 1st MODEL")
train_result = classifier_list[0].train(input_fn=lambda: my_input_fn(FILE_TRAIN, args.epoch, args.shuffle)) # file path, repeat, shuffle
tf.logging.info("...done Train 1st MODEL")
tf.logging.info("{}".format(train_result))

# 2nd MODEL TRAIN
tf.logging.info("Before Train 2nd MODEL")
train_result_2nd = classifier_2nd.train(input_fn=lambda: my_input_fn(FILE_TRAIN, args.epoch, args.shuffle)) # file path, repeat, shuffle
tf.logging.info("...done Train 2nd MODEL")
tf.logging.info("{}".format(train_result_2nd))


# Display Weight and Bias
weight_model_1_layer_1 = classifier.get_variable_value('Model_1_Layers/First_Hidden_Layer/kernel')
bias_model_1_layer_1 = classifier.get_variable_value('Model_1_Layers/First_Hidden_Layer/bias')
#print(weight_model_1_layer_1)
#print(bias_model_1_layer_1)
#plt.imshow(weight_model_1_layer_1, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
#os.system('cls')
weight_model_1_layer_2 = classifier.get_variable_value('Model_1_Layers/Second_Hidden_Layer/kernel')
bias_model_1_layer_2 = classifier.get_variable_value('Model_1_Layers/Second_Hidden_Layer/bias')
#print(weight_model_1_layer_2)
#print(bias_model_1_layer_2)
#plt.imshow(weight_model_1_layer_2, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
#os.system('cls')
weight_model_1_layer_3 = classifier.get_variable_value('Model_1_Layers/Third_Hidden_Layer/kernel')
bias_model_1_layer_3 = classifier.get_variable_value('Model_1_Layers/Third_Hidden_Layer/bias')
#print(weight_model_1_layer_3)
#print(bias_model_1_layer_3)
#plt.imshow(weight_model_1_layer_3, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
#os.system('cls')
weight_model_2_layer_1 = classifier_2nd.get_variable_value('Model_2_Layers/First_Hidden_Layer/kernel')
bias_model_2_layer_1 = classifier_2nd.get_variable_value('Model_2_Layers/First_Hidden_Layer/bias')
#print(weight_model_2_layer_1)
#print(bias_model_2_layer_1)
#plt.imshow(weight_model_2_layer_1, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
#os.system('cls')
weight_model_2_layer_2 = classifier_2nd.get_variable_value('Model_2_Layers/Second_Hidden_Layer/kernel')
bias_model_2_layer_2 = classifier_2nd.get_variable_value('Model_2_Layers/Second_Hidden_Layer/bias')
#print(weight_model_2_layer_2)
#print(bias_model_2_layer_2)
#plt.imshow(weight_model_2_layer_2, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
#os.system('cls')
weight_model_2_layer_3 = classifier_2nd.get_variable_value('Model_2_Layers/Third_Hidden_Layer/kernel')
bias_model_2_layer_3 = classifier_2nd.get_variable_value('Model_2_Layers/Third_Hidden_Layer/bias')
#print(weight_model_2_layer_3)
#print(bias_model_2_layer_3)
#plt.imshow(weight_model_2_layer_3, cmap='gray')
#plt.show()
#a = input("Press Any Key to Resume")
os.system('cls')

'''
fig = plt.figure()
fig.suptitle('Weight of Models')
im1 = fig.add_subplot(2,3,1)
im1.set_title('1st Layer Weight [1st Model]')
im2 = fig.add_subplot(2,3,2)
im2.set_title('2nd Layer Weight [1st Model]')
im3 = fig.add_subplot(2,3,3)
im3.set_title('3rd Layer Weight [1st Model]')
im1.imshow(weight_model_1_layer_1, cmap='gray')
im2.imshow(weight_model_1_layer_2, cmap='gray')
im3.imshow(weight_model_1_layer_3, cmap='gray')

im4 = fig.add_subplot(2,3,4)
im4.set_title('1st Layer Weight [2nd Model]')
im5 = fig.add_subplot(2,3,5)
im5.set_title('2nd Layer Weight [2nd Model]')
im6 = fig.add_subplot(2,3,6)
im6.set_title('3rd Layer Weight [2nd Model]')
im4.imshow(weight_model_2_layer_1, cmap='gray')
im5.imshow(weight_model_2_layer_2, cmap='gray')
im6.imshow(weight_model_2_layer_3, cmap='gray')

plt.show()
'''
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
ax1.set_title('1st Layer Weight [1st Model]')
im1 = ax1.imshow(weight_model_1_layer_1, cmap='gray') #, aspect='auto')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right",size="5%",pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1, ticks=MultipleLocator(0.2), format="%.1f")
#ax1.xaxis.set_visible(False)
#ax1.set_yticks([-2.0, 2.0])

ax2.set_title('2nd Layer Weight [1st Model]')
im2 = ax2.imshow(weight_model_1_layer_2, cmap='gray', aspect='auto')
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right",size="5%",pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2, ticks=MultipleLocator(0.2), format="%.1f")

ax3.set_title('3rd Layer Weight [1st Model]')
im3 = ax3.imshow(weight_model_1_layer_3, cmap='gray', aspect='auto')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes("right",size="5%",pad=0.05)
cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.1f")

ax4.set_title('1st Layer Weight [2nd Model]')
im4 = ax4.imshow(weight_model_2_layer_1, cmap='gray') #, aspect='auto')
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right",size="5%",pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4, ticks=MultipleLocator(0.2), format="%.1f")

ax5.set_title('2nd Layer Weight [2nd Model]')
im5 = ax5.imshow(weight_model_2_layer_2, cmap='gray', aspect='auto')
divider5 = make_axes_locatable(ax5)
cax5 = divider5.append_axes("right",size="5%",pad=0.05)
cbar5 = plt.colorbar(im5, cax=cax5, ticks=MultipleLocator(0.2), format="%.1f")

ax6.set_title('3rd Layer Weight [2nd Model]')
im6 = ax6.imshow(weight_model_2_layer_3, cmap='gray', aspect='auto')
divider6 = make_axes_locatable(ax6)
cax6 = divider6.append_axes("right",size="5%",pad=0.05)
cbar6 = plt.colorbar(im6, cax=cax6, ticks=MultipleLocator(0.2), format="%.1f")

plt.show()

'''
plt.plot(bias_model_1_layer_1)
plt.plot(bias_model_1_layer_2)
plt.plot(bias_model_1_layer_3)
plt.plot(bias_model_2_layer_1)
plt.plot(bias_model_2_layer_2)
plt.plot(bias_model_2_layer_3)
plt.show()
'''

bias_figure = plt.figure()
bias_fig = bias_figure.add_subplot(1,1,1)
bias_fig.set_title("Bias Values for All Layers")
bias_fig.set_xlabel("per NODE")
# bias_fig.axes.get_xaxis().set_ticks([]) # It hide grid lines
bias_fig.axes.get_xaxis().set_ticklabels([]) # It remains grid line
# bias_fig.axes.get_xaxis().set_visible(False) # It hide all ??
bias_fig.set_ylabel("bias value")
bias_fig.plot(bias_model_1_layer_1, color='b', marker=".", label="Model 1 - Layer 1", linestyle="--")
bias_fig.plot(bias_model_1_layer_2, color='g', marker=".", label="Model 1 - Layer 2", linestyle="--")
bias_fig.plot(bias_model_1_layer_3, color='r', marker=".", label="Model 1 - Layer 3", linestyle="--")
bias_fig.plot(bias_model_2_layer_1, color='c', marker=".", label="Model 2 - Layer 1", linestyle="--")
bias_fig.plot(bias_model_2_layer_2, color='m', marker=".", label="Model 2 - Layer 2", linestyle="--")
bias_fig.plot(bias_model_2_layer_3, color='y', marker=".", label="Model 2 - Layer 3", linestyle="--")
bias_fig.legend(loc="best")
bias_fig.grid(linestyle=':', linewidth=1)
plt.show()

# Evaluate the model using the data contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
tf.logging.info("Before Evaluate 1st MODEL")
evaluate_result = classifier.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, 1))
tf.logging.info("...done Evaluate 1st MODEL")

tf.logging.info("Evaluation results of 1st MODEL")
tf.logging.info(" ***** Evaluation results *****")
tf.logging.info(" *****      1st MODEL     *****")
for key in evaluate_result:
    tf.logging.info("   {} ---> {}".format(key, evaluate_result[key]))
# print("__________________________________________________________________>>>> ACCURACY : {0:0.2f}%".format(evaluate_result["rmse"] / 1000000))
tf.logging.info(" ******************************")

#2nd evaluation
tf.logging.info("Before Evaluate 2nd MODEL")
evaluate_result_2nd = classifier_2nd.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, 1))
tf.logging.info("...done Evaluate 2nd MODEL")

tf.logging.info("Evaluation results of 2nd MODEL")
tf.logging.info(" ***** Evaluation results *****")
tf.logging.info(" *****      2nd MODEL     *****")
for key in evaluate_result_2nd:
    tf.logging.info("   {} ---> {}".format(key, evaluate_result_2nd[key]))
tf.logging.info(" ******************************")

#
# Predict 1st in the data in FILE_TEST, repeat only once.
predict_results = classifier.predict(input_fn=lambda: my_input_fn(FILE_PRACTICE, 1))

tf.logging.info(" ***** Prediction on test file ***** ")
tf.logging.info(" *****        1st MODEL        ***** ")

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

# Predict 2nd
predict_results_2nd = classifier_2nd.predict(input_fn=lambda: my_input_fn(FILE_PRACTICE, 1))
tf.logging.info(" ***** Prediction on test file ***** ")
tf.logging.info(" *****        2nd MODEL        ***** ")

i = 1
for prediction in predict_results_2nd:
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
'''
prediction_input = [["Small",3293645.567,2715268.166,4618940.65,4413955.593,4480569.725,4552422.069,5431506.792,5563818.164,473107.6725,4881040.94,7045602.396,12696502.03],  # 3.2365
                    ["Small",3318218.418,2600008.29,4612452.857,4474004.258,4556371.066,4678142.104,5496556.892,5691566.969,468235.5288,4750460.695,6866286.919,12308454.28], # 3.3037
                    ["Mid",2921273.767,2177641.446,5442658.679,4851815.006,5082138.491,4042851.847,5099255.254,5796905.869,1608061.116,5853879.927,8339309.588,11762607.14],  # 1.9793
                    ["Mid",2170544.56,1758619.822,4354809.471,3279413.545,3349865.656,3303847.143,6234404.299,4068885.507,1623483.656,6999403.505,7781783.337,9622231.531],   # 2.8941
                    ["Mid",2238351.122,1668485.997,2003292.317,1973515.219,2209203.303,2652439.601,2605576.788,1525788.73,2342003.69,4275505.64,3511535.296,6074749.822],     # 1.6860
                    ["RV",5687829.643,5454108.154,5597662.465,6195690.931,5657034.246,7137244.598,8338401.036,6576310.191,3773925.522,4935750.015,8899399.268,12519402.42],   # 3.8611
                    ["RV",3633324.819,2962522.398,2856583.851,3764539.384,4491660.64,6689820.516,8132642.118,4508807.686,2466626.362,4079588.813,4534564.643,8127702.292]]    # 2.8237
'''
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
# print(" **** Manual Predictions ****")

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

def memory_input(features, labels, batch_size=1):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "Insert batch size"
    dataset = dataset.batch(batch_size)
    return dataset


# First Model
# Predict all our prediction_input
new_predict_results = classifier.predict(input_fn=lambda:memory_input(prediction_input,None,1))
new_predict_results_2nd = classifier_2nd.predict(input_fn=lambda:memory_input(prediction_input,None,1))

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
