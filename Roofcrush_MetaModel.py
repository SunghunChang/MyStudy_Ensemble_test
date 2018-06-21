#####################################################
####                                             ####
####  2018.06.19 SUNGHUN, CHANG                  ####
####                                             ####
####  For BIW Roofcrush Performance Prediction   ####
####       - Using Ensemble Meta Model -         ####
####                                             ####
#####################################################

import logging.handlers
import argparse
import tensorflow as tf
import os
import Model_Definitions_Expand_Features_v2 as models

os.system('cls')
tf.logging.set_verbosity(tf.logging.INFO)

DATA_dir = "./02_DATA_for_META_Regression"
DATA_TRAIN= "DATA_for_META_Regression_TRAIN.csv"
DATA_TEST= "DATA_for_META_Regression_TEST.csv"

DATA_file_format = []

print()
print("##### Build Meta-Regressor Model #####")
print()
cnt = 0
for modeldirfind in os.listdir("./"):
    if os.path.isdir(modeldirfind) :
        if modeldirfind.startswith("model_") :
            cnt = cnt + 1

for j in range(0,cnt+1):
    DATA_file_format.append([0.])

meta_feature_names = []
for i in range(0,cnt):
    meta_feature_names.append("Model_" + str(i+1).rjust(3,'0'))
meta_feature_names.append("Labels")

meta_feature_columns = []
for k in range(0,cnt):
    meta_feature_columns.append(tf.feature_column.numeric_column(meta_feature_names[k]))

def meta_input_fn(file_path, repeat_count=1, shuffle_count=1, batch_size = 32):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, DATA_file_format, field_delim=',')
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(meta_feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path).skip(1).map(decode_csv).cache().shuffle(shuffle_count)
        .repeat(repeat_count).batch(batch_size).prefetch(1))
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels

def meta_model_fn(features, labels, mode,params):
    regularizer = tf.keras.regularizers.l2(l=0.01)
    initializer = tf.keras.initializers.glorot_normal(seed=None)
    input_layer = tf.feature_column.input_layer(features, params["feature_columns"])
    h1 = tf.layers.Dense(units=30,activation=tf.nn.relu,
                         kernel_initializer = initializer,kernel_regularizer = regularizer,
                         activity_regularizer = None,name="First_Hidden_Layer")(input_layer)
    h1 = tf.layers.batch_normalization(h1, training=mode == tf.estimator.ModeKeys.TRAIN)
    h1 = tf.nn.relu(h1)
    h2 = tf.layers.Dense(units=30,activation=tf.nn.relu,
                         kernel_initializer = initializer,kernel_regularizer = regularizer,
                         activity_regularizer = None,name="Second_Hidden_Layer")(h1)
    h2 = tf.layers.batch_normalization(h2, training=mode == tf.estimator.ModeKeys.TRAIN)
    h2 = tf.nn.relu(h2)
    h3 = tf.layers.Dense(units=30, activation=tf.nn.relu,
                         kernel_initializer=initializer, kernel_regularizer=regularizer,
                         activity_regularizer=None, name="Second_Hidden_Layer")(h2)
    h3 = tf.layers.batch_normalization(h2, training=mode == tf.estimator.ModeKeys.TRAIN)
    h3 = tf.nn.relu(h3)
    output_layer = tf.layers.Dense(units = 1, name = "Output_Layer")(h3)

    # 나중에 dictionary key 추가 방식으로 바꾸자
    predictions = {'Squeeze': tf.squeeze(output_layer, 1),
                   'M01': features['Model_001'],
                   'M02': features['Model_002'],
                   'M03': features['Model_003'],
                   'M04': features['Model_004'],
                   'M05': features['Model_005'],
                   'M06': features['Model_006'],
                   'M07': features['Model_007'],
                   'M08': features['Model_008'],
                   'M09': features['Model_009'],
                   'M10': features['Model_010'],
                   'M11': features['Model_011'],
                   'M12': features['Model_012']
                   }

    ################################################ 1. Prediction mode ################################################
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ################################################ 2. Training mode ##################################################
    average_loss = tf.losses.mean_squared_error(labels, predictions['Squeeze'])
    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, predictions['Squeeze'])))

    with tf.name_scope('my_summary_training'):
        tf.summary.scalar('batch_size', batch_size)
        tf.summary.scalar('Total_Loss', total_loss)
        tf.summary.scalar('Mean_Squared_Error_aver.loss', average_loss)
        tf.summary.scalar('Root_Mean_Squared_Error', rmse)

    logging_hook = tf.train.LoggingTensorHook({"_average_loss[mse]":average_loss,
                                               "_total_loss":total_loss,
                                               "_batch_size":batch_size},
                                               every_n_iter=1000)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope('Training_Stage'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(0.001, name="My_Optimizer").minimize(loss=average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=average_loss, train_op=train_op, training_hooks=[logging_hook], predictions=predictions)

    assert mode == tf.estimator.ModeKeys.EVAL, "EVAL is only ModeKey left"
    rmse = tf.metrics.root_mean_squared_error(labels, predictions['Squeeze'])
    eval_metrics =  {"rmse": rmse}

    ################################################ 3. Evaluation mode ################################################
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=average_loss, eval_metric_ops=eval_metrics, predictions = predictions)

run_config = tf.estimator.RunConfig(session_config=tf.ConfigProto(device_count={'/GPU':0}),
                                    save_checkpoints_steps=100,
                                    save_summary_steps = 10, # Its default is 100
                                    keep_checkpoint_max= 10)

meta_regressor = tf.estimator.Estimator(model_fn=meta_model_fn,
                                        model_dir="./meta_model",
                                        config=run_config,
                                        params={"feature_columns": meta_feature_columns})

meta_regressor.train(input_fn=lambda: meta_input_fn(DATA_dir + "/" + DATA_TRAIN, 250, 500, 32))
evaluate_result = meta_regressor.evaluate(input_fn=lambda: meta_input_fn(DATA_dir + "/" + DATA_TEST, 1))
for key in evaluate_result:
    tf.logging.info("   {} ---> {}".format(key, evaluate_result[key]))



_, tmp_labels = meta_input_fn(DATA_dir + "/" + DATA_TEST, repeat_count=1, batch_size=500, shuffle_count=1)
with tf.Session() as sess:
    labels = sess.run(tmp_labels)

predict_results = meta_regressor.predict(input_fn=lambda:meta_input_fn(DATA_dir + "/" + DATA_TEST, repeat_count=1, batch_size=32, shuffle_count=1))
i = 0
for prediction in predict_results:
    print("Predicted : {0:0.2f}\tExpected : {1:0.2f}\t\tError : {2:0.2f}%".format(prediction["Squeeze"], labels[i], (abs(prediction["Squeeze"]-labels[i]) / labels[i])*100 ))
#    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t".format(prediction["M01"],prediction["M02"],
#                                                                                  prediction["M03"],prediction["M04"],
#                                                                                  prediction["M05"],prediction["M06"],
#                                                                                  prediction["M07"], prediction["M08"],
#                                                                                  prediction["M09"], prediction["M10"],
#                                                                                  prediction["M11"], prediction["M12"],
#                                                                                  prediction["Squeeze"], tmp_labels[i]))
    i=i+1