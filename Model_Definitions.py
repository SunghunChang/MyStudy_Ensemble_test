########################################
##                                    ##
##     Module of Model Definitions    ##
##                                    ##
########################################

import tensorflow as tf


# Create features
feature_names = ['vehicle', 'Mp01', 'Mp02', 'Mp03', 'Mp04', 'Mp05', 'Mp06', 'Mp07', 'Mp08', 'Mp09', 'Mp10', 'Mp11', 'Mp12']

vehicle = tf.feature_column.categorical_column_with_vocabulary_list('vehicle', ['Small', 'Mid', 'RV'])
feature_columns = [tf.feature_column.indicator_column(vehicle)]
for k in range(1,13):
    feature_columns.append(tf.feature_column.numeric_column(feature_names[k]))

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1, batch_size = 32):
    # print(" ***** My Input Function Calling ***** ")
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[""], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], field_delim=',')
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    with tf.name_scope('DATA_Feeding'):
        dataset = (tf.data.TextLineDataset(file_path)  # Read text file
            .skip(1)  # Skip header row
            .map(decode_csv)  # Decode each line
            .cache() # Warning: Caches entire dataset, can cause out of memory
            .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
            .repeat(repeat_count)    # Repeats dataset this # times
            .batch(batch_size) # default 32
            .prefetch(1)  # Make sure you always have 1 batch ready to serve
        )
        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

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

# Definition of 1st Model
def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,
    params):    # And instance of tf.estimator.ModeKeys, see below

    if mode == tf.estimator.ModeKeys.PREDICT:
        #tf.logging.info("***************** MODEL FUNCTION MODE : PREDICT, {}".format(mode))
        print("***************** MODEL #{0:s} ESTIMATOR MODE : PREDICT, {1:s}".format(params["model_identifier"], mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        #tf.logging.info("***************** MODEL FUNCTION MODE : EVAL, {}".format(mode))
        print("***************** MODEL #{0:s} ESTIMATOR MODE : EVAL, {1:s}".format(params["model_identifier"], mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        #tf.logging.info("***************** MODEL FUNCTION MODE : TRAIN, {}".format(mode))
        print("***************** MODEL #{0:s} ESTIMATOR MODE : TRAIN, {1:s}".format(params["model_identifier"], mode))

    #with tf.name_scope('Layers'):
    with tf.variable_scope('Model_' + params["model_identifier"] + '_Layers'):
        regularizer = tf.keras.regularizers.l2(l=0.01)
        initializer = tf.keras.initializers.glorot_normal(seed=None)
        input_layer = tf.feature_column.input_layer(features, params["feature_columns"])
        # h1 = tf.layers.dense(input_layer, units=30)
        h1 = tf.layers.Dense(units=30,
                             activation=None, #tf.nn.relu,
                             kernel_initializer = initializer,
                             kernel_regularizer = regularizer,
                             activity_regularizer = None,
                             #bias_regularizer= regularizer,#
                             #bias_constraint=tf.keras.constraints.MaxNorm(0.1),
                             name="First_Hidden_Layer")(input_layer)
        h1 = tf.layers.batch_normalization(h1, training=mode == tf.estimator.ModeKeys.TRAIN)
        h1 = tf.nn.relu(h1)
        h2 = tf.layers.Dense(units=30,
                             activation=None, #tf.nn.relu,
                             kernel_initializer = initializer,
                             kernel_regularizer = regularizer,
                             activity_regularizer = None,
                             #bias_regularizer=regularizer,
                             #bias_constraint=tf.keras.constraints.MaxNorm(0.1),
                             name="Second_Hidden_Layer")(h1)
        h2 = tf.layers.batch_normalization(h2, training=mode == tf.estimator.ModeKeys.TRAIN)
        h2 = tf.nn.relu(h2)
        h3 = tf.layers.Dense(units=30,
                             activation=None, #tf.nn.relu,
                             kernel_initializer = initializer,
                             kernel_regularizer = regularizer,
                             activity_regularizer = None,
                             #bias_regularizer=regularizer,
                             #bias_constraint=tf.keras.constraints.MaxNorm(0.1),
                             name="Third_Hidden_Layer")(h2)
        h3 = tf.layers.batch_normalization(h3, training=mode == tf.estimator.ModeKeys.TRAIN)
        h3 = tf.nn.relu(h3)
        #h3 = tf.layers.Dropout(0.5)(h2)
    #with tf.name_scope('Output_Layer'):
        output_layer = tf.layers.Dense(units = 1, name = "Output_Layer")(h3)

    predictions = {'Squeeze': tf.squeeze(output_layer, 1),  # Squeeze is result value
                   'MP01': features['Mp01'],
                   'MP02': features['Mp02'],
                   'MP03': features['Mp03'],
                   'MP04': features['Mp04'],
                   'MP05': features['Mp05'],
                   'MP06': features['Mp06'],
                   'MP07': features['Mp07'],
                   'MP08': features['Mp08'],
                   'MP09': features['Mp09'],
                   'MP10': features['Mp10'],
                   'MP11': features['Mp11'],
                   'MP12': features['Mp12'],
                   'vehicle_type': features['vehicle']
                   }

    ################################################ 1. Prediction mode ################################################
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # print(features)
        # predictions['Labels'] = tf.convert_to_tensor(labels, dtype=tf.int32)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation and Training mode
    # Calculate the loss
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=Ouput_Layer)

    # Calculate loss using mean squared error
    # ---> It means sum of squared loss [maybe average ......by sunghun chang 2018.04.26]
    average_loss = tf.losses.mean_squared_error(labels, predictions['Squeeze'])

    # Pre-made estimators use the total_loss instead of the average,
    # so report total_loss for compatibility.
    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    with tf.variable_scope('Model_' + params["model_identifier"] + '_Layers', reuse=True):
        weight_value_1 = tf.get_variable("First_Hidden_Layer/kernel")
        weight_value_2 = tf.get_variable("Second_Hidden_Layer/kernel")
        weight_value_3 = tf.get_variable("Third_Hidden_Layer/kernel")
        bias_value_1 = tf.get_variable("First_Hidden_Layer/bias")
        bias_value_2 = tf.get_variable("Second_Hidden_Layer/bias")
        bias_value_3 = tf.get_variable("Third_Hidden_Layer/bias")
        with tf.name_scope('my_summary'):
            tf.summary.histogram('weight_layer_1', weight_value_1)
            tf.summary.histogram('weight_layer_2', weight_value_2)
            tf.summary.histogram('weight_layer_3', weight_value_3)
            tf.summary.histogram('bias_layer_1', bias_value_1)
            tf.summary.histogram('bias_layer_2', bias_value_2)
            tf.summary.histogram('bias_layer_3', bias_value_3)
    with tf.name_scope('my_summary'):
        tf.summary.histogram('batch_size', batch_size)
        tf.summary.histogram('average_loss', average_loss)
        tf.summary.histogram('total_loss', total_loss)
        tf.summary.histogram('First_Layer_Activation', h1)
        tf.summary.histogram('Second_Layer_Activation', h2)
        tf.summary.histogram('Third_Layer_Activation', h3)

    ################################################# 2. Training mode #################################################
    logging_hook = tf.train.LoggingTensorHook({"_average_loss":average_loss,
                                               "_total_loss":total_loss,
                                               "_batch_size":batch_size},
                                               #"output_layer":output_layer},  # It works but too many number.
                                               #"_predictions":predictions['Squeeze']}, # It works but too many number. ex>32
                                               every_n_iter=params["train_logging"])

    # Default optimizer for DNN Regression : Adam with learning rate=0.001
    # Our objective (train_op) is to minimize loss
    # Provide global step counter (used to count gradient updates)
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope('Training_Stage'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                #optimizer = tf.train.AdamOptimizer(0.001, name="My_Optimizer")
                train_op = tf.train.AdamOptimizer(0.001, name="My_Optimizer").minimize(loss=average_loss, global_step=tf.train.get_global_step())
        # Return training operations: loss and train_op
        return tf.estimator.EstimatorSpec(
            mode,
            # loss=total_loss,
            loss=average_loss,
            train_op=train_op,
            training_hooks=[logging_hook],
            predictions=predictions)

    # If mode is not PREDICT nor TRAIN, then we must be in EVAL
    assert mode == tf.estimator.ModeKeys.EVAL, "EVAL is only ModeKey left"

    #with tf.name_scope('Eval_Scope'):
    with tf.variable_scope('Eval_Scope'):
        # Calculate the accuracy between the true labels, and our predictions
        rmse = tf.metrics.root_mean_squared_error(labels, predictions['Squeeze'])
        # accuracy = tf.metrics.accuracy(labels, predictions['Squeeze'])
        accuracy = tf.metrics.recall(labels, predictions['Squeeze'])

        # Add the rmse to the collection of evaluation metrics.
        eval_metrics = {"rmse": rmse, "accuracy": accuracy}

    # Set the TensorBoard scalar my_accuracy to the accuracy
     # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops

    #    tf.summary.scalar('my_accuracy', accuracy)
        tf.summary.scalar('diff', predictions['Squeeze']-labels)

    ################################################ 3. Evaluation mode ################################################
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            #loss=total_loss,
            loss=average_loss,
            eval_metric_ops=eval_metrics,
            predictions = predictions
            )
