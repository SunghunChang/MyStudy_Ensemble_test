########################################
##                                    ##
##     Module of Model Definitions    ##
##                                    ##
########################################

# Use Adam-Optimizer
# - Learning Rate : 0.001
# - Training and Evaluation Loss -> Average Loss를 사용함
#                                 [mean squared error between labels and predictions]
# - Total Loss = Average Error * Batch Size
#
# 3-Hidden Layers with 30 unit nodes
#
# Activation is RELU function

import tensorflow as tf
import numpy as np

# Create features
feature_names = ['vehicle','A','A_Ang_X','A_Ang_Z','S1','S1_Ang_Y','S1_Ang_Z','S2','S2_Ang_Y','S2_Ang_Z','B','B_Ang_X','B_Ang_Y','R','R_Ang_X','R_Ang_Z','BU', #17개
                'Mp01', 'Mp02', 'Mp03', 'Mp04', 'Mp05', 'Mp06', 'Mp07', 'Mp08', 'Mp09', 'Mp10', 'Mp11', 'Mp12'] # 12개

vehicle = tf.feature_column.categorical_column_with_vocabulary_list('vehicle', ['Small', 'Mid', 'RV'])
feature_columns = [tf.feature_column.indicator_column(vehicle)]

# 원래 feature columns
#for k in range(1,29):
#    feature_columns.append(tf.feature_column.numeric_column(feature_names[k]))

# Bucket 및 Crossed Column 테스트 용
for k in range(1,28):
    feature_columns.append(tf.feature_column.numeric_column(feature_names[k]))
temp_numeric_column = tf.feature_column.numeric_column(feature_names[28])
feature_columns.append(tf.feature_column.bucketized_column(source_column=temp_numeric_column, boundaries=list(range(5,21,1))))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x,axis=axis, keep_dims=True)
    devs_squared = tf.square(x-m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1, batch_size = 32):
    # print(" ***** My Input Function Calling ***** ")
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[""], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                                           [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]], field_delim=',') #feature 포함
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

        '''
        # Display batch_features / batch_labels
        with tf.Session() as sess:
            tmp_features = sess.run(batch_features)
            tmp_labels = sess.run(batch_labels)
            #for key, expec in zip(tmp_features, tmp_labels):
            #    tmp_string = ''
            #    for i in range(0,len(tmp_features[key])):
            #        tmp_string = tmp_string + str("{0}\t".format(tmp_features[key][i]))
            #    print("{} .....>>> {}".format(key, tmp_string))
            #print(tmp_labels)
            for i in range(0,batch_size):
                tmp_string = ''
                for key in sorted(tmp_features):
                    tmp_string = tmp_string + str("{0}\t".format(tmp_features[key][i]))
                print(str(tmp_labels[i]) + '\t' + tmp_string)
        '''

    return batch_features, batch_labels #batch_features, batch_labels

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

    #init = tf.global_variables_initializer()
    #sess = tf.InteractiveSession()
    #sess.run(init)

    if mode == tf.estimator.ModeKeys.PREDICT:
        #tf.logging.info("***************** MODEL FUNCTION MODE : PREDICT, {}".format(mode))
        print("***************** MODEL #{0:s} Estimator MODE : PREDICT, {1:s}".format(params["model_identifier"], mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        #tf.logging.info("***************** MODEL FUNCTION MODE : EVAL, {}".format(mode))
        print("***************** MODEL #{0:s} Estimator MODE : EVAL, {1:s}".format(params["model_identifier"], mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        #tf.logging.info("***************** MODEL FUNCTION MODE : TRAIN, {}".format(mode))
        print("***************** MODEL #{0:s} Estimator MODE : TRAIN, {1:s}".format(params["model_identifier"], mode))

    #with tf.name_scope('Layers'):
    #with tf.variable_scope('Model_' + params["model_identifier"] + '_Layers'):
    with tf.variable_scope('Model_Layer_Informations'):
        regularizer = tf.keras.regularizers.l2(l=0.01)
        initializer = tf.keras.initializers.glorot_normal(seed=None)
        #regularizer = None
        #initializer = None
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
                   'A': features['A'], 'A_Ang_X': features['A_Ang_X'], 'A_Ang_Z': features['A_Ang_Z'],
                   'S1': features['S1'], 'S1_Ang_Y': features['S1_Ang_Y'], 'S1_Ang_Z': features['S1_Ang_Z'],
                   'S2': features['S2'], 'S2_Ang_Y': features['S2_Ang_Y'], 'S2_Ang_Z': features['S2_Ang_Z'],
                   'B': features['B'], 'B_Ang_X': features['B_Ang_X'], 'B_Ang_Y': features['B_Ang_Y'],
                   'R': features['R'], 'R_Ang_X': features['R_Ang_X'], 'R_Ang_Z': features['R_Ang_Z'],
                   'BU': features['BU'],
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
                   'vehicle_type': features['vehicle'],
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

    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, predictions['Squeeze'])))

    #with tf.variable_scope('Model_' + params["model_identifier"] + '_Layers', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('Model_Layer_Informations', reuse=tf.AUTO_REUSE):
        weight_value_1 = tf.get_variable("First_Hidden_Layer/kernel")  # (15,30)
        weight_value_2 = tf.get_variable("Second_Hidden_Layer/kernel") # 30,30
        weight_value_3 = tf.get_variable("Third_Hidden_Layer/kernel") # 30,30
        bias_value_1 = tf.get_variable("First_Hidden_Layer/bias")  # (30,)
        bias_value_2 = tf.get_variable("Second_Hidden_Layer/bias") # (30,)
        bias_value_3 = tf.get_variable("Third_Hidden_Layer/bias")  # (30,)
        with tf.name_scope('my_summary_layers'):
            tf.summary.histogram('weight_layer_1', weight_value_1)
            tf.summary.histogram('weight_layer_2', weight_value_2)
            tf.summary.histogram('weight_layer_3', weight_value_3)
            tf.summary.histogram('bias_layer_1', bias_value_1)
            tf.summary.histogram('bias_layer_2', bias_value_2)
            tf.summary.histogram('bias_layer_3', bias_value_3)
            tf.summary.histogram('First_Layer_Activation', h1)
            tf.summary.histogram('Second_Layer_Activation', h2)
            tf.summary.histogram('Third_Layer_Activation', h3)
        # 아래처럼 해도 Tree처럼 되지는 않음. Filter Tag를 편하게 쓸 수 있어서 좋긴 함
        with tf.name_scope('layers_1'):
            with tf.name_scope('std'):
                tf.summary.scalar('Weight_Std', reduce_std(tf.reshape(weight_value_1, [1, -1])))
                tf.summary.scalar('Bias_Std', reduce_std(tf.reshape(bias_value_1, [1, -1])))
            with tf.name_scope('mean'):
                tf.summary.scalar('Weight_Mean', tf.reduce_mean(tf.reshape(weight_value_1, [1, -1])))
                tf.summary.scalar('Bias_Mean', tf.reduce_mean(tf.reshape(bias_value_1,[1,-1])))
        with tf.name_scope('layers_2'):
            with tf.name_scope('std'):
                tf.summary.scalar('Weight_Std', reduce_std(tf.reshape(weight_value_2,[1,-1])))
                tf.summary.scalar('Bias_Std', reduce_std(tf.reshape(bias_value_2, [1, -1])))
            with tf.name_scope('mean'):
                tf.summary.scalar('Weight_Mean', tf.reduce_mean(tf.reshape(weight_value_2, [1, -1])))
                tf.summary.scalar('Bias_Mean', tf.reduce_mean(tf.reshape(bias_value_2,[1,-1])))
        with tf.name_scope('layers_3'):
            with tf.name_scope('std'):
                tf.summary.scalar('Weight_Std', reduce_std(tf.reshape(weight_value_3,[1,-1])))
                tf.summary.scalar('Bias_Std', reduce_std(tf.reshape(bias_value_3, [1, -1])))
            with tf.name_scope('mean'):
                tf.summary.scalar('Weight_Mean', tf.reduce_mean(tf.reshape(weight_value_3, [1, -1])))
                tf.summary.scalar('Bias_Mean', tf.reduce_mean(tf.reshape(bias_value_3,[1,-1])))
    with tf.name_scope('my_summary_training'):
        tf.summary.scalar('batch_size', batch_size)
        tf.summary.scalar('Total_Loss', total_loss)
        tf.summary.scalar('Mean_Squared_Error_aver.loss', average_loss)
        tf.summary.scalar('Root_Mean_Squared_Error', rmse)
        #tf.summary.scalar('Mean_Absolute_Error', mae)
        tf.summary.histogram('Mean_Squared_Error_aver.loss', average_loss)
        tf.summary.histogram('Total_Loss_aver.loss*batch_size', total_loss)


    ################################################# 2. Training mode #################################################
    logging_hook = tf.train.LoggingTensorHook({"_average_loss[mse]":average_loss,
                                               "_total_loss":total_loss,
                                               "_batch_size":batch_size,
                                               "_weight_std_1":reduce_std(tf.reshape(weight_value_1,[1,-1])),
                                               "_weight_std_2":reduce_std(tf.reshape(weight_value_2,[1,-1])),
                                               "_weight_std_3":reduce_std(tf.reshape(weight_value_3,[1,-1]))},
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


        # 내부 변수 출력 #################################################################################################
        # Trainable 변수 이름과 Shape 출력  - 되긴함 : 별도 로그파일로 출력
        # trainable_variables / all_variables
        Variable_File = open(params["log_dir"] + "/03_Training_Graph_Variables.log", 'w')
        variables_names = [v.name for v in tf.all_variables()]
        values = [v for v in tf.all_variables()] #variables_names.eval()
        Variable_File.writelines("Write ALL Variables in Train Graph [tf.all_variables()]\n")
        for k, v in zip(variables_names, values):
            #print("Name : ", k, " / shape : ", v.shape) #이건 잘됨
            Variable_File.writelines("* Name : " + k + " ------> " + str(v.shape) + '\n')
        #    print("value : ", tf.get_variable('Model_1_Layers/First_Hidden_Layer/kernel')) 이건 안됨
        Variable_File.close()
        # 내부 변수 출력 #################################################################################################

        # Operation 출력 ################################################################################################
        Op_File = open(params["log_dir"] + "/04_Training_Graph_Operations.log", 'w')
        Op_File.writelines("Write ALL Operations in Train Graph\n")
        for op in tf.get_default_graph().get_operations(): Op_File.writelines(str(op.name) + '\n')
        Op_File.close()
        # Operation 출력 ################################################################################################

        return tf.estimator.EstimatorSpec(
            mode,
            # loss=total_loss,
            loss=average_loss,
            train_op=train_op,
            training_hooks=[logging_hook],
            predictions=predictions)

    # If mode is not PREDICT nor TRAIN, then we must be in EVAL
    assert mode == tf.estimator.ModeKeys.EVAL, "EVAL is only ModeKey left"


    #with tf.variable_scope('Eval_Scope'):
    with tf.name_scope('Eval_Scope'):
        # Calculate the accuracy between the true labels, and our predictions
        # 아래 Label등은 숫자 하나가 아니라 배치가 다 들어온 것인가보다...
        rmse = tf.metrics.root_mean_squared_error(labels, predictions['Squeeze'])
        mae = tf.metrics.mean_absolute_error(labels, predictions['Squeeze'])
        #mre = tf.metrics.mean_relative_error(labels, predictions['Squeeze'])
        mse = tf.metrics.mean_squared_error(labels, predictions['Squeeze'])

        #accuracy = tf.metrics.recall(labels, predictions['Squeeze'])

    # Add the rmse to the collection of evaluation metrics.
    eval_metrics = {"rmse": rmse, "mae": mae, "mse": mse}

    # Set the TensorBoard scalar my_accuracy to the accuracy
     # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops

    # 얘네는 왜 안나오나....
    with tf.name_scope('Errors'):
        tf.summary.scalar('Root_Mean_Squared_Error', rmse)
        tf.summary.scalar('Mean_Squared_Error', mse)
        tf.summary.scalar('Mean_Absolute_Error', mae) # predictions['Squeeze']-labels)

    ################################################ 3. Evaluation mode ################################################
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:

        ''' 내부 변수 출력 예제
        # Write ALL Operations in Graph
        # 그래프 오퍼레이션 이름 모두 출력
        # 이게 Training에서 찍을때랑 Eval에서 찍을때 보이는 그래프들이 다름. 파일로 별도출력하여 아래와 비교 필요 - 로그파일로 빼자
        print("Write ALL Operations in This Graph")
        for op in tf.get_default_graph().get_operations(): print(str(op.name))

        # Write ALL Operations in Graph. I think it is same as above
        # 위에꺼랑 결과 같은것 같은데... 둘다 따로 파일로 찍어보고 비교 필요함
        print("Write ALL Operations in Tensor Name")
        for n in tf.get_default_graph().as_graph_def().node: print(str(n.name))
        '''

        # Operation 출력 ################################################################################################
        Op_File = open(params["log_dir"] + "/05_Evaluation_Graph_Operations.log", 'w')
        Op_File.writelines("Write ALL Operations in Eval Graph\n")
        for op in tf.get_default_graph().get_operations(): Op_File.writelines(str(op.name) + '\n')
        Op_File.close()
        # Operation 출력 ###############################################################################################

        return tf.estimator.EstimatorSpec(
            mode,
            #loss=total_loss,
            loss=average_loss,
            eval_metric_ops=eval_metrics,
            predictions = predictions
            )
