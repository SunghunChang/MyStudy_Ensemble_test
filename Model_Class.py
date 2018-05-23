import tensorflow as tf

class Model:
	def my_native_model(self, features, labels, mode, params):  # And instance of tf.estimator.ModeKeys, see below

		if mode == tf.estimator.ModeKeys.PREDICT:
			# tf.logging.info("***************** MODEL FUNCTION MODE : PREDICT, {}".format(mode))
			print("***************** MODEL FUNCTION MODE : PREDICT, {}".format(mode))
		elif mode == tf.estimator.ModeKeys.EVAL:
			# tf.logging.info("***************** MODEL FUNCTION MODE : EVAL, {}".format(mode))
			print("***************** MODEL FUNCTION MODE : EVAL, {}".format(mode))
		elif mode == tf.estimator.ModeKeys.TRAIN:
			# tf.logging.info("***************** MODEL FUNCTION MODE : TRAIN, {}".format(mode))
			print("***************** MODEL FUNCTION MODE : TRAIN, {}".format(mode))

		# with tf.name_scope('Layers'):
		with tf.variable_scope('Model_1_Layers'):
			regularizer = tf.keras.regularizers.l2(l=0.01)
			initializer = tf.keras.initializers.glorot_normal(seed=None)
			input_layer = tf.feature_column.input_layer(features, params["feature_columns"])
			h1 = tf.layers.Dense(units=30,
			                     activation=tf.nn.relu,
			                     kernel_initializer=initializer,
			                     kernel_regularizer=regularizer,
			                     activity_regularizer=None,
			                     name="First_Hidden_Layer")(input_layer)
			h2 = tf.layers.Dense(units=30,
			                     activation=tf.nn.relu,
			                     kernel_initializer=initializer,
			                     kernel_regularizer=regularizer,
			                     activity_regularizer=None,
			                     name="Second_Hidden_Layer")(h1)
			h3 = tf.layers.Dense(units=30,
			                     activation=tf.nn.relu,
			                     kernel_initializer=initializer,
			                     kernel_regularizer=regularizer,
			                     activity_regularizer=None,
			                     name="Third_Hidden_Layer")(h2)
			# h3 = tf.layers.Dropout(0.5)(h2)
			# with tf.name_scope('Output_Layer'):
			output_layer = tf.layers.Dense(units=1, name="Output_Layer")(h3)

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

		with tf.variable_scope('Model_1_Layers', reuse=True):
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

		# Default optimizer for DNN Regression : Adam with learning rate=0.001
		# Our objective (train_op) is to minimize loss
		# Provide global step counter (used to count gradient updates)
		if mode == tf.estimator.ModeKeys.TRAIN:
			with tf.name_scope('Training_Stage'):
				optimizer = tf.train.AdamOptimizer(0.001, name="My_Optimizer")
				train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step())
			# Return training operations: loss and train_op
			return tf.estimator.EstimatorSpec(
				mode,
				# loss=total_loss,
				loss=average_loss,
				train_op=train_op,
				predictions=predictions)

		# If mode is not PREDICT nor TRAIN, then we must be in EVAL
		assert mode == tf.estimator.ModeKeys.EVAL, "EVAL is only ModeKey left"

		# with tf.name_scope('Eval_Scope'):
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
			tf.summary.scalar('diff', predictions['Squeeze'] - labels)

		################################################ 3. Evaluation mode ################################################
		# Return our loss (which is used to evaluate our model)
		# Set the TensorBoard scalar my_accuracy to the accuracy
		# Obs: This function only sets value during mode == ModeKeys.EVAL
		# To set values during training, see tf.summary.scalar
		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(
				mode,
				# loss=total_loss,
				loss=average_loss,
				eval_metric_ops=eval_metrics,
				predictions=predictions
			)