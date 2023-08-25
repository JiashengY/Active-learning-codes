
�<root"_tf_keras_sequential*�<{"name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_32_input"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "units": 71, "activation": {"class_name": "Activation", "config": {"name": "activation_32", "trainable": true, "dtype": "float32", "activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 43, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_33", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 215, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_34", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}}]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 63]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 63]}, "float32", "dense_32_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 63]}, "float32", "dense_32_input"]}, "keras_version": "2.10.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_32_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "units": 71, "activation": {"class_name": "Activation", "config": {"name": "activation_32", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 1}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 43, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 8}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9}, {"class_name": "Activation", "config": {"name": "activation_33", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 10}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 215, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 13}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14}, {"class_name": "Activation", "config": {"name": "activation_34", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 18}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19}, {"class_name": "Activation", "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 20}]}}, "training_config": {"loss": {"class_name": "MeanSquaredError", "config": {"reduction": "auto", "name": "mean_squared_error"}, "shared_object_id": 23}, "metrics": [[{"class_name": "MeanSquaredError", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 24}, {"class_name": "MeanAbsolutePercentageError", "config": {"name": "MAPE", "dtype": "float32"}, "shared_object_id": 25}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.000648833866348725, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�
root.layer_with_weights-0"_tf_keras_layer*�
{"name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 63]}, "dtype": "float32", "units": 71, "activation": {"class_name": "Activation", "config": {"name": "activation_32", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 1}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 4}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 63}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 63]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 43, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 8}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 71}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 71]}}2
�root.layer-2"_tf_keras_layer*�{"name": "activation_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_33", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43]}}2
�root.layer_with_weights-2"_tf_keras_layer*�{"name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 215, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 13}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 43}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43]}}2
�root.layer-4"_tf_keras_layer*�{"name": "activation_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_34", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 15, "build_input_shape": {"class_name": "TensorShape", "items": [null, 215]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0023787394165992737}, "shared_object_id": 18}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 215}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 215]}}2
�root.layer-6"_tf_keras_layer*�{"name": "activation_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_35", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}2
�$root.layer_with_weights-0.activation"_tf_keras_layer*�{"name": "activation_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation_32", "trainable": true, "dtype": "float32", "activation": "relu"}, "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null, 71]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 30}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanSquaredError", "name": "mean_squared_error", "dtype": "float32", "config": {"name": "mean_squared_error", "dtype": "float32"}, "shared_object_id": 24}2
��root.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "MeanAbsolutePercentageError", "name": "MAPE", "dtype": "float32", "config": {"name": "MAPE", "dtype": "float32"}, "shared_object_id": 25}2