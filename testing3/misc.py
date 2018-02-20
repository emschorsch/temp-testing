"""Simple TensorFlow Multitask model.
More specifically, a simple shared Bottom model with shared hidden layers at
the bottom.
"""
import collections
import six
import tensorflow as tf
def _stack_layers_and_drop_out(input_layer, hidden_layer_sizes, activation_fn,
                               variables_collections, is_training):
  """Constructs stack hidden layers and adds dropout layer.
  Args:
    input_layer: A tensor representing input layer.
    hidden_layer_sizes: A list of layer sizes. This is used as stack_args in
      tf.contrib.layers.stack.
    activation_fn: Activation function applied to each layer.
    variables_collections: A list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    is_training: A bool `Tensor` indicating whether or not the model
      is in training mode. Used in creating dropout layer.
  Returns:
    A tensor representing output layer.
  Raises:
    ValueError: if hidden_layer_sizes is not a list.
  """
  if not isinstance(hidden_layer_sizes, list):
    raise ValueError("hidden_layer_sizes should be a list.")
  outputs = input_layer
  for layer_id, num_hidden_units in enumerate(hidden_layer_sizes):
    with tf.variable_scope(
        "hiddenlayer_%d" % layer_id, values=[outputs]) as current_scope:
      outputs = tf.contrib.layers.fully_connected(
          outputs,
          num_hidden_units,
          activation_fn=activation_fn,
          variables_collections=variables_collections,
          scope=current_scope)
      outputs = tf.contrib.layers.dropout(
          inputs=outputs, keep_prob=1.0, is_training=is_training)
    tf.summary.scalar("hiddenlayer_%d_fraction_of_zero_values" % layer_id,
                      tf.nn.zero_fraction(outputs))
    tf.summary.histogram("hiddenlayer_%d_activation" % layer_id, outputs)
  return outputs
def build_bottom_model_block(features,
                             feature_columns,
                             is_training,
                             hidden_units=None,
                             activation_fn=None,
                             embedding_collection_names=None,
                             variables_collections=None):
  """Creates input layers and hidden layers from features.
  Args:
    features: A `dict` of tensors returned by input function.
    feature_columns: A `list` of feature columns used to build the bottom block.
    is_training: A boolean indicating if the model is in training phase.
    hidden_units: A `list` of hidden units per layer.
    activation_fn: Activation function applied to each layer.
    embedding_collection_names: A `list` of collection names for
      embedding weights.
    variables_collections: A `list` of collection names for all model variables.
  Returns:
    A `Tensor` of top hidden layer outputs.
  """
  with tf.variable_scope(
      "input_layer", values=list(six.itervalues(features))) as scope:
    input_layer = tf.contrib.layers.input_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=feature_columns,
        weight_collections=embedding_collection_names,
        scope=scope)
  if not hidden_units:
    hidden_layer = input_layer
  else:
    hidden_layer = _stack_layers_and_drop_out(
        input_layer=input_layer,
        hidden_layer_sizes=hidden_units,
        activation_fn=activation_fn,
        variables_collections=variables_collections,
        is_training=is_training)
  return hidden_layer
def build_top_model_block(bottom_hidden_layer,
                          output_dimension,
                          is_training,
                          hidden_units=None,
                          activation_fn=None,
                          variables_collections=None):
  """Creates hidden layers and predictions/logits.
  Args:
    bottom_hidden_layer: A `Tensor` of hidden layer outputs from the bottom
      part of the model.
    output_dimension: The dimensionality of prediction.
    is_training: A boolean indicating if the model is in training phase.
    hidden_units: A `list` of hidden units per layer.
    activation_fn: Activation function applied to each layer.
    variables_collections: A `list` of collection names for all model variables.
  Returns:
    A `Tensor` of predictions/logits.
  """
  if not hidden_units:
    hidden_layer = bottom_hidden_layer
  else:
    hidden_layer = _stack_layers_and_drop_out(
        input_layer=bottom_hidden_layer,
        hidden_layer_sizes=hidden_units,
        activation_fn=activation_fn,
        variables_collections=variables_collections,
        is_training=is_training)
  # Output predictions/logits. No activation_fn will be used for the top
  # layer of logits.
  return tf.contrib.layers.fully_connected(
      inputs=hidden_layer,
      num_outputs=output_dimension,
      activation_fn=None,
      variables_collections=variables_collections)
def build_shared_bottom_model(
    features, feature_columns, targets_dict, shared_hidden_units,
    hidden_units_dict, is_training, activation_fn, scale_gradient_targets_dict):
  """Builds SharedBottom model block for MultiTask models.
  Args:
    features: A `dict` of tensors returned by input function.
    feature_columns: A `list` of feature columns used to build shared bottom
      model block.
    targets_dict: A `dict` of integer. Each Key is a string representing a
      target name, which should be the same as the head_name if `Head` is used.
      Value is the dimensionality of predicted targets, which should be the
      same as the head's logits_dimension if `Head` is used.
    shared_hidden_units: A `list` of hidden units per layer for the shared
      bottom.
    hidden_units_dict: A 'dict'. Each key is a target name same as the key
      of targets_dict, and the value is a list of hidden units for that
      particular target. All such hidden units are built on top of the shared
      hidden units. They are not shared across targets.
    is_training: A boolean indicating if the model is in training phase.
    activation_fn: Activation function applied to each layer.
    scale_gradient_targets_dict: A `dict` of float used as gradient_multiplier
      scales. Each key is a target name (e.g., head.head_name). The value is
      the scale of gradient_multiplier for a specific target on shared
      variables. If the target_name here has appeared in
      stop_gradient_targets, then only stop_gradient will be applied,
      gradient_multiplier will not be executed.
  Returns:
    A `dict` of predictions/logits.
  Raises:
    ValueError: if the keys of targets_dict are not the same as the keys of
      hidden_units_dict.
  """
  if set(targets_dict.keys()) != set(hidden_units_dict.keys()):
    raise ValueError(
        "Mismatched keys between targets_dict and hidden_units_dict")
  parent_scope = "shared_bottom"
  hidden_layer_partitioner = tf.min_max_variable_partitioner(max_partitions=0)
  with tf.variable_scope(
      parent_scope,
      values=tuple(six.itervalues(features)),
      partitioner=hidden_layer_partitioner):
    shared_hidden_layer = build_bottom_model_block(
        features=features,
        feature_columns=feature_columns,
        is_training=is_training,
        hidden_units=shared_hidden_units,
        activation_fn=activation_fn,
        embedding_collection_names=[None, parent_scope],
        variables_collections=["shared_hidden_layers"])
    prediction_dict = {}
    # Iterate through targets in a sorted way so that multiple workers and ps
    # will build graph and allocate resources in the same way.
    for name in sorted(targets_dict):
      with tf.variable_scope(
          name, values=[shared_hidden_layer]) as multitower_scope:
        if name in scale_gradient_targets_dict:
          tf.contrib.layers.scale_gradient(shared_hidden_layer,
                                           scale_gradient_targets_dict[name])
        prediction_dict[name] = build_top_model_block(
            bottom_hidden_layer=shared_hidden_layer,
            output_dimension=targets_dict[name],
            is_training=is_training,
            hidden_units=hidden_units_dict[name],
            activation_fn=activation_fn,
            variables_collections=[multitower_scope.name])
  return prediction_dict
def merge_multiple_input_fn(input_fn_dict, weight_column_name):
  """Merges multiple input streams by concatenating them together.
  Merges multiple input functions into one input function. Each input function
  should return the same set of features and targets for a batch of examples.
  The merged input function will concat all such batches into a single large
  batch of examples, keeping the feature and targets list the same.
  Args:
    input_fn_dict: A dict of input functions keyed by their names.
      Here, we assume all input functions have same type of features and
      targets, i.e., the keys of features and targets (if targets is a dict)
      should be the same for all input functions. If there are some unique
      features or targets in some input functions, these features and
      targets after being merged will not have the correct batch size and
      should not be used.
    weight_column_name: The column name of example weights in features.
      The features returned by each input function will contain a weight_column
      representing the example weights. In the merged input function, for each
      original input function, we will construct a new weight column, with name:
      weight_column_name + '_' + input function name. In this new weight
      column, same example weights will be used for the examples coming from
      that input function, and for examples coming from other input functions,
      the weight will be 0.
  Example usage:
    Given input_fn_1 and input_fn_2,
    input_fn_1: features = {"f1": [[1.0], [2.0]], "weight": [1, 1]},
                targets = {"t1": [0, 1], "t2": [0, 0]}
    input_fn_2: features = {"f1": [[-1.0]], "weight": [0.5]},
                targets = {"t1": [0], "t2": [1]}
    merged_input_fn =
      merge_multiple_input_fn({"fn1": input_fn_1, "fn2": input_fn_2}, "weight")
    The resulting merged_input_fn will be:
      features = {
        "f1", [[1.0], [2.0], [-1.0]], "weight": [1, 1, 0.5],
        "weight_fn1": [1, 1, 0], "weight_fn2": [0, 0, 0.5]}
      targets = {"t1": [0, 1, 0], "t2": [0, 0, 1]}
  Returns:
    A merged input_fn.
  Raises:
    ValueError: if weight_column_name not in any of the input functions.
  """
  def merged_input_fn():
    """Merges input function from multiple input functions."""
    # All features and weights from multiple input functions.
    tensors_to_concat_dict = collections.defaultdict(list)
    # All targets from multiple input functions.
    targets_to_concat_dict = collections.defaultdict(list)
    for current_input_fn_name in sorted(six.iterkeys(input_fn_dict)):
      # Get features from each data_source.
      features, targets = input_fn_dict[current_input_fn_name]()
      if weight_column_name not in features:
        raise ValueError(
            "Weight column name: %s should be in input_fn name: %s." %
            (weight_column_name, current_input_fn_name))
      for feature_key in sorted(six.iterkeys(features)):
        tensors_to_concat_dict[feature_key].append(features[feature_key])
      # Construct weight tensor for multiple input functions. The new weight
      # tensor for the current input function will be: for the examples coming
      # from the current input function, the original weight tensor; for the
      # examples coming from other input functions, zero tensor.
      for input_fn_name in sorted(six.iterkeys(input_fn_dict)):
        if input_fn_name == current_input_fn_name:
          weight_tensor = features[weight_column_name]
        else:
          weight_tensor = tf.zeros_like(features[weight_column_name])
        new_weight_column = weight_column_name + "_" + input_fn_name
        tensors_to_concat_dict[new_weight_column].append(weight_tensor)
      if isinstance(targets, dict):
        for target_key in sorted(six.iterkeys(targets)):
          targets_to_concat_dict[target_key].append(targets[target_key])
      else:
        # Use "" as a placeholder when targets is not a `dict`.
        targets_to_concat_dict[""].append(targets)
    # Concat features and weights from all input functions into one tensor.
    features = {}
    for feature_key in sorted(six.iterkeys(tensors_to_concat_dict)):
      if isinstance(tensors_to_concat_dict[feature_key][0], tf.SparseTensor):
        features[feature_key] = tf.sparse_concat(
            0, tensors_to_concat_dict[feature_key], expand_nonconcat_dim=True)
      else:
        features[feature_key] = tf.concat(tensors_to_concat_dict[feature_key],
                                          0)
    # Generate targets for multitask input_fn.
    target_keys = sorted(six.iterkeys(targets_to_concat_dict))
    if target_keys == [""]:
      targets = tf.concat(targets_to_concat_dict[""], 0)
    else:
      targets = {}
      for target_key in target_keys:
        targets[target_key] = tf.concat(targets_to_concat_dict[target_key], 0)
    return features, targets
  return merged_input_fn
