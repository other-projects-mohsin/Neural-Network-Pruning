import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
dataset_train = tf.data.Dataset.from_tensor_slices((tf.cast(x_train/255, tf.float32),
                                                    tf.cast(y_train, tf.int64))).shuffle(1000).batch(64)
dataset_test = tf.data.Dataset.from_tensor_slices((tf.cast(x_test/255, tf.float32),
                                                   tf.cast(y_test, tf.int64))).batch(64)

# Define the baseline dense model
def create_dense_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1000, activation='relu', use_bias=False),
        tf.keras.layers.Dense(1000, activation='relu', use_bias=False),
        tf.keras.layers.Dense(500, activation='relu', use_bias=False),
        tf.keras.layers.Dense(200, activation='relu', use_bias=False),
        tf.keras.layers.Dense(10, use_bias=False)
    ])
    return model

# Function to test model accuracy and loss
def test(model, dataset):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for x, y in dataset:
        outputs = model(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, outputs)
        epoch_loss_avg.update_state(loss)
        epoch_accuracy.update_state(y, outputs)

    return epoch_loss_avg.result().numpy(), epoch_accuracy.result().numpy()

# Function for weight-based pruning
def prune_weights(model, percentile):
    pruned_model = tf.keras.models.clone_model(model)
    pruned_model.set_weights(model.get_weights())

    for layer in pruned_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            critical_value = np.percentile(np.abs(weights), percentile)
            prune_mask = np.abs(weights) < critical_value
            weights[prune_mask] = 0
            layer.set_weights([weights])

    return pruned_model

# Function for unit (column norm) based pruning
def prune_units(model, percentile):
    pruned_model = tf.keras.models.Sequential()
    pruned_model.add(tf.keras.layers.Flatten(input_shape=(28, 28))))
    num_layers = len(model.layers)
    prev_kept_columns = None

    for i_layer, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            if i_layer < num_layers - 1:
                column_norms = np.linalg.norm(weights, ord=2, axis=0)
                critical_value = np.percentile(column_norms, percentile)
                keep_mask = column_norms >= critical_value
                weights = weights[:, keep_mask]

                if prev_kept_columns is not None:
                    weights = weights[prev_kept_columns, :]

                prev_kept_columns = np.argwhere(keep_mask).reshape(-1)

            new_layer = tf.keras.layers.Dense(weights.shape[1], activation='relu', use_bias=False)
            new_layer.set_weights([weights])
            pruned_model.add(new_layer)

    return pruned_model

# Function to dynamically decide pruning strategy based on layer characteristics
def dynamic_prune(model, percentile):
    pruned_weight_model = tf.keras.models.clone_model(model)
    pruned_unit_model = tf.keras.models.Sequential()
    pruned_unit_model.add(tf.keras.layers.Flatten(input_shape=(28, 28))))

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()[0]
            column_norms = np.linalg.norm(weights, ord=2, axis=0)
            critical_value_weight = np.percentile(np.abs(weights), percentile)
            critical_value_norm = np.percentile(column_norms, percentile)

            if np.mean(column_norms) > 0.5:  # Heuristic: if mean column norm is high, use unit pruning
                keep_mask = column_norms >= critical_value_norm
                weights = weights[:, keep_mask]
                pruned_unit_layer = tf.keras.layers.Dense(weights.shape[1], activation='relu', use_bias=False)
                pruned_unit_layer.set_weights([weights])
                pruned_unit_model.add(pruned_unit_layer)

            else:  # Otherwise, use weight pruning
                prune_mask = np.abs(weights) < critical_value_weight
                weights[prune_mask] = 0
                layer.set_weights([weights])
                pruned_weight_model.add(layer)

    return pruned_weight_model, pruned_unit_model

# Percentiles for pruning
percentiles = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]

# Initialize lists for storing results
weight_pruned_losses = []
weight_pruned_accuracies = []
unit_pruned_losses = []
unit_pruned_accuracies = []
weight_pruned_timings = []
unit_pruned_timings = []

# Iterate over different percentiles and perform pruning and evaluation
for percentile in percentiles:
    print(f"Pruning at {percentile}% sparsity...")
    
    # Dynamic pruning approach
    start_time = time.time()
    pruned_weight_model, pruned_unit_model = dynamic_prune(model, percentile)
    
    # Evaluate weight pruned model
    weight_pruned_loss, weight_pruned_accuracy = test(pruned_weight_model, dataset_test)
    weight_pruned_timings.append(time.time() - start_time)
    weight_pruned_losses.append(weight_pruned_loss)
    weight_pruned_accuracies.append(weight_pruned_accuracy)
    
    # Evaluate unit pruned model
    start_time = time.time()
    unit_pruned_loss, unit_pruned_accuracy = test(pruned_unit_model, dataset_test)
    unit_pruned_timings.append(time.time() - start_time)
    unit_pruned_losses.append(unit_pruned_loss)
    unit_pruned_accuracies.append(unit_pruned_accuracy)

    print(f"Weight Pruned Test Accuracy: {weight_pruned_accuracy:.4f}, Test Loss: {weight_pruned_loss:.4f}, Time: {weight_pruned_timings[-1]:.2f}s")
    print(f"Unit Pruned Test Accuracy: {unit_pruned_accuracy:.4f}, Test Loss: {unit_pruned_loss:.4f}, Time: {unit_pruned_timings[-1]:.2f}s")

# Plotting results
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(percentiles, weight_pruned_accuracies, marker='o', label='Weight Pruned')
plt.plot(percentiles, unit_pruned_accuracies, marker='s', label='Unit Pruned')
plt.xlabel('Sparsity (%)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Sparsity')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(percentiles, weight_pruned_losses, marker='o', label='Weight Pruned')
plt.plot(percentiles, unit_pruned_losses, marker='s', label='Unit Pruned')
plt.xlabel('Sparsity (%)')
plt.ylabel('Test Loss')
plt.title('Test Loss vs Sparsity')
plt.legend()

plt.tight_layout()
plt.show()

# Display timings
plt.figure(figsize=(10, 5))
plt.plot(percentiles, weight_pruned_timings, marker='o', label='Weight Pruned')
plt.plot(percentiles, unit_pruned_timings, marker='s', label='Unit Pruned')
plt.xlabel('Sparsity (%)')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Sparsity')
plt.legend()
plt.show()
