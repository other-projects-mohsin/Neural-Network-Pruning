
# Neural Network Pruning

Model pruning involves disabling/removing away less critical parameters from a deep learning neural network to make the model smaller and improve its efficiency. Typically, this process focuses on pruning the weights while keeping the biases intact, as removing biases can lead to more substantial negative impacts.

## Importance of Neural Network Pruning

Research has shown that neural networks rely on specific parameters to make their predictions. The idea is that effective model compression should eliminate unnecessary weights, much like the brain reduces the use of certain neural connections to highlight more important pathways. 

## Types of Neural Network Pruning

Neural network pruning can be done in two primary ways: during training or after training. Both methods focus on reducing the network's size and computational demands, but the key difference is the timing of when the pruning takes place. Train-time pruning happens as the model is being trained, while post-training pruning occurs after the training is complete.

|  **Train-Time Pruning**   | **Post-Training Pruning**   |
|------------|------------|
| Pruning is integrated into the training process, making the model sparse as it learns. | Pruning occurs after the model has been fully trained. |
| The model is trained to be sparse from the beginning, rather than pruning after training. | The training process does not consider pruning; it focuses solely on learning. |
| Less important connections or neurons are removed during the learning process. | Pruning techniques are applied once the model has reached convergence and training is complete. |
| Pruning decisions are made alongside weight updates throughout training. | Techniques are used to identify and remove less important connections, neurons, or entire structures from the trained model. |
| Pruning masks are applied during optimization to enforce sparsity. | Pruning is performed as a distinct step following the completion of the training phase. |

## Types of Post-Training Pruning

### Unstructured Pruning

Unstructured pruning is a straightforward, basic method for pruning a neural network, making it easy for beginners to use. It typically involves setting a minimum threshold based on the raw weights or their activations to decide if a particular parameter should be removed. If a parameter doesn't meet this threshold, it's set to zero. Because this method zeros out individual weights within the weight matrices, it doesn’t significantly speed up the model since all calculations are done before pruning. However, unstructured pruning can help make model weights more consistent and reduce the model size without losing any important information. Unstructured pruning is typically safe to use right away without much risk.

### Structured Pruning

Structured pruning takes a more sophisticated and design-focused approach to pruning down neural networks. Instead of just removing individual weights, it eliminates entire groups of them. This reduces the overall computational load during the model's forward pass. Because it aims to prune larger sections, structured pruning needs to be done carefully and precisely, as it affects the connections between different parts of the network. If not done carefully, it can significantly harm the model's performance.

## Neural Network Pruning based on Scopes

### Local Pruning
Local pruning in neural networks means cutting(pruning) away specific neurons, connections, or weights within a single layer. This process targets and removes less critical parts based on factors like low weight, minimal importance in that particular layer, or little impact on the model's overall performance. The pruning is usually done gradually, either one element at a time or in small groups, following specific guidelines.

### Global Pruning
Global pruning, on the other hand, means removing entire neurons, layers, or large sections of the model all at once. This method looks at the importance of neurons or layers across the whole network rather than just within specific layers. It typically uses more advanced methods that consider how different parts of the network interact and depend on each other.

















