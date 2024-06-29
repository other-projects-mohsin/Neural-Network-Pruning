# Project Details

 - Prototyped a neural network pruning tool using TensorFlow, aimed at reducing model size and computational complexity by selectively removing less critical weights.
 - Implemented both weight pruning (individual weight removal) and unit pruning (removal of entire neurons or filters) strategies to optimize model efficiency.
 - Conducted  experimentation to analyze the trade-offs between sparsity levels and model performance metrics such as accuracy and inference speed.
 - Achieved  reductions in model size while maintaining competitive performance, demonstrating proficiency in neural network optimization and TensorFlow frameworks.
 - Proposed a simple dyanmic approach automatically determines when, where and how to prune a neural network based on mean column norm calculation.

# Analysis

 - Weight pruning generally maintains higher test accuracy than unit pruning as sparsity increases.
 - At lower levels of sparsity, both methods perform similarly, but unit pruning's performance drops significantly at higher sparsity levels.
 - Weight pruning remains relatively robust up to 80% sparsity, after which its accuracy also declines.

# Others Paper Reviewed

I studied multiple other papers on NN pruning utilizing different approaches and assessed and tried to understand their methodologies and respective codebases from the literature survey conducted by [He and Xiao](https://github.com/he-y/Awesome-Pruning?tab=readme-ov-file), specifically the following:

- [Model Preserving Compression for Neural Networks](https://openreview.net/pdf?id=gt-l9Hu2ndd)
    - This paper introduces a  technique for compressing neural networks, which aims to retain the model's functional properties throughout the compression process.
    - Highlighted the importance of maintaining model interpretability, which is often lost in conventional compression techniques, by prioritizing the preservation of the underlying model structure and logic.
    - The authors demonstrate the effectiveness of their method across the ATOM3D, CIFAR-10, and ImageNet datasets using a variety of models. Their method does not require labeled data, automatically determines the sizes for each layer, and often needs little to no fine-tuning.
    - The proposed approach may t may increase computational costs during the compression process due to the emphasis on maintaining model functionality.

- [PRUNING DEEP NEURAL NETWORKS FROM A SPARSITY PERSPECTIVE](https://openreview.net/pdf?id=i-DleYh34BM) (re-read :/)
    
    - The research introduces the PQ Index (PQI) as a novel metric for assessing the compressibility of deep neural networks during pruning iterations. This index serves to gauge when a model is appropriately regularized, potentially underfitting, or on the verge of collapsing.

    - Additionally, the study presents a new algorithm called Sparsity-informed Adaptive Pruning (SAP). This algorithm utilizes the PQ Index (index for helping in understanding when a model is effectively regularized, when it starts to underfit, and when it collapses) to direct the pruning process, ensuring that the model avoids both under-pruning and over-pruning pitfalls. Compared to conventional iterative pruning methods such as lottery ticket-based strategies, SAP demonstrates enhanced efficiency and resilience.

    - Empirical experiments conducted in the research provide evidence supporting the effectiveness of the PQ Index and SAP algorithm. The findings indicate that by selecting appropriate hyper-parameters, the adaptive pruning approach enhances compression efficiency and maintains model performance stability, underscoring the importance of integrating sparsity information in deep neural network pruning.

- [Recent Advances on Neural Network Pruning at Initialization](https://www.ijcai.org/proceedings/2022/0786.pdf)

    - Paper focused on a new approach called Pruning at Initialization (PaI), which targeted pruning a randomly initialized network rather than a pretrained model. 

    - Reviewed existing/traditional pruning methods v/s emergin PaI methods

    - The goal of PaI is to make the network leaner by removing unnecessary connections or neurons early in the starting of the trainig process, aiming to reach a point where the pruned network can achieve similar accuracy to the original dense network.

    - ToDo: Study how PaI effectively intergrates with emerging Pruning at Initialization


# Acknowledgement

The source code is adapted form [Nitarshan's NN Pruning Experiment](https://github.com/nitarshan/neural-network-pruning-and-sparsification/tree/master)

