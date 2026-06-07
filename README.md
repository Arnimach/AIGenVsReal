# AIGenVsReal
 AI-Generated Image Detector (CIFAKE Pipeline)
 
**Objective:** 
This project focuses on the design and implementation of a custom Convolutional Neural Network (CNN) engineered to detect synthetic artifacts and distinguish AI-generated images from authentic media. Utilizing the CiFAKE Dataset from Kaggle, the custom architecture's performance is benchmarked against fine-tuned, pre-trained computer vision networks, specifically ResNet and DenseNet. To ensure transparency and model auditability, the pipeline integrates Explainable AI (XAI) methodologies to map and analyze the specific visual features driving the network's classification decisions.

**Dataset Description:**
- Dataset: Kaggle CiFAKE Dataset
- Total samples: 20000 images
- Target: FAKE(AI generated) or REAL 
- Task: Classification 

**Approach**

1. Data loading and Preprocessing 
- CiFake dataset has been downloaded from kaggle and data directory has been loaded using tensor flow data pipeline
- Images were verified for curruption and valid extensions
- Data was devided into train, validation and test sets
- Images in the train dataset were augmented using random rotation, random flip and brightness variations to
- improve model robustness and avoid model overfitting.
- Data was cached and pre-fetched mor model optimization.

2. Baseline CNN model
   - Designed and fitted a baseline CNN with three blocks of convolution layers.
   - Global average pooling and 0.5 dropout is applied for model rebustness and reliability.

4. Transfer learning - Model ResNet
   - loaded pre-trained model ResNet trained on vast imagenet dataset
   - Initial layers were prozen and fined tuned the final layer for CiFAKE data.

5. Explainable AI with Grad-CAM
   - In the final step Grad-CAM is used to may and analyze the pixels that influence the model's decision the most.

**Results**
Performance on test set:
Baseline Model: 
- Accuracy: 0.8745
- Precision: 0.9035
- Recall: 0.8402
- mean Runtime/Epoch : ~ 27s

Fine-tuned ResNet Model:
- Accuracy: 0.9449
- Precision: 0.9181
- Recall: 0.9708
- Runtime/epoch : ~ 150 s



**Key Learnings**

* **The Accuracy vs. Computational Cost Trade-off:** While the fine-tuned ResNet model achieved a significantly higher accuracy (94.49% vs. 87.45%) and an exceptional recall rate (97.08%), it came with a massive computational penalty. ResNet required ~150 seconds per epoch compared to the custom baseline’s sleek ~27 seconds—representing a ~5.5x increase in training runtime. 
* **High Recall in AI Forensics:** The custom baseline offers a compelling alternative despite a slightly lower recall.
* **Architectural Bottlenecks dictate Interpretability:** Grad-CAM exposed fundamentally different feature-extraction behaviors between the two models due to their tensor dimensions:
  * **The Custom Baseline** preserved higher spatial dimensions up to its final layer, allowing Grad-CAM to reveal pixel-perfect, high-resolution paths targeting micro-textures and sharp edge artifacts along structural borders.
  * **ResNet** downsampled the input space into a compressed $4\times4$ semantic grid, forcing its Grad-CAM maps to evaluate macro-structural geometry and the global composition of the object.
* **Pipeline Optimization is Mandatory:** Working with tens of thousands of image files proved that programmatic preprocessing (corrupt-file handling, format validation) and TensorFlow pipeline optimizations (`.cache()` and `.prefetch()`) are crucial to preventing CPU bottlenecks and keeping GPU utilization high.
