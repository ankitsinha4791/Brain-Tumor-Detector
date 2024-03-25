
# Brain Tumor Detection using Deep Learning

Brain Tumor Detection using Deep Learning is an innovative and potentially life-saving project that leverages the power of artificial intelligence to assist in the early diagnosis and detection of brain tumors. Brain tumors are a critical health concern, and their early detection is crucial for effective treatment and improved patient outcomes.

This project utilizes advanced deep learning algorithms and neural networks to analyze medical imaging data, such as MRI (Magnetic Resonance Imaging) and CT (Computed Tomography) scans of the brain. The core objective is to develop a robust and accurate model capable of identifying the presence, location, and type of brain tumors within these scans.

See it Live at:- https://huggingface.co/spaces/Ankit4791/Brain-Tumor-Detection

## Tech stack used
- Deep Learning Framework: TensorFlow

- Deep Learning Library: Keras (integrated with TensorFlow)

- Programming Language: Python

- Data Preprocessing: OpenCV, NumPy

- Image Processing: PIL (Python Imaging Library)

- Data Splitting and Evaluation: Scikit-Learn

- Normalization: Scikit-Learn

- Model Architecture: Convolutional Neural Networks (CNNs)

- Model Layers: Conv2D, MaxPooling2D, Dense, Dropout

- Activation Function: Rectified Linear Unit (ReLU)

- Optimizer: Adam

- Loss Function: Categorical Cross-Entropy, Binary Cross-Entropy

- Performance Metrics: Accuracy, Precision, Recall, F1-Score

- Visualization: Matplotlib (for data and results visualization)

- IDE (Integrated Development Environment): VS Code

# Data Set





 - Used for Training- [Dataset Br35H](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)
 - Used for testing- [Dataset MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
 
 

| Folder| Description               |
| :-------- |:------------------------- |
| Yes | Contains 1500 Brain MRI Images that are tumorous|
| No  | Contains 1500 Brain MRI Images that are non-tumorous |
| Pred | Contains 60 Brain MRI Images that are random



# Data Preprocessing

- Data Cleaning:
Checked for any corrupt or unusable images and removed them from the dataset.

Verified the correctness of labels and resolved any labeling errors or inconsistencies.

- Data Splitting:

Split the dataset into three subsets: training, validation, and test sets. Common splits include 70-80% for training, 10-15% for validation, and 10-15% for testing.

- Data Rescaling and Normalization:

Rescaled image pixel values to a standardized range (e.g., [0, 1]) to ensure consistent data input.

Resized all images to a uniform size, typically required by the model architecture (e.g., 64x64 pixels or 224x224 pixels).

Converted images to the appropriate format (e.g., RGB) if necessary.

Created numerical arrays from the images for model input, typically using libraries like NumPy.





# Model architecture

Input Image Dimensions: Each input image (x) has a shape of (240, 240, 3), representing a 240x240 pixel image with three color channels (RGB).

- Layer 1: Zero Padding Layer with Pooling

A Zero Padding layer was applied with a pool size of (2, 2) to add padding around the input image.
- Layer 2: Convolutional Layer

A convolutional layer was employed with 32 filters, each having a filter size of (7, 7) and a stride equal to 1.
This layer is responsible for capturing spatial features from the input image.
- Layer 3: Batch Normalization

A batch normalization layer was used to normalize pixel values, which helps in speeding up computation and enhancing training stability.
- Layer 4: ReLU Activation

Rectified Linear Unit (ReLU) activation was applied to introduce non-linearity into the model.
ReLU is commonly used to enable the network to learn complex patterns in the data.
Layer 5: Max Pooling Layer

- The Max Pooling layer with a filter size (f) of 4 and a stride (s) of 4 was applied.
This layer reduces the spatial dimensions of the feature maps, effectively downsampling the information.
- Layer 6: Max Pooling Layer (Repeating)

A second Max Pooling layer with the same parameters (f=4, s=4) was applied.
Repetition of this layer further reduces the spatial dimensions and focuses on capturing important features.
- Layer 7: Flatten Layer

A flatten layer was added to reshape the 3-dimensional matrix into a one-dimensional vector.
This step is necessary to connect the convolutional layers to the fully connected layers.
- Layer 8: Dense (Output) Layer

The final layer is a Dense fully connected layer with one neuron, suitable for binary classification tasks and two neuron for categorical classification.
A sigmoid activation function was applied to produce binary output predictions (0 or 1), indicating the presence or absence of a brain tumor.

# Training the model

The model was trained for 25 epochs and these are the loss & accuracy plots:

- Binary model accuracy:- 
![binary_model_accuracy](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/122fd5b7-24a6-44ee-b353-02c02fb49c47)

- Binary model loss:-
![binary_model_loss](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/40de8e52-0148-49b4-98f7-a111ef1df1eb)

- Categorical model accuracy:-
 ![Categorical_model_accuracy](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/fe73487d-0d7e-4de6-bc1c-0fd6787474aa)

- Categorical model loss:-
![categorical_model_loss](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/e07bcca9-0f47-4f18-8a92-e93fb91fad04)



# Results
Now, the best model (the one with the best validation accuracy) detects brain tumor with:

94.6% accuracy on the test set.

0.93 f1 score on the test set.

These resutls are very good considering that the data is balanced.
| Parameter| Validation set | Test set |
| :-------- |:------------------------- |:--------------|
| Accuracy | 98.9%|    94.6%          |
| F1 score | 0.98|   0.93         |



# Flask App

I have proficiently made a Flask-based application, empowered with a custom machine learning model, designed for the purpose of image upload and subsequent tumor prediction.

Here are some images...

![flaskAppTest](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/ff739b7c-6eff-411c-b67a-57556fe62ecc)

![flaskAppTest1](https://github.com/ankitsinha4791/Brain-Tumor-Detection/assets/97782545/d563d3d7-3c75-4df7-9c60-d118627f408c)

# Help and Support
Thank you for taking the time to explore our project! We appreciate your interest and support. If you have any questions, feedback, or encounter issues, please don't hesitate to reach out. We value your input and contributions.

Feel free to visit my website