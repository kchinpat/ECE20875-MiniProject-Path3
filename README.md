# ECE20875-MiniProject-Path3
**Objective**
The aim of this project is to evaluate and compare the robustness of three classification algorithms - Gaussian Naive Bayes (GaussianNB), K-Nearest Neighbors (KNN), and Multi-Layer Perception (MLP), we will be testing these three algorithm with three different scenario of the digit image under clean, poisoned, and denoised conditions.
Code explanation (Step by step implementation)
Part 1-2: Data Extraction and Visualization, we implemented two helper functions:
dataset_searcher(): Fetches all image samples for selected digit classes.
dataset_searcher_one_each(): Fetches one representative image per digit for display
Image from digit classes [2,0,8,7,5] and [0-9] were visualized for sanity check.

Figure 1: Display of the class [2,0,8,7,5]

Figure 2: Display of the class [0,1,2…,9]
Part 3-6: Model Training and Evaluation on Clean Data Three classifiers were trained on clean, reshaped image data:
GaussianNB
KNN (k = 10)
MLP (max iteration = 500)

Figure 3: Example of clean data
Part 8-11: Impact of Poisoned Data Gaussian noise (scale = 10.0) was added to the training and test data to simulate poisoned conditions.  All three models were retrained using this noisy data.

Figure 4: Example of Poisoned (Noisy) data

Part 12-13: Denoising with Kernel PCA To mitigate poisoning effects, we applied Kernel PCA to denoise the training and test datasets.

Figure 5: Example of Denoised Data
Part 14-15: Visualization and Summary, Prediction results of each model under all three scenarios were visualized side-by-side.  This provide insights into how models interpret images under clean vs noisy and vs denoised conditions.

Figure 6: Example of Formatted Output
When run the program the expected output will be consisting of 11 Figures including
2 Samples, 3 of clean data, 3 of poisoned data, and 3 of denoised data.
Then the program will also analyze the accuracy of each of the models respectively with the quality of image.

