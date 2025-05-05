import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import copy

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=True)

# Part 1
def dataset_searcher(number_list,images,labels):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)
    selected_images = []
    selected_labels = []
    for number in number_list:
        indices = np.where(labels == number)[0]
        selected_images.extend(images[indices])
        selected_labels.extend(labels[indices])
    return np.array(selected_images), np.array(selected_labels)  #return images_nparray, labels_nparray

# NEW helper to get one image per digit for display purposes
def dataset_searcher_one_each(number_list, images, labels):
  # Return one sample per class (first match)
    selected_images = []
    selected_labels = []
    for number in number_list:
        index = np.where(labels == number)[0][0]
        selected_images.append(images[index])
        selected_labels.append(labels[index])
    return np.array(selected_images), np.array(selected_labels)

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
    plt.figure(figsize=(10, 2))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher_one_each(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images[:5], class_number_labels[:5])

def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
    return np.mean(results == actual_values)

model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
model1_results = model_1.predict(X_test_reshaped)

# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher_one_each(allnumbers, images, labels)
reshape_allnumbers_images = allnumbers_images.reshape(allnumbers_images.shape[0], -1)
predictions = model_1.predict(reshape_allnumbers_images)
print_numbers(allnumbers_images, allnumbers_labels)

#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0, max_iter=500)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10
poison = rng.normal(scale=noise_scale, size=X_train.shape)
X_train_poison = X_train + poison
X_train_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)

# New: poison test set for visualizing poisoned model predictions correctly
test_poison = rng.normal(scale=noise_scale, size=X_test.shape)
X_test_poison = X_test + test_poison
X_test_poison_reshaped = X_test_poison.reshape(X_test_poison.shape[0], -1)

#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
model_1.fit(X_train_poison_reshaped, y_train)
poisoned_results_1 = model_1.predict(X_test_poison_reshaped)
acc_poisoned_1 = OverallAccuracy(poisoned_results_1, y_test)

model_2.fit(X_train_poison_reshaped, y_train)
poisoned_results_2 = model_2.predict(X_test_poison_reshaped)
acc_poisoned_2 = OverallAccuracy(poisoned_results_2, y_test)

model_3.fit(X_train_poison_reshaped, y_train)
poisoned_results_3 = model_3.predict(X_test_poison_reshaped)
acc_poisoned_3 = OverallAccuracy(poisoned_results_3, y_test)

print("GaussianNB Accuracy (Poisoned):", acc_poisoned_1)
print("KNN Accuracy (Poisoned):", acc_poisoned_2)
print("MLP Accuracy (Poisoned):", acc_poisoned_3)

#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64
X_train_poison_flat = X_train_poison.reshape(X_train_poison.shape[0], -1)
kpca = KernelPCA(n_components=10, kernel='linear', random_state=0)
X_train_denoised = kpca.fit_transform(X_train_poison_flat)
X_test_denoised = kpca.transform(X_test_reshaped)

model_1.fit(X_train_denoised, y_train)
denoised_results_1 = model_1.predict(X_test_denoised)
acc_denoised_1 = OverallAccuracy(denoised_results_1, y_test)

model_2.fit(X_train_denoised, y_train)
denoised_results_2 = model_2.predict(X_test_denoised)
acc_denoised_2 = OverallAccuracy(denoised_results_2, y_test)

model_3.fit(X_train_denoised, y_train)
denoised_results_3 = model_3.predict(X_test_denoised)
acc_denoised_3 = OverallAccuracy(denoised_results_3, y_test)

print("GaussianNB Accuracy (Denoised):", acc_denoised_1)
print("KNN Accuracy (Denoised):", acc_denoised_2)
print("MLP Accuracy (Denoised):", acc_denoised_3)

#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.

def visualize_predictions(model_name, X, y_true, y_pred, n=10):
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(X[i].reshape(8, 8), cmap='gray')
        plt.title(f"T:{y_true[i]} P:{y_pred[i]}")
        plt.axis('off')
    plt.suptitle(f"{model_name} Predictions vs True Labels", fontsize=14)
    plt.show()

visualize_predictions("GaussianNB (Clean)", X_test, y_test, model1_results)
visualize_predictions("KNN (Clean)", X_test, y_test, model2_results)
visualize_predictions("MLP (Clean)", X_test, y_test, model3_results)

# NEW: Use poisoned test images to match poisoned models
visualize_predictions("GaussianNB (Poisoned)", X_test_poison, y_test, poisoned_results_1)
visualize_predictions("KNN (Poisoned)", X_test_poison, y_test, poisoned_results_2)
visualize_predictions("MLP (Poisoned)", X_test_poison, y_test, poisoned_results_3)

# NEW: Use clean test images but denoised model outputs
visualize_predictions("GaussianNB (Denoised)", X_test, y_test, denoised_results_1)
visualize_predictions("KNN (Denoised)", X_test, y_test, denoised_results_2)
visualize_predictions("MLP (Denoised)", X_test, y_test, denoised_results_3)

print("\nMODEL ACCURACY SUMMARY")
print("-" * 50)
print(f"{'Model':<15}{'Clean':>10}{'Poisoned':>12}{'Denoised':>12}")
print("-" * 50)
print(f"{'GaussianNB':<15}{Model1_Overall_Accuracy:10.3f}{acc_poisoned_1:12.3f}{acc_denoised_1:12.3f}")
print(f"{'KNN':<15}{Model2_Overall_Accuracy:10.3f}{acc_poisoned_2:12.3f}{acc_denoised_2:12.3f}")
print(f"{'MLP':<15}{Model3_Overall_Accuracy:10.3f}{acc_poisoned_3:12.3f}{acc_denoised_3:12.3f}")
print("-" * 50)
