############## FACE RECOGNITION USING PCA ##################
# ----------------------------------------------------------
# PCA - PRINCIPLE COMPONENT ANALYSIS
# aka KLT - Karhunen Loeve Transform and Hotelling Transform
############################################################
print("Importing Libraries...\n")
import os

from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


####### HELPERS ##########################################
def imgraysc(img_vec):
    """Scales Intensities to occupy entire range 0-255."""
    img_min = min(img_vec)
    img_max = max(img_vec)
    a = 255 / (img_max - img_min)
    b = -255 * img_min / (img_max - img_min)
    return a * img_vec + b


def imshow(title, img):
    print(":::::Displaying", title)
    img = np.array(img, np.uint8)
    cv.imshow(title, img)
    cv.waitKey(0)


####### PREPROCESSING #####################################
print("Extracting Images from Dataset...\n")
training_data = "/home/rohitth007/Documents/MOOCs/Computer Vision/Prof.Rajagopalan Labs/Face Recognition using PCA/Training"
training_images = [
    cv.imread(training_data + "/" + folder + "/" + img, cv.IMREAD_GRAYSCALE)
    for folder in os.listdir(training_data)
    for img in os.listdir(training_data + "/" + folder)
]

image_shape = cv.pyrDown(training_images[1]).shape
image_size = cv.pyrDown(training_images[1]).size
data_points = len(training_images)

print("Creating Training ImageData matrix with Image Vectors as columns...")
print("Downsampling images by 2 on each axis...\n")
ImageData = np.zeros((image_size, data_points), np.int16)
for i, img in enumerate(training_images):
    img = cv.pyrDown(img)
    img_vec = img.reshape((image_size,))
    ImageData[:, i] = img_vec


####### DIMENSIONALITY REDUCTION #############################
print("Finding the Mean Image...")
mean_img_vec = ImageData.mean(axis=1)
imshow("Average Human Face\n", mean_img_vec.reshape(image_shape))

print("Centering Images by subtracting Mean Image...")
CenteredImageData = np.zeros((image_size, data_points))
for i in range(data_points):
    CenteredImageData[:, i] = ImageData[:, i] - mean_img_vec

# Display Example
example = ImageData[:, 20].reshape(image_shape)
imshow("Original Face Example", example)

example = np.abs(CenteredImageData[:, 20]).reshape(image_shape)
imshow("Mean Subtracted Face Example", example)

example = imgraysc(CenteredImageData[:, 20]).reshape(image_shape)
imshow("Mean Subtracted Scaled Face Example\n", example)

###### Too Slow #################################################
# print("Finding the Covariance Matrix...")
# CenteredImageData.astype("float32")
# C = CenteredImageData @ CenteredImageData.T / (data_points - 1)
# print("Finding the Eigenvectors of the Covariance Matrix...")
# # C = S Lambda inv(S)
# Lambda, S = np.linalg.eig(C)
# plt.plot(L)
# plt.show()
#################################################################

print("Finding the eigenvectors of Covariance Matrix...")
print("(Same as finding U from the SVD of CenteredImageData/√data_points-1)")
# A = U S Vh
U, Sigma, _ = np.linalg.svd(CenteredImageData / np.sqrt(data_points - 1))
Lambda = np.multiply(Sigma, Sigma)  # S^2 is Eigenvalue matrix of Covariance Matrix

plt.title("Explained Variance")
plt.plot(Lambda)
plt.show()

plt.title("Explained Variance Ratio or Energy Fraction per Principle Component")
plt.plot([var / sum(Lambda) for var in Lambda])
plt.show()

plt.title("Cumulative Variance Ratio or Energy Packing Efficiency")
plt.plot([sum(Lambda[:pc]) / sum(Lambda) for pc in range(len(Lambda))])
plt.show()

pc = int(input("How many Principle Components do you want? > "))
Phi = U[:, :pc].T
EPE = 100 * sum(Lambda[:pc]) / sum(Lambda)
print(f"Energy Packing Efficiency: {EPE}%")


# Display Eigen Faces
for i, row in enumerate(Phi[:10, :]):
    eigen_face = imgraysc(row).reshape(image_shape)
    imshow(f"Eigen Face {i}", eigen_face)

example = ImageData[:, 20].reshape(image_shape)
imshow("Original Image Example", example)

example = ((Phi.T @ (Phi @ CenteredImageData[:, 20])) + mean_img_vec).reshape(
    image_shape
)
imshow("Dimension Reduced Image Example\n", example)


####### TRAINING ###############################################
avg = input("Show Average Faces? [y/n] > ")
og = input("Show Original Faces too? [y/n] > ")

face_classes = []
for person in range(0, 450, 10):
    transformed_images = []
    for centered_img_vec in CenteredImageData.T[person : person + 10, :]:
        if og == "y" and avg == "y":
            imshow("Og Face", (centered_img_vec + mean_img_vec).reshape(image_shape))
        transformed_img = Phi @ centered_img_vec
        transformed_images.append(transformed_img)
    avg_transformed_face = np.mean(transformed_images, axis=0)
    if avg == "y":
        example = imgraysc((Phi.T @ avg_transformed_face) + mean_img_vec)
        imshow("Avg Face", example.reshape(image_shape))
        cv.destroyAllWindows()
    face_classes.append(avg_transformed_face)


###### TESTING ###################################################
testing_data = "/home/rohitth007/Documents/MOOCs/Computer Vision/Prof.Rajagopalan Labs/Face Recognition using PCA/Testing"

testing = "y"
while testing == "y":

    folder_name = input("Which face do you want to recognize? (folder/person name) > ")
    folder_path = testing_data + "/" + folder_name
    try:
        testing_images = [
            cv.pyrDown(cv.imread(folder_path + "/" + img, cv.IMREAD_GRAYSCALE))
            for img in os.listdir(folder_path)
        ]
    except:
        print("Folder does not exist, try again.")
        testing_images = []

    for face in testing_images:
        distances = []
        for avg_transformed_face in face_classes:
            transformed_test_face = Phi @ (face.reshape((image_size,)) - mean_img_vec)
            euclidean_dist = (
                np.linalg.norm(transformed_test_face - avg_transformed_face) ** 2
            )
            distances.append((euclidean_dist, avg_transformed_face))
        _, closest_avg_transformed_face = min(distances)
        imshow("Test Face", face)
        recog_avg_face = (Phi.T @ closest_avg_transformed_face) + mean_img_vec
        imshow(
            "Recognized Average Face",
            imgraysc(recog_avg_face).reshape(image_shape),
        )
    cv.destroyAllWindows()

    testing = input("Do you want to recognize someone else? [y/n] > ")

####### ISSUES: ###############################################################
#       * Not robust to Shading, Illumination, Expressions, Non Frontal Images
#       * PCA works only for Linear Basis and Large Variance having importance
#
####### TODO: #####################################################################################
#       * Display Accuracy, Precision, Recall, F1-Score, Support and Confusion Matrix
#         as shown in https://www.geeksforgeeks.org/ml-face-recognition-using-pca-implementation/
#       * Try out different distance measures.
#       * Try selecting only those eigenfaces which encode useful features.
#       * Try ordering eigenfaces based on Like-Face characteristics.
#       * Try preprocessing images to align faces, histogram equilize lighting/shading, etc.
#       * Try tracking faces using DELAUNAY TRIANGULATION.
#       * Try Kernel PCA
#
###################################################################################################
###################################################################################################
