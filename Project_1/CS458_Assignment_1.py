#
# file  CS490_Assignment_1.py
# brief Purdue University Fall 2022 CS490 robotics Assignment 1 -
#       Gaussian Discriminant Analysis
# date  2022-09-01
#

#you can only import modules listed in the handout
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
from roipoly import RoiPoly

# Helpers
#************************************************************************************************
def draw_polygon_mask(img, title, positive = True):
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    roi = RoiPoly(color='r' if positive else 'b')
    mask = roi.get_mask(img[:, :, 0])  # boolean mask same HxW as image (only use red channel)
    plt.close()
    return mask

def class_stats(features, labels):
    # Returns priors, means, and covariances for classes 0 and 1
    
    priors, means, covs = [], [], []
    N = len(labels)
    for k in [0, 1]:
        features_k = features[labels == k]
        Nk = len(features_k)
        priors.append(Nk / N)
        means.append(features_k.mean(axis=0))
        C = features_k - means[k]
        covs.append((C.T @ C) / Nk)
    
    return priors, means, covs

#hand label region related functions
#**************************************************************************************************
#hand label pos/neg region for training data
#write regions to files, this function should not return anything
def label_training_dataset(training_path, region_path):
    pngs = os.listdir(training_path)

    for fname in pngs:
        img_path = os.path.join(training_path, fname)
        img = imread(img_path)

        # POS polygon (barrel)
        pos_mask = draw_polygon_mask(img, f"TRAIN POS (barrel): {fname}")

        # NEG polygon (non-barrel, similar size, no overlap with barrel)
        neg_mask = draw_polygon_mask(img, 
                                     f"TRAIN NEG (non-barrel, similar area, no overlap): {fname}", positive=False)

        # Save masks
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(region_path, base + "_train_regions.npz")
        np.savez(out_path, pos_mask=pos_mask, neg_mask=neg_mask) # Zip into pos_mask and neg_mask

    print("All training regions saved to:", region_path)

#hand label pos region for testing data
#write regions to files, this function should not return anything
def label_testing_dataset(testing_path, region_path):
    pngs = os.listdir(testing_path)

    for fname in pngs:
        img_path = os.path.join(testing_path, fname)
        img = imread(img_path)

        # POS polygon (barrel)
        pos_mask = draw_polygon_mask(img, f"TEST POS (barrel): {fname}")

        # Save masks
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(region_path, base + "_test_regions.npz")
        np.savez(out_path, pos_mask=pos_mask)

    print("All testing regions saved to:", region_path)
#**************************************************************************************************


#import labeled regions related functions
#**************************************************************************************************
#import pre hand labeled region for trainning data
def import_pre_labeled_training(training_path, region_path):
    # N_train = (# Pixels in all Positive masks) + (# Pixels in all Negative masks)
    # features (N_train, 3): RGB Vector for each pixel in Positive and Negative masks
    #   [[R0, G0, B0]
    #    [R1, G1, B1]
    #    [R2, G2, B2]
    #    [R3, G3, B3]]
    # labels (N_train): The corresponding labels for each pixel in Positive and Negative masks
    #   0 -> Not Barrel Pixel, 1 -> Is Barrel Pixel

    sample_features, sample_labels = [], []
    for p in os.listdir(training_path):
        # Unzip the file containing the positive and negative masks for sample p 
        npz = os.path.join(region_path,  f"{os.path.splitext(p)[0]}_train_regions.npz")
        img = imread(os.path.join(training_path, p))
        img = np.array(img, copy=True, dtype=np.float32)
        if img.max() > 1.0: # Non PNGs come in 0..255 so normalize them
            img /= 255.0
        data = np.load(npz)

        # pos and neg are the masks for sample p
        pos = data["pos_mask"].astype(bool)
        neg = data["neg_mask"].astype(bool)

        # Collect features
        features_pos = img[pos].reshape(-1, 3)
        features_neg = img[neg].reshape(-1, 3)

        # Labels - All pos labels are 1, all neg labels are 0
        labels_pos = np.ones(len(features_pos), dtype=np.int32)
        labels_neg = np.zeros(len(features_neg), dtype=np.int32)
        
        # Combine features and labels in the correct order
        sample_features += [features_pos, features_neg]
        sample_labels += [labels_pos, labels_neg]

    # Combine the samples from all the training examples
    features = np.vstack(sample_features)
    labels = np.concatenate(sample_labels)

    return features, labels

#import per hand labeled region for testing data
def import_pre_labeled_testing(testing_path, region_path):
    # N_test = (#Pixels in all test images)
    # features (N_test, 3): RGB Vector for each pixel in all test images
    #   [[R0, G0, B0]
    #    [R1, G1, B1]
    #    [R2, G2, B2]
    #    [R3, G3, B3]]
    # labels (N_test): The corresponding lables for each pixel in all test images
    #   0 -> Not Barrel Pixel, 1 -> Is Barrel Pixel

    sample_features, sample_labels = [], []
    for p in os.listdir(testing_path):
        npz = os.path.join(region_path,  f"{os.path.splitext(p)[0]}_test_regions.npz")
        img = imread(os.path.join(testing_path, p))
        img = np.array(img, copy=True, dtype=np.float32)
        if img.max() > 1.0:  # JPGs come in 0..255 so normalize them
            img /= 255.0
        data = np.load(npz)

        H, W = img.shape[:2]

        # pos is the positive mask for sample p
        pos = data["pos_mask"].astype(bool)

        # Features: all pixels - Shaped correctly (see comment above)
        features_img = img.reshape(-1, 3)

        # Labels: 1 inside POS polygon, 0 elsewhere
        labels_img = np.zeros(W*H, dtype=np.int32)
        labels_img[pos.reshape(-1)] = 1

        sample_features.append(features_img)
        sample_labels.append(labels_img)

    features = np.vstack(sample_features)
    labels = np.concatenate(sample_labels)

    return features, labels
#**************************************************************************************************


#main GDA training functions
#**************************************************************************************************
def train_GDA_common_variance(features, labels):
    # Perform Linear Discriminate Analysis (LDA) - Pool the variances
    prior, mu, cov = class_stats(features, labels)
    n0 = len(labels[labels==0])
    n1 = len(labels[labels==1])
    pooled = (n0*cov[0] + n1*cov[1]) / (n0+n1)
    # Return pooled for both class 0 and 1 covariances
    return prior, mu, [pooled, pooled]

def train_GDA_variable_variance(features, labels):
    # Perform Quadratic Discriminate Analysis (QDA)
    prior, mu, cov = class_stats(features, labels)
    return prior, mu, cov
#**************************************************************************************************


#GDA testing and accuracy analyis functions
#**************************************************************************************************
#assign labels using trained GDA parameters for testing features
def predict(testing_features, theta, mu, cov):
    # Calculate f_k(X): the decision boundary equation for each class k in {0, 1}
    # Where X is the new data we want to classify
    X = np.asarray(testing_features)

    S0, S1 = cov
    inv0 = np.linalg.inv(S0)
    inv1 = np.linalg.inv(S1)
    logdet0 = np.log(np.linalg.det(S0))
    logdet1 = np.log(np.linalg.det(S1))
    f0 = -0.5*logdet0 - 0.5*np.sum((X - mu[0]) @ inv0 * (X - mu[0]), axis=1) + np.log(theta[0])
    f1 = -0.5*logdet1 - 0.5*np.sum((X - mu[1]) @ inv1 * (X - mu[1]), axis=1) + np.log(theta[1])

    predicted_labels = (f1 > f0).astype(int)
    return predicted_labels

#print precision/call for both classes to console
#
#example console printout:
#GDA with common variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#GDA with variable variance:
#precision of label 0: xx.xx%
#recall of label 0:    xx.xx%
#precision of label 1: xx.xx%
#recall of label 1:    xx.xx%
#
_call_count = 0
def accuracy_analysis(predicted_labels, ground_truth_labels):
    global _call_count
    _call_count += 1

    if _call_count == 1:
        print("GDA with common variance:")
    elif _call_count == 2:
        print("GDA with variable variance:")

    for k in [0, 1]:
        # Compute True Positive, False Positive, False Negative of the predictions vs groud truth
        TP = np.sum((predicted_labels == k) & (ground_truth_labels == k))
        FP = np.sum((predicted_labels == k) & (ground_truth_labels != k))
        FN = np.sum((predicted_labels != k) & (ground_truth_labels == k))

        # Compute the precision and recall
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        print(f"precision of label {k}: {precision*100:.2f}%")
        print(f"recall of label {k}: {recall*100:.2f}%")
#**************************************************************************************************
    
# Segmentation
#**************************************************************************************************
def segment_test_images(testing_path, out_folder, prior, mu, cov):
    os.makedirs(out_folder, exist_ok=True)

    pngs = os.listdir(testing_path)

    for fname in pngs:
        img_path = os.path.join(testing_path, fname)
        img = imread(img_path)

        H, W = img.shape[:2]
        X = img.reshape(-1, 3)

        predictions = predict(X, prior, mu, cov).reshape(H, W)

        # Build red/black segmentation
        seg = np.zeros_like(img, dtype=np.float32)
        seg[predictions == 1] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        out_path = os.path.join(out_folder, fname)
        plt.imsave(out_path, seg)
#**************************************************************************************************


if __name__ == '__main__':
    #Please read this block before coding
    #**********************************************************************************************
    #caution: when you submit this file, make sure the main function is unchanged otherwise your
    #         grade will be affected because the grading script is designed based on the current
    #         main function
    #
    #         Also, do not print unnecessary values other than the accuracy analysis in the console

    #Labeling during runtime can be very time-consuming during the debugging phase. 
    #Also, it is hard to ensure the labelings are consistent during each testing run. 
    #Thus, we do this in separate stages.
    #First, implement all the functions and uncomment the three lines in the data loader block.
    #Then, revert the main function back to what it is used to be and start implementing the rest
    #**********************************************************************************************


    #data loader used to generate your labeling
    #ideally this block should only be called once
    #**********************************************************************************************
    #label_training_dataset('trainset', 'train_region')
    #label_testing_dataset('testset', 'test_region')
    #sys.exit(1)
    #**********************************************************************************************


    #import your generated labels from saved data
    #**********************************************************************************************
    training_features, training_labels = import_pre_labeled_training('trainset', 'train_region')
    testing_features, ground_truth_labels = import_pre_labeled_testing('testset', 'test_region')
    #**********************************************************************************************


    #GDA with common varianve
    #**********************************************************************************************
    prior, mu, cov = train_GDA_common_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************

    # LDA Segmentation - Uncomment to generate LDA segmented images
    #segment_test_images('testset', 'segmentation_GDA_common', prior, mu, cov)
    #segment_test_images('trainset', 'segmentation_GDA_common_training', prior, mu, cov)

    #GDA with variable variance
    #**********************************************************************************************
    prior, mu, cov = train_GDA_variable_variance(training_features, training_labels)

    predicted_labels = predict(testing_features, prior, mu, cov)

    accuracy_analysis(predicted_labels, ground_truth_labels)
    #**********************************************************************************************

    # QDA Segmentation - Uncomment to generate QDA segmented images
    #segment_test_images('testset', 'segmentation_GDA_variable', prior, mu, cov)
    #segment_test_images('trainset', 'segmentation_GDA_variable_training', prior, mu, cov)

