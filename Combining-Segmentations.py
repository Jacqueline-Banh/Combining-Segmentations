# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:05:36 2023

@author: Jacqueline Banh
"""

import SimpleITK as sitk
import numpy as np
import logging
import itertools
import math

import time
start = time.time()
class NewSegmentation:
    def __init__(self, imgPaths):
        # Read the input segmentations using SimpleITK:
        # Import 3D image
        self.image = None
        if imgPaths[0][-1] == '/' or imgPaths[0][-1] == 'm':
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(imgPaths[0])
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(imgPaths[0])
        
        # The output segmentation will be the sum of all the segmentations.
        # We create a blank output image with the desired size, spacing, and origin.
        self.output = sitk.Image(image.GetSize(), sitk.sitkUInt16)
        self.output.SetSpacing(image.GetSpacing())
        self.output.SetOrigin(image.GetOrigin())
        self.output.SetDirection(image.GetDirection())
        self.imgPaths = imgPaths
        self.images = [sitk.ReadImage(path) for path in self.imgPaths[1:]]
        
    def clean_segmentation(self):
        # Remove the small structures that are not part of the structure 
        cleaned_img = sitk.OpeningByReconstruction(self.output, [5, 5, 5])
        cleaned_img = sitk.BinaryClosingByReconstruction(cleaned_img, [5, 5, 5])
        # Now there are also some over-segmentation. So let's connect the over-segmented structures and
        # mark each potential object with a unique label this way:
        self.output = sitk.ConnectedComponent(cleaned_img!= 0)

    def MAJORITY_VOTING(self, threshold):
        # for each image in the collection of segmentations:
        for image in self.images:
            resampled_image = sitk.Resample(image, self.output, sitk.Transform(), sitk.sitkLinear, 0.0, image.GetPixelIDValue())
            self.output += resampled_image
        # for each pixel value in the image 'self.output', 
        # if pixel value > vote threshold, pixel value = 1
        # but if pixel value < vote threshold, pixel value = 0
        self.output = sitk.Cast(self.output > threshold*(len(self.imgPaths)-1), sitk.sitkUInt16)
        
    def SHAPE_BASED_AVERAGING(self, threshold):
        # for each image in the collection of segmentations:
        for image in self.images:
            resampled_image = sitk.Resample(image, self.output, sitk.Transform(), sitk.sitkLinear, 0.0, image.GetPixelIDValue())
            # Distance map filter
            dmf = sitk.DanielssonDistanceMapImageFilter()
            # Compute outside distance map
            odm = dmf.Execute(resampled_image)          
            # Compute inside distance map
            invertLabelsF = sitk.InvertIntensityImageFilter()
            invertLabelsF.SetMaximum(1)
            invertLabels = invertLabelsF.Execute(resampled_image)
            idm = dmf.Execute(invertLabels) 
            fdm = odm - idm # final distance map is the odm minus the idm
            # for each pixel value in the fdm,
            # if the pixel value is < distance threshold, pixel value = 1
            # if the pixel value is > distance threshold, pixel value = 0
            self.output += sitk.Cast(fdm < threshold*(len(self.imgPaths)-1), sitk.sitkUInt16)  
        
    def STAPLE(self, threshold):
        resampledImages = []
        for i,image in enumerate(self.images):
            resampled_image = sitk.Resample(image, self.output, sitk.Transform(), sitk.sitkLinear, 0.0, image.GetPixelIDValue())
            resampledImages.append(resampled_image)
        # the STAPLE filter needs the resampled images and a foreground value (1.0) to produce a single output 
        # with a range of floating point values from 0 - 1. To get a binary output the values must be 
        # greater than the threshold
        self.output = (sitk.STAPLE(resampledImages, 1)) > threshold
        
    def binaryMav(self, candidates, weights=None):
        '''
        binaryMav performs majority vote fusion on an arbitary number of input segmentations with
        only two classes each (1 and 0).
        
        Args:
            candidates (list): the candidate segmentations as binary numpy arrays of same shape
            weights (list, optional): associated weights for each segmentation in candidates. Defaults to None.
        
        Return
            array: a numpy array with the majority vote result
        '''       
        num = len(candidates)
        if weights == None:
            weights = itertools.repeat(1,num)
        # manage empty calls
        if num == 0:
            print('ERROR! No segmentations to fuse.')
        elif num == 1: 
            return candidates[0]
        # load first segmentation and use it to create initial numpy arrays
        temp = candidates[0]
        result = np.zeros(temp.shape)
        #loop through all available segmentations and tally votes for each class
        label = np.zeros(temp.shape)
        for c, w in zip(candidates, weights):
            if c.max() != 1 or c.min() != 0:
                logging.warning('The passed segmentation contains labels other than 1 and 0.')
            print('weight is: ' + str(w))
            label[c == 1] += 1.0*w
        num = sum(weights)
        result[label >= (num/2.0)] = 1
        return result
    
    def _score(self, seg, gt, method='dice'):
        ''' Calculates a similarity score based on the
        method specified in the parameters
        Input: Numpy arrays to be compared, need to have the 
        same dimensions (shape)
        Default scoring method: DICE coefficient
        method may be:  'dice'
                        'auc'
                        'bdice'
        returns: a score [0,1], 1 for identical inputs
        '''
        try: 
            # True Positive (TP): we predict a label of 1 (positive) and the true label is 1.
            TP = np.sum(np.logical_and(seg == 1, gt == 1))
            # True Negative (TN): we predict a label of 0 (negative) and the true label is 0.
            TN = np.sum(np.logical_and(seg == 0, gt == 0))
            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(seg == 1, gt == 0))
            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(seg == 0, gt == 1))
            FPR = FP/(FP+TN)
            FNR = FN/(FN+TP)
            TPR = TP/(TP+FN)
            TNR = TN/(TN+FP)
        except ValueError:
            print('Value error encountered!')
            return 0
        # faster dice? Oh yeah!
        if method is 'dice':
            # default dice score
            score = 2*TP/(2*TP+FP+FN)
        elif method is 'auc':
            # AUC scoring
            score = 1 - (FPR+FNR)/2
        elif method is 'bdice':
            # biased dice towards false negatives
            score = 2*TP/(2*TP+FN)
        elif method is 'spec':
            #specificity
            score = TN/(TN+FP)
        elif method is 'sens':
            # sensitivity
            score = TP/(TP+FN)
        elif method is 'toterr':
            score = (FN+FP)/(155*240*240)
        elif method is 'ppv':
            prev = np.sum(gt)/(155*240*240)
            temp = TPR*prev
            score = (temp)/(temp + (1-TNR)*(1-prev))
        else:
            score = 0
        if np.isnan(score) or math.isnan(score):
            score = 0
        return score
    
    def SIMPLE(self, t=0.05, stop=25, inc=0.07, method='dice', iterations=25, labels=None):
        '''
        simple implementation using DICE scoring
        Iteratively estimates the accuracy of the segmentations and dynamically assigns weights 
        for the next iteration. Continues for each label until convergence is reached. 

        Args:
            candidates (list).
            t (float, optional). Defaults to 0.05.
            stop (int, optional). Defaults to 25.
            inc (float, optional). Defaults to 0.07.
            method (str, optional). Defaults to 'dice'.
            iterations (int, optional). Defaults to 25.
            labels (list, optional). Defaults to None.
        
        Raises:
            IOError: If no segmentations to be fused are passed
        
        Returns:
            array: a numpy array with the SIMPLE fusion result
        '''
        # manage empty calls
        candidates = []
        for i,image in enumerate(self.images):
            resampled_image = sitk.Resample(image, self.output, sitk.Transform(), sitk.sitkLinear, 0.0, image.GetPixelIDValue())
            candidates.append(sitk.GetArrayFromImage(resampled_image))
        num = len(candidates)
        # handle unpassed weights
        weights = itertools.repeat(1,num)
        backup_weights = weights # ugly save to reset weights after each round
        # get unique labels for multi-class fusion
        if labels == None:
            labels = np.unique(candidates[0])
            for c in candidates:
                labels = np.append(labels, np.unique(c))
                labels = np.unique(labels).astype(int)
            result = np.zeros(candidates[0].shape)
        # remove background label
        if 0 in labels:
            labels = np.delete(labels, 0)
        # loop over each label
        for l in sorted(labels):
            # load first segmentation and use it to create initial numpy arrays IFF it contains labels
            bin_candidates = [(c == l).astype(int) for c in candidates]
            # baseline estimate
            estimate = self.binaryMav(bin_candidates)
            #initial convergence baseline
            conv = np.sum(estimate)
            # reset tau before each iteration
            tau = t
            for i in range(iterations):
                t_weights = [] # temporary weights
                for c in bin_candidates:
                    # score all canidate segmentations
                    t_weights.append((self._score(c, estimate, method)+1)**2) #SQUARED DICE!
                weights = t_weights
                # save maximum score in weights
                max_phi = max(weights)
                # remove dropout estimates
                bin_candidates = [c for c, w in zip(bin_candidates, weights) if (w > t*max_phi)]
                # calculate new estimate
                estimate = self.binaryMav(bin_candidates, weights)
                # increment tau 
                tau = tau+inc
                # check if it converges
                if np.abs(conv-np.sum(estimate)) < stop:
                    break
                conv = np.sum(estimate)
            # assign correct label to result
            result[estimate == 1] = l
            # reset weights
            weights = backup_weights
            
        result = sitk.GetImageFromArray(result)
        result.CopyInformation(self.output)    
        self.output = sitk.Cast(result, sitk.sitkUInt8)
    
# Specify the paths of the input image files (Replace with your file paths)
# 1st argument is path to 3D image, subsequent arguments are paths of segmentations to combine
image_paths = [r"C:/Users/Jacqueline Banh/Desktop/MSegmentations-20230913T132022Z-001/MSegmentations/LightGreen29/Volume3D.mha",
               r"C:/Users/Jacqueline Banh/Desktop/MSegmentations-20230913T132022Z-001/MSegmentations/LightGreen29/Structure.mha",
                r"C:/Users/Jacqueline Banh/Desktop/HSegmentations-20230913T132010Z-001/HSegmentations/LightGreen29.nrrd",
                r"C:/Users/Jacqueline Banh/Desktop/KSegmentations-20230913T132015Z-001/KSegmentations/LightGreen29/LightGreen29.nrrd"]

# image_paths = ["/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/", 
#                 "/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/Segmentation_Liver_1.mha",
#                 "/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/Segmentation_Liver_2.mha",
#                 "/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/Segmentation_Liver_3.mha",
#                 "/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/Segmentation_Liver_4.mha",
#                 "/Users/meganreyes/Desktop/Research/data/manifest-1671406182956/COVID-19-NY-SBU/A750765/11-28-1900-NA-MRI ABD WWO IV CONT-05790/11.000000-T2 HASTE COR THIN-35150/Segmentation_Liver_5.mha"]


# Initialize a blank segmentation image
output = NewSegmentation(image_paths)

""" Use one approach to combine segmentations """
# TO DO: decide on threshold value:
thresholdMV = 0.34
output.MAJORITY_VOTING(thresholdMV)

#thresholdSBA = 0.33
#output.SHAPE_BASED_AVERAGING(thresholdSBA)
#thresholdSTAPLE = 0.01
#output.STAPLE(thresholdSTAPLE)

#output.SIMPLE()
       
# Save the combined image to a file:
output.clean_segmentation()
output_path = r"C:\Users\Jacqueline Banh\Desktop\SBAdarkblue14.mha"  # Replace with your desired output file pathoutput_path = r"/Users/meganreyes/Desktop/STAPLE_test.mha"  # Replace with your desired output file path
end = time.time()
sitk.WriteImage(output.output, output_path)
print ("Excecution time is " + str(end - start) + " seconds")


# # read each segmentation image path to be resampled 
# images = [sitk.ReadImage(file_name) for file_name in image_paths[1:]]
# interpolator = sitk.sitkLinear

# # each image in images is replaced with the resampled image
# for i,image in enumerate(images):
#     resampled_image = sitk.Resample(image, output.output, sitk.Transform(), interpolator, 0.0, image.GetPixelIDValue())
#     images[i] = resampled_image
    
    
# # Empty numpy arrays to hold results of each measure
# overlap_results = np.zeros((len(images), 4))
# # Overlap measures filter
# overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

# # compare each segmentation with the reference segementation (majority vote output) to
# # get each overlap measure 
# for i,seg in enumerate(images):
#     overlap_measures_filter.Execute(output.output, seg)
#     overlap_results[i, 0] = overlap_measures_filter.GetJaccardCoefficient()
#     overlap_results[i, 1] = overlap_measures_filter.GetDiceCoefficient()
#     overlap_results[i, 2] = overlap_measures_filter.GetFalseNegativeError()
#     overlap_results[i, 3] = overlap_measures_filter.GetFalsePositiveError()

# print(overlap_results) # show each measurement for each segmentation
