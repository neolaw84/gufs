from pathlib import Path

import cv2
import numpy as np

from typing import List

from gufs.watermark_removal_pytorch.api import remove_watermark_np

class Query:
  def __init__(self, image, mask):
    self.image = image
    self.mask = mask

def _to_gray(image:np.ndarray):
  # Check for 3 channels (color)
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

MAX_SCORE = np.finfo(np.float32).max

def mask_by_query(queries:List[Query], input_image, lowe_ratio_thr=0.7):

  # Initialize SIFT detector
  sift = cv2.SIFT_create()

  input_image = _to_gray(input_image)

  mask = np.zeros(input_image.shape, dtype=np.int8)

  # Find keypoints and descriptors for input_image
  kp2, des2 = sift.detectAndCompute(input_image, None)

  for q in queries:
    
    query_image = _to_gray(q.image)

    # Find keypoints and descriptors for query image
    kp1, des1 = sift.detectAndCompute(query_image, None)
    
    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test for good matches
    # there are only two matches but multiple points
    good_matches = [m for m, n in matches if m.distance < lowe_ratio_thr * n.distance]

    # Ensure enough matches for homography calculation
    if len(good_matches) > 0.1 * len(matches):
      src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
      dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

      if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        continue

      # Find homography matrix
      M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.95)
      
      # Get dimensions of query_mask
      # h, w = query_mask.shape[:2]

      if M is not None:

        # Warp query_mask onto the input_image
        transformed_mask = cv2.warpPerspective(q.mask, M, (input_image.shape[1], input_image.shape[0]))

        if transformed_mask.sum() <= 2 * q.mask.sum():
          mask = np.bitwise_or(mask, transformed_mask) 
  mask[mask > 128] = 255
  mask[mask <= 128] = 0
  return mask 

def dilate_mask(input_image, dilation_factor=1, inv:bool=False):
  input_image = _to_gray(input_image)
  if inv: input_image = cv2.bitwise_not(input_image)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  dilate = cv2.dilate(input_image, kernel, iterations=dilation_factor)
  return dilate

def remove_by_mask_using_deep_image_prior(input_f, input_mask_f, output_f, inv_mask:bool=False, max_dim=1920, reg_noise=0.03, input_depth=32, learning_rate=0.01, training_steps=150):
  # the api expects the mask as black to generate; white as leave as-is
  inv = not inv_mask 
  mask_image = cv2.imread(input_mask_f) # mask does not need cvt color
  if inv: mask_image = cv2.bitwise_not(mask_image)
  input_image = cv2.imread(input_f)
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

  output_image = remove_watermark_np(
    input_image, mask_image, max_dim=max_dim, 
    reg_noise=reg_noise, input_depth=input_depth, 
    lr=learning_rate, show_step=0, training_steps=training_steps 
  )
  cv2.imwrite(output_f, output_image)