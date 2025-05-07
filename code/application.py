import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import helper
import cv2
import os
from scipy.ndimage import distance_transform_edt

os.environ["QT_QPA_PLATFORM"] = "xcb"

#Load images
def createPanorama(input_img_list):
    images = []
    dims = []
    for index, path in enumerate(input_img_list):
        print(f"Loading image: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"Error: Could not load image at {path}")
        images.append(img)
        dims.append(images[index].shape)
        print(f"Image {index} dimensions: {dims[-1]}")
    return images, dims

#Resize Images if necessary
def reSizeImages(images, dims):
    ht, _, _ = min(dims, key=lambda val: val[0])
    _, wt, _ = min(dims, key=lambda val: val[1])
    _, _, ch = min(dims, key=lambda val: val[2])
    print(f"Resizing all images to: Height={ht}, Width={wt}")
    images = [cv2.resize(x, (ht, wt)) for x in images]
    blank = np.zeros((wt, ht, ch), np.uint8)
    return images, blank

#Generate Panorama
def getMatches(image1, image2, features, ransac_iterations=10, dist_threshold=0.7):
    print("Detecting and computing features...")
    points1, des1 = features.detectAndCompute(image1, None)
    points2, des2 = features.detectAndCompute(image2, None)
    print(f"Detected {len(points1)} keypoints in Image 1 and {len(points2)} in Image 2")

    # Draw keypoints on the images and save them
    image1_with_keypoints = cv2.drawKeypoints(image1, points1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_with_keypoints = cv2.drawKeypoints(image2, points2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save the images with keypoints
    cv2.imwrite("image1_keypoints.jpg", image1_with_keypoints)
    cv2.imwrite("image2_keypoints.jpg", image2_with_keypoints)

    matcher = cv2.FlannBasedMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    print(f"Total matches found: {len(matches)}")

    imp = []
    for i, (one, two) in enumerate(matches):
        if one.distance < dist_threshold * two.distance:
            imp.append((one.trainIdx, one.queryIdx))
    print(f"Filtered important matches: {len(imp)}")

    good_matches = [cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0) for j, i in imp]

    # Save image with matching keypoints
    matched_image = cv2.drawMatches(image1, points1, image2, points2, good_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("matched_keypoints.jpg", matched_image)

    matched1 = np.float32([points1[i].pt for (__, i) in imp])
    matched2 = np.float32([points2[i].pt for (i, __) in imp])

    print("Computing homography...")
    H, err, Ps = helper.ransac_calibrate(matched2, matched1, matched1.shape[0], "", ransac_iterations)
    print("Homography matrix (H):")
    print(H)
    print(f"Homography matrix computed with error: {min(err)}")

###max pixel
    hmax = max(image1.shape[0], image1.shape[0])
    wmax = max(image1.shape[1], image1.shape[1])
    out = cv2.warpPerspective(image2, H, (4*wmax,4*hmax))
    print("Perspective warp completed.")

    final_out = np.zeros(out.shape, np.uint8)
    h, w = final_out.shape[:2]
    print(f"Final output image size: {h}x{w}")

    for ch in tqdm(range(3), desc="Merging images"):
        for x in range(h):
            for y in range(w):
                final_out[x, y, ch] = max(out[x, y, ch], image1[x, y, ch] 
                if x < image1.shape[0] and y < image1.shape[1] else 0)
    print("Image merging completed.")

    final_out = getNonZeroImage(final_out)
    return final_out, out, des1[0]

###weighted distance map
    # #Create binary masks for each image
    # h1, w1 = image1.shape[:2]
    # h2, w2 = image2.shape[:2]
    # hmax = max(h1, h2)
    # wmax = w1 + w2  # Combine width of both images

    # # Warp image2 to the size of the combined image
    # out = cv2.warpPerspective(image2, H, (wmax, hmax))
    # print("Perspective warp completed.")

    # # Save the warped image for debugging
    # #cv2.imwrite("warped_image2.jpg", out)

    # # Create binary masks for each image
    # mask1 = np.zeros((hmax, wmax), dtype=np.uint8)
    # mask1[:h1, :w1] = 1

    # mask2 = cv2.warpPerspective(np.ones_like(image2[:, :, 0], dtype=np.uint8), H, (wmax, hmax))

    # # Compute distance transforms
    # dist1 = distance_transform_edt(mask1)
    # dist2 = distance_transform_edt(mask2)

    # # Normalize distance maps
    # weights1 = dist1 / (dist1 + dist2 + 1e-6)
    # weights2 = dist2 / (dist1 + dist2 + 1e-6)

    # # Create a combined image to hold both images
    # combined_image = np.zeros((hmax, wmax, 3), dtype=np.uint8)
    # combined_image[:h1, :w1] = image1

    # # Blend the images using the computed weights
    # blended_image = np.zeros_like(combined_image, dtype=np.float32)
    # for ch in tqdm(range(3), desc="Blending channels"):
    #     blended_image[:, :, ch] = (
    #         weights1 * combined_image[:, :, ch] +
    #         weights2 * out[:, :, ch]
    #     )

    # blended_image = cv2.convertScaleAbs(blended_image)

    # # Save the blended image
    # # cv2.imwrite("blended_image.jpg", blended_image)

    # print("Weighted blending completed.")

    # # Return the blended image and any other necessary outputs
    # return blended_image, out, des1[0]

def getNonZeroImage(image):
    out_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = cv2.findNonZero(out_gray)
    if pixels is None:
        print("Error: No non-zero pixels found!")
        return image
    pixels = pixels.reshape([pixels.shape[0], 2])
    extreme_min_x, extreme_max_x = min(pixels[:, 0]), max(pixels[:, 0])
    extreme_min_y, extreme_max_y = min(pixels[:, 1]), max(pixels[:, 1])
    print(f"Cropping image to non-zero region: X({extreme_min_x}-{extreme_max_x}), Y({extreme_min_y}-{extreme_max_y})")
    return image[extreme_min_y:extreme_max_y, extreme_min_x:extreme_max_x, :]


def showAndWrite(output, temp, path="out.png"):
    print(f"Saving stitched image to {path}")
    output = getNonZeroImage(output)
    cv2.imwrite(path, output)
    #cv2.imwrite("temp.png", temp)
    print("Images saved successfully.")

if __name__ == "__main__":
    # Provide a list of multiple images
    image_list = [
        r"/home/mukta-hacker/Desktop/Mukta/sem6/CV_Project/Panorama-Effect-Image-Stitching/test_images/img4_1.png",
        r"/home/mukta-hacker/Desktop/Mukta/sem6/CV_Project/Panorama-Effect-Image-Stitching/test_images/img4_2.png",
    ]

    # Load images and their dimensions
    images, dims = createPanorama(image_list)

    # Define feature type
    feature_type = cv2.SIFT_create()

    # Initialize the output with the first image
    out = images[0]

    # Iteratively match and stitch each image in the list
    for index in range(1, len(images)):
        print(f"Stitching image {index} to the panorama...")
        im2 = images[index]
        im1 = out  # Current stitched panorama
        dist_threshold = 0.7 #Adjust threshold as needed
        out, temp, _ = getMatches(im1, im2, feature_type, ransac_iterations=10, dist_threshold=dist_threshold)

        # Crop to non-zero area after each iteration
        out = getNonZeroImage(out)

    # Save the final stitched image
    showAndWrite(out, temp, path="stitched_panorama.png")