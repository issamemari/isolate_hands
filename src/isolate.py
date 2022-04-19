import cv2
import os
import argparse
import numpy as np
import math


def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def isolate_hands(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(
        thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    masks = []

    for i in range(3):
        mask = np.zeros_like(image)
        mask = cv2.drawContours(
            mask, contours[i : i + 1], -1, (0, 255, 0), thickness=cv2.FILLED
        )
        masks.append(mask[:, :, 1])

    angles = []
    centers = []
    areas = []

    for mask in masks:
        moments = cv2.moments(mask)
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]

        a = moments["m20"] / moments["m00"] - center_x ** 2
        b = 2 * (moments["m11"] / moments["m00"] - center_x * center_y)
        c = moments["m02"] / moments["m00"] - center_y ** 2

        flag = 1 if a < c else 0
        theta = 1 / 2 * math.atan(b / (a - c)) + flag * math.pi / 2

        angles.append(theta * 180 / math.pi - 90)
        centers.append((center_x, center_y))
        areas.append(moments["m00"])

    rotated_masks = []

    for mask, angle, center in zip(masks, angles, centers):
        rotated_masks.append(rotate_image(mask, angle, center))

    bboxes = []

    for rotated_mask in rotated_masks:
        where = np.where(rotated_mask > 0)
        y1 = where[0].min()
        x1 = where[1].min()
        y2 = where[0].max()
        x2 = where[1].max()
        bboxes.append((x1, y1, x2, y2))

    rotated_images = []

    for angle, center in zip(angles, centers):
        rotated_images.append(rotate_image(image, angle, center))

    cropped_images = []

    for rotated_image, bbox in zip(rotated_images, bboxes):
        cropped_images.append(rotated_image[bbox[1] : bbox[3], bbox[0] : bbox[2], :])

    cropped_masks = []

    for rotated_mask, bbox in zip(rotated_masks, bboxes):
        cropped_masks.append(rotated_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]])

    transparent_crops = []

    for cropped_image, cropped_mask in zip(cropped_images, cropped_masks):
        where = np.where(cropped_mask == 0)
        transparent_crop = np.full(cropped_image.shape[:2] + (4,), 255, dtype=np.int32)
        transparent_crop[:, :, :3] = cropped_image
        transparent_crop[where[0], where[1], 3] = 0
        transparent_crops.append(transparent_crop)

    # sort based on area
    transparent_crops = [
        x for _, x in sorted(zip(areas, transparent_crops), reverse=True)
    ]

    # shortest hand is the hour hand
    min_length = 10000
    hour_hand_index = None
    for i, crop in enumerate(transparent_crops):
        if crop.shape[0] < min_length:
            min_length = crop.shape[0]
            hour_hand_index = i

    minute_hand_index = None
    second_hand_index = None
    for i, _ in enumerate(transparent_crops):
        if i != hour_hand_index:
            if minute_hand_index is not None:
                second_hand_index = i
            else:
                minute_hand_index = i

    canvases = []
    for cropped_image, transparent_crop in zip(cropped_images, transparent_crops):
        img = cropped_image.copy()
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            cimg, cv2.HOUGH_GRADIENT, 0.05, 20, param1=20, param2=60, minRadius=2
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 1:
                circles = circles[1:]
        else:
            return {
                "hour": None,
                "minute": None,
                "second": None,
            }

        canvas_size = 1000
        canvas = np.full([canvas_size, canvas_size, 4], 0, dtype=np.int32)

        y1 = canvas_size // 2 - circles[0][1]
        y2 = canvas_size // 2 + cropped_image.shape[0] - circles[0][1]

        x1 = canvas_size // 2 - cropped_image.shape[1] + circles[0][0]
        x2 = canvas_size // 2 + circles[0][0]

        canvas[y1:y2, x1:x2, :] = transparent_crop
        canvases.append(canvas)


    return {
        "hour": canvases[hour_hand_index],
        "minute": canvases[minute_hand_index],
        "second": canvases[second_hand_index],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)]

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        for hand_name, hand_image in isolate_hands(image).items():
            if hand_image is not None:
                cv2.imwrite(
                    os.path.join(args.output_dir, f"{image_name}_{hand_name}.png"),
                    hand_image,
                )


if __name__ == "__main__":
    main()
