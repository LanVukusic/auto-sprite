import numpy as np
import PIL.Image as pil
import os
import cv2


IMAGE_TOLERANCE = 50

# define a function to remove a bg color from an image
def flood_fill_remove(color, start_x, start_y, image):
    # check if the color is the same as the color to remove, using a tolerance
    pixles_to_check = [(start_x, start_y)]

    # create a mask to keep track of which pixles have been visited
    # pad the mask with 1 pixel to avoid index out of bounds errors
    visited_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=bool)

    print("starting flood fill", color, start_x, start_y)

    i = 0
    while len(pixles_to_check) > 0:
        x, y = pixles_to_check.pop()

        i += 1
        if i % 50000 == 0:
            print("i", i)

        # check if the pixel is within the image
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            # check if the color is the same as the color to remove, using a tolerance
            color_diff = image[x, y] - color
            collective_diff = np.sum(np.abs(color_diff))

            if collective_diff < IMAGE_TOLERANCE:
                # set the color to black
                image[x, y] = [0, 0, 0, 0]

                # add the pixles around the current pixle to the list of pixles to check if they are not in the visited mask
                if not visited_mask[x + 1, y]:
                    pixles_to_check.append((x + 1, y))
                    visited_mask[x + 1, y] = True
                if not visited_mask[x - 1, y]:
                    pixles_to_check.append((x - 1, y))
                    visited_mask[x - 1, y] = True
                if not visited_mask[x, y + 1]:
                    pixles_to_check.append((x, y + 1))
                    visited_mask[x, y + 1] = True
                if not visited_mask[x, y - 1]:
                    pixles_to_check.append((x, y - 1))
                    visited_mask[x, y - 1] = True


# find the bounding box of an image
def find_bounding_box(image, start_x, start_y):
    pixles_to_check = [(start_x, start_y)]
    visited_mask = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=bool)

    min_x, max_x, min_y, max_y = 999, -1, 999, -1

    non_empty_pixles = 0

    # iterate over all the pixels around the current pixel
    while len(pixles_to_check) > 0:
        x, y = pixles_to_check.pop()

        # check if the pixel is within the image
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            # check if alpha is non-zero
            if image[x, y, 3] > 0:
                non_empty_pixles += 1

                # update the bounding box
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                # add the pixles around the current pixle to the list of pixles to check if they are not in the visited mask
                if not visited_mask[x + 1, y]:
                    pixles_to_check.append((x + 1, y))
                    visited_mask[x + 1, y] = True
                if not visited_mask[x - 1, y]:
                    pixles_to_check.append((x - 1, y))
                    visited_mask[x - 1, y] = True
                if not visited_mask[x, y + 1]:
                    pixles_to_check.append((x, y + 1))
                    visited_mask[x, y + 1] = True
                if not visited_mask[x, y - 1]:
                    pixles_to_check.append((x, y - 1))
                    visited_mask[x, y - 1] = True

    return min_x, max_x, min_y, max_y


# dialate and erode the image to remove small artifacts
def clean_image(image):
    # create a binary image from the alpha channel
    mask = image[:, :, 3]
    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    # dialate and erode the image to remove small artifacts
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # set the alpha channel to the cleaned mask
    image[:, :, 3] = mask * 255
    return image


# iterate over all images in the "images_in" folder
for filename in os.listdir("images_in"):
    # open the image in RGBA mode
    img = pil.open("images_in/" + filename)
    # lower color depth to 8 bits
    img = img.quantize(256).convert("RGBA")
    # convert the image to a numpy array
    img = np.array(img)
    print("shape", img.shape, filename)
    # do some processing on the image

    # get the most used color
    colors, counts = np.unique(
        img.reshape(-1, img.shape[2]), axis=0, return_counts=True
    )
    most_used_color = colors[np.argmax(counts)]
    print("most used color", most_used_color)

    # check if image has transparency
    if most_used_color[3] != 0:
        # remove the bg color from the image
        flood_fill_remove(most_used_color, 0, 0, img)

        # clean the image
        img = clean_image(img)

    # visualize the image
    pil_img = pil.fromarray(img)
    pil_img.show()

    i = 0
    filename = filename.split(".")[0]

    # iterate over the image and find the first non-transparent pixel indicating a new icon
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # check if alpha is non-zero and a new icon has not been found
            if img[x, y, 3] > 0:
                # find the bounding box of the image
                min_x, max_x, min_y, max_y = find_bounding_box(img, x, y)
                if min_x == None:
                    print("skipping image")
                    continue
                # crop the image
                img_out = img[min_x:max_x, min_y:max_y]
                # save the image
                # if image has invalid dimensions, skip it
                if img_out.shape[0] == 0 or img_out.shape[1] == 0:
                    continue
                # if image is too small, skip it
                if img_out.shape[0] < 10 or img_out.shape[1] < 10:
                    continue

                # # dialate image out
                # kernel = np.ones((3, 3), npt.uint8)
                # img_out[:, :, 3] = cv2.(img_out[:, :, 3], kernel, iterations=2)

                i += 1
                pils = pil.fromarray(img_out)
                pils.save(
                    "images_out/" + "{filename}_{i}.png".format(filename=filename, i=i),
                )

                # remove the icon from the image
                img[min_x:max_x, min_y:max_y] = np.zeros(
                    (max_x - min_x, max_y - min_y, 4)
                )

                print("found icon", i)
                print(img.shape)
                break
        else:
            continue
