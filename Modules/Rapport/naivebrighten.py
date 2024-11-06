import cv2 as cv
import copy

image = cv.imread("Data/LOLdataset/our485/low/750.png")
#cv.imshow("Original Image", image)

converted = cv.cvtColor(image, cv.COLOR_RGB2HSV)

def scale_channel(image):
    working_image = copy.deepcopy(image)
    multiplier = 4
    channel_to_scale = 2
    h = working_image.shape[0]
    start = working_image.shape[1] // 2
    end = working_image.shape[1]

    for y in range(0, h):
        for x in range (start, end):
            working_image[y, x][channel_to_scale] = working_image[y, x][channel_to_scale] * multiplier
    return working_image

scaled = scale_channel(converted)
returned = cv.cvtColor(scaled, cv.COLOR_HSV2RGB)
cv.imshow("Scaled", returned)


cv.waitKey(0)