# from src.amr_simulation.amr_simulation.robot import Robot

# import math
# import numpy as np

# s = np.random.uniform(0, 1, 1000)  # Draws 1000 samples from an interval of [0, 1)


# # a = [2, 3]
# # b = a[:]
# # print(a == b)
# # print(a is b)


# # # a = [a + 1 for a in a]
# # for x in a:
# #     if x < math.pi():
# #         c.append()
# # c = [x if x < math.pi else x - math.pi * 2 for x in a]
# # print(c)

# a = np.random.random((10, 3))
# print(a)
# for index, x in enumerate(a):
#     print(index, x[0])

import cv2
import os


# Create an object to read
# from camera
# video = cv2.VideoCapture(0)

# We need to check if camera
# is opened previously or not
# if video.isOpened() == False:
#     print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
im0 = cv2.imread("./src/amr_localization/img/2023-03-09_20-21-08/0000 initialization.png")
frame_width = im0.shape[0]
frame_height = im0.shape[1]

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(
    "whitenoise0.mp4",
    #  cv2.VideoWriter_fourcc(*'MP4V'),
    0x7634706D,
    10,
    size,
)

for im_name in os.listdir("./src/amr_localization/img/2023-03-09_20-21-08"):
    # Write the frame into the
    # file 'filename.avi'
    frame = cv2.imread(os.path.join("./src/amr_localization/img/2023-03-09_20-21-08", im_name))
    result.write(frame)

    # Display the frame
    # saved in the file
    # cv2.imshow("Frame", frame)

    # Press S on keyboard
    # to stop the process
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

# When everything done, release
# the video capture and video
# write objects
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
