import cv2
import os

resolution=(1280,720)
folder = '../../Calibration Videos Data/1'

client_video = os.path.join(folder, 'client.h264')
server_right_video = os.path.join(folder, 'server_right.h264')
server_left_video = os.path.join(folder, 'server_left.h264')

cap_client = cv2.VideoCapture(client_video)
cap_serv_right = cv2.VideoCapture(server_right_video)
cap_serv_left = cv2.VideoCapture(server_left_video)

if cap_client.isOpened() and cap_serv_right.isOpened() and cap_serv_left.isOpened():
    ret_client, frame_client = cap_client.read()
    ret_serv_right, frame_serv_right = cap_serv_right.read()
    ret_serv_left, frame_serv_left = cap_serv_left.read()

    cv2.namedWindow("Client", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Server Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Server Left", cv2.WINDOW_NORMAL)

    i=0
    
    while True:
        if ret_serv_right == False or ret_client == False or ret_serv_left == False:
            print("Camera Calibration: No more Frames")
            cap_serv_left.release()
            cap_serv_right.release()
            cap_client.release()

            cv2.destroyAllWindows()
            break

        cv2.imshow("Server Right", frame_serv_right)
        cv2.imshow("Server Left", frame_serv_left)
        cv2.imshow("Client", frame_client)

        cv2.resizeWindow("Server Right", int(resolution[0] / 2), int(resolution[1] / 2))
        cv2.resizeWindow("Server Left", int(resolution[0] / 2), int(resolution[1] / 2))
        cv2.resizeWindow("Client", int(resolution[0] / 2), int(resolution[1] / 2))

        key_press = cv2.waitKey(1) & 0xFF

        if key_press == ord("n"):
            ret_serv_right, frame_serv_right = cap_serv_right.read()
            ret_serv_left, frame_serv_left = cap_serv_left.read()
            ret_client, frame_client = cap_client.read()
            continue

        elif key_press == ord("s"):
            cv2.imwrite(
                'image/calib_img_serv_right_{}.tiff'.format(i), frame_serv_right)
            cv2.imwrite(
                'image/calib_img_serv_left_{}.tiff'.format(i), frame_serv_left)
            cv2.imwrite(
                'image/calib_img_client_{}.tiff'.format(i), frame_client)

            i += 1
            ret_serv_right, frame_serv_right = cap_serv_right.read()
            ret_serv_left, frame_serv_left = cap_serv_left.read()
            ret_client, frame_client = cap_client.read()
            continue

        elif key_press == ord("q"):
            cap_serv_right.release()
            cap_serv_left.release()
            cap_client.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
