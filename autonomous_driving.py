import torch
import cv2
from pynput import keyboard
from picamera2 import Picamera2
import numpy as np
import os
import threading
import time
import motor_cont
import util
from datetime import datetime
from keyboard_cont import KeyboardController


# Load the trained model
model = torch.load(".//model//best_model.pth", map_location=torch.device("cpu"))
model.eval()

# Create folder if it does not exist
util.makeImgDir()

# Function for capturing keyboard input
key = KeyboardController()

pred_image = None

# Capture and save camera images
def capture_img():
    global pred_image, show_image
    save_cnt = 0
    frame_cnt = 0
    t0 = 0

    try:
        picam2 = Picamera2()
        # just for preview
        picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1332, 990)}))
        picam2.start()

        while True:             
            # Capture camera image
            frame = picam2.capture_array()
            
            # Preprocessing raw camera image
            frame = cv2.flip(frame, -1)
            frame = cv2.resize(frame, (512, 512)) # for debugging the area of ROI
            frame = frame[200:,:]                 # remove none ROI area
            frame = cv2.resize(frame, (64, 64))   # resizs for matching the model unput
            
            # Just for realtime debugging
            show_image = frame
            cv2.imshow('Frame', show_image)
            cv2.waitKey(40)
            
            # Save preprocessed image
            save_image = frame

            # Another preprocess image before inference
            frame = np.asarray(frame, dtype=np.float32).reshape(1, 64, 64, 3)  # (1, 224, 224, 3)
            frame = (frame / 127.5) - 1

            # Assign image for training/inference and saving (pred_image)
            pred_image = frame

            # Save 1 out of every 4 frames
            frame_cnt += 1
            capture_freq = 4

            # Save image
            if frame_cnt % capture_freq == 0 and key.save_flag == 1:
                formatted_time = datetime.now().strftime("%M%S%f")[:-3]  # Remove the last 3 digits to get milliseconds

                # Determine image save directory
                if (key.go_flag == 1) and (key.left_flag == 1) and (key.right_flag == 0):
                    directory = 'image' + os.sep + 'left'
                elif (key.go_flag == 1) and (key.left_flag == 0) and (key.right_flag == 1):
                    directory = 'image' + os.sep + 'right'
                elif (key.go_flag == 1) and (key.left_flag == 0) and (key.right_flag == 0):
                    directory = 'image' + os.sep + 'go'
                else:
                    directory = 'image' + os.sep + 'other'

                # Image file name
                file_name = f"{directory}/{key.go_flag}{key.left_flag}{key.right_flag}{key.back_flag}_{frame_cnt}_{formatted_time}.jpg"
                
                # Save the image
                cv2.imwrite(file_name, save_image)
                save_cnt += 1
                
                # Calculate fps
                fps = util.calc_fps(t0)
                t0 = time.time()  # for fps calculation
                print(f"Image saved as {file_name}, fps: {fps},  cnt : {save_cnt}")
            else:
                pass

    except Exception as error:
        print('An error occurred in capture_img function!')
        print(error)
        pass


def drive_mode():
    global pred_image
    t0 = 0

    while True:
        time.sleep(0.08)
        if pred_image is None:
            pass
        else:
            pred_image = pred_image[:, :, ::-1] 
            input_tensor = torch.from_numpy(pred_image).float()  # Convert numpy array to torch tensor
            input_tensor = input_tensor.permute(0, 3, 1, 2)  # (1, 64, 64, 3) -> (1, 3, 64, 64)
            with torch.no_grad():
                prediction = model(input_tensor)
            prediction = prediction.numpy()  # Convert result to numpy array (for postprocessing and np.argmax)
                        
            print('go: ', str(round(prediction[0][0], 2)), '  left:', str(round(prediction[0][1], 2)), '   right:', str(round(prediction[0][2], 2)))
            prediction = np.argmax(prediction, axis=1)
            pass

        # Autonomous driving mode
        if key.manual == 0:
            if prediction == 0:  # Go straight
                key.go_flag = 1
                key.left_flag = 0
                key.right_flag = 0
                key.back_flag = 0
            elif prediction == 1:  # Turn left
                key.go_flag = 1
                key.left_flag = 1
                key.right_flag = 0
                key.back_flag = 0
            elif prediction == 2:  # Turn right
                key.go_flag = 1
                key.left_flag = 0
                key.right_flag = 1
                key.back_flag = 0
                break
        else:
            None

        # Execute motor control
        motor_cont.drive(key.go_flag, key.left_flag, key.right_flag, key.back_flag)

        fps = util.calc_fps(t0)
        t0 = time.time()  # for fps calculation
        print(f"fps: {fps}")



if __name__ == "__main__":
    getkey_thread = threading.Thread(target=key.getkeyboard)
    getkey_thread.start()
    print('keyboard on!')

    drive_thread = threading.Thread(target=drive_mode)
    drive_thread.start()
    print('Motor on!')

    main_thread = threading.Thread(target=capture_img)
    main_thread.start()
    print('main on!')
