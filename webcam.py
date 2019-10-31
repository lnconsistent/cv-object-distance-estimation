import cv2
import pandas as pd
import os
import argparse
from gtts import gTTS
from utils import generate_csv as gv
import pygame

parser = argparse.ArgumentParser(description='Control the audio output.')
parser.add_argument('--verbose', type=str, default='',
                    help='change output to audio')
args = parser.parse_args()
VERBOSITY = args.verbose

def main():
    df_label = pd.read_csv("labels.csv")
    video_capture = cv2.VideoCapture(0)
    ct = 0
    '''
    while True:
        # Capture frame by frame
        ret, frame = video_capture.read()
        ct += 1
        if ct%20 != 0:
            continue
        df = gv.gen_depth(frame.transpose((2, 0, 1)))
        # Draw a rectangle around detected landmarks
        for idx, row in df.iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if int(row['zloc_pred']) > 0:
                dist = str(row['zloc_pred']) + ' ft'
                if float(row['confidence']) < 0.75:
                    identity = 'unknown'
                else:
                    identity = df_label.at[int(row['class']), 'label']
                cv2.putText(frame, dist, (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 1,
                            cv2.LINE_AA)
                cv2.putText(frame, identity, (x1, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255), 1, cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    '''

    cv2.namedWindow("test")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            ct += 1
            df = gv.gen_depth(frame.transpose((2, 0, 1)))
            # Draw a rectangle around detected landmarks
            for idx, row in df.iterrows():
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if int(row['zloc_pred']) > 0:
                    dist = str(row['zloc_pred']) + ' ft'
                    if float(row['confidence']) < 0.75:
                        identity = 'unknown'
                    else:
                        identity = df_label.at[int(row['class']), 'label']
                    if VERBOSITY==1 and int(row['zloc_pred']) <= 20:
                        pygame.mixer.init()
                        tts = gTTS(text=identity+' is '+dist+' ahead', lang='en')
                        tts.save('sound.mp3')
                        pygame.mixer.music.load('sound.mp3')
                        pygame.mixer.music.play()
                        busy = True
                        while busy == True:
                            if pygame.mixer.music.get_busy() == False:
                                busy = False
                        pygame.quit()
                        os.remove('sound.mp3')
                    cv2.putText(frame, dist, (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 1,
                                cv2.LINE_AA)
                    cv2.putText(frame, identity, (x1, y2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Video', frame)


    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()