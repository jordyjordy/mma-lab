import numpy as np
import cv2
import matplotlib.pyplot as plt
from video_tools import *
import feature_extraction as ft
import scipy.io.wavfile as wav
from scikits.talkbox.features import mfcc

def colorhist_diff(frame, prev_frame):
    diff = 0
    hist = ft.colorhist(frame)
    prev_hist = ft.colorhist(prev_frame)

    for i in range(hist.shape[1]):
        diff = diff + np.sum(np.abs(prev_hist[:, i] - hist[:, i]))
    return diff


def temporal_diff(frame, prev_frame,threshold=130):
    diff = 0
    difftotal = np.abs(prev_frame.astype('int16') -     frame.astype('int16'))
    difftotal = np.where(difftotal>threshold,1,0)
    for i in range(difftotal.shape[2]):
        diff += np.sum(difftotal[:, :, i])

    return diff

def videoframe_times(framenum,framerate):
    return (framenum*(1/framerate),framenum+1*(1/framerate))

def audio_signal_power(startsample,length,samples):
    power = 0
    for x in range(startsample,startsample+length):
        samplespot = samples[x][0] + samples[x][1]
        power += samplespot**2
    return power/length


# Path to video file to analyse 
video = '../Videos/BlackKnight.avi'
audio = '../Videos/BlackKnight.wav'
samplerate, samples = wav.read(audio)
print samplerate
frame_rate = get_frame_rate(video)
print frame_rate , "framerate"
# starting point
S = 0 # seconds
# stop at
E = 60 # seconds
(start,end) = videoframe_times(0,frame_rate)
framelength = end-start
audioperframe = int(framelength*samplerate)
# Retrieve frame count. We need to add one to the frame count because cv2 somehow 
# has one extra frame compared to the number returned by avprobe.
frame_count = get_frame_count(video) + 1


# create an cv2 capture object
cap = cv2.VideoCapture(video)

# store previous frame
prev_frame = None

# set video capture object to specific point in time
cap.set(cv2.CAP_PROP_POS_MSEC, S*1000)
hist = []
previoushist = []
CH = []
AI = []
i = 0
while(cap.isOpened() and cap.get(cv2.CAP_PROP_POS_MSEC) < (E*1000)):

    # 
    retVal, frame = cap.read()
    # 
    if retVal == False:
        break

    #== Do your processing here ==#
    if prev_frame is not None:
        # CH.append(colorhist_diff(frame,prev_frame))
        CH.append(temporal_diff(frame,prev_frame))
        startsample = i * audioperframe
        AI.append(audio_signal_power(int(startsample), audioperframe, samples))


    # 
    # cv2.imshow('Video', frame)



    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    prev_frame = frame
    i += 1

#
framerate = cap.get(cv2.CAP_PROP_FPS)
print framerate
time = np.arange(start=S,stop=E, step=1/framerate)
print time.shape
cap.release()
cv2.destroyAllWindows()
plt.stem(time[0:len(CH)],CH)
plt.show()
plt.stem(time[0:len(AI)],AI)
plt.show()