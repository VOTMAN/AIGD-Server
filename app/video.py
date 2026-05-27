import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

VIDEO_DIR = os.path.join(DATA_DIR, "extractedFrames")
os.makedirs(VIDEO_DIR, exist_ok=True)

def getVideoDetails(videoPath):
    cap = cv2.VideoCapture(videoPath)
    
    if not cap.isOpened():
        print("Error: Cound not open the video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    durationSecond = frameCount / fps

    cap.release()
    return fps, frameCount, durationSecond

def videoToFrames(videoPath, outputFolder = "misc", startTime = "00:00", endTime = None, intervalSeconds = 2, fileExt = "jpg"):
    SMin, SSec = startTime.split(":")
    startTime = (int(SMin) * 60) + int(SSec)

    if endTime is not None: 
        EMin, ESec = endTime.split(":")
        endTime = (int(EMin) * 60) + int(ESec)
    

    if endTime is not None and endTime <= startTime:
        print("Error: Invalid range")
        return
    
    finalOpFolder = os.path.join(VIDEO_DIR, outputFolder)
    os.makedirs(finalOpFolder, exist_ok=True)

    smallDir = os.path.join(finalOpFolder, "small")
    medDir = os.path.join(finalOpFolder, "medium")
    # origDir = os.path.join(finalOpFolder, "original")

    os.makedirs(smallDir, exist_ok=True)
    os.makedirs(medDir, exist_ok=True)
    # os.makedirs(origDir, exist_ok=True)
    
    fps, frameCount, durationSecond = getVideoDetails(videoPath)

    cap = cv2.VideoCapture(videoPath)
    
    if endTime == None:
        endTime = durationSecond
    
    if startTime >= durationSecond:
        print("Error: startTime exceeds video duration")
        return
        
    if endTime is not None and endTime > durationSecond:
        print("Error: endTime exceeds video duration")
        return
    
    selectedDuration = (
        endTime - startTime
        if endTime is not None
        else durationSecond - startTime
    )

    print("Selected Duration: ", selectedDuration)
    
    currentMsec = startTime * 1000
    frameIndex = 0

    if endTime == None:
        endTime = durationSecond
    
    while currentMsec <= endTime * 1000:
        cap.set(cv2.CAP_PROP_POS_MSEC, currentMsec)

        ret, frame = cap.read()

        if not ret:
            break
        
        # Small Frames
        small_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        filename_small = os.path.join(smallDir, f"frame_{frameIndex:04d}_small.{fileExt}")
        cv2.imwrite(filename_small, small_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Medium Frames
        medium_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        filename_med = os.path.join(medDir, f"frame_{frameIndex:04d}_medium.{fileExt}")
        cv2.imwrite(filename_med, medium_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Original Frame
        # filename = os.path.join(origDir, f"frame_{frameIndex:04d}_original.{fileExt}")
        # cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        frameIndex += 1
        currentMsec += intervalSeconds * 1000

    cap.release()
    print("Extraction Complete\n")
    return finalOpFolder
