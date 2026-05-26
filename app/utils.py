import os
import pickle
from time import time

from collections import Counter
from detector import detectGameTopK
from embeddings import buildReferenceEmbeddings
from video import getVideoDetails, videoToFrames

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cachedEmbeddings")
os.makedirs(CACHE_DIR, exist_ok=True)


def cacheEmbeddings(embeddings):
    # print(embeddings)

    with open(os.path.join(CACHE_DIR, "refEmbed.pkl"), "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Cached Reference Embeddings")


def loadReferenceEmbeddings():
    if os.path.exists(os.path.join(CACHE_DIR, "refEmbed.pkl")):
        print(
            "-> Using cached Embedding.\n-> Delete cache if you want to create new embeddings"
        )

        with open(os.path.join(CACHE_DIR, "refEmbed.pkl"), "rb") as f:
            referenceEmbeddings = pickle.load(f)

        print("-> Loaded Successfully")

    else:
        # Build reference embeddings once
        print("-> No cache, building embeddings\n")

        referenceEmbeddings, _ = buildReferenceEmbeddings()

        cacheEmbeddings(referenceEmbeddings)

        print("-> Saved embeddings to cache")

    return referenceEmbeddings


def timeToSeconds(time: str):
    min, sec = time.split(":")
    secs = (int(min) * 60) + int(sec)

    return secs


def secondsToTime(seconds: int):

    minutes = seconds // 60
    remainingSeconds = seconds % 60

    return f"{minutes:02d}:{remainingSeconds:02d}"


def detectVideo(path, referenceEmbeddings, startTime="00:00", endTime=None):
    start = time()

    print("VIDEO DETECTION: ")
    if not os.path.exists(path):
        print("Video does not exist.\n")
        return { "error": "Video does not exist" }

    videoDuration = getVideoDetails(path)[2]

    startSeconds = timeToSeconds(startTime)

    if endTime is not None:
        endSeconds = timeToSeconds(endTime)
    else:
        endSeconds = int(videoDuration)

    if startSeconds >= videoDuration:
        print("Invalid start time")
        return {"error": "Invalid start time"}

    if endSeconds > videoDuration:
        endSeconds = int(videoDuration)

    if (endSeconds - startSeconds) > 90:
        endSeconds = startSeconds + 90
        print(f"-> Clip too long, trimmed to {secondsToTime(endSeconds)}")

    endTime = secondsToTime(endSeconds)

    clipName = os.path.splitext(os.path.basename(path))[0]

    print("\nExtracting frames...")

    frameFolder = videoToFrames(
        path,
        outputFolder=clipName,
        intervalSeconds=2,
        startTime=startTime,
        endTime=endTime,
    )

    if frameFolder is None:
        print("Frame extraction failed.\n")
        return {"error" : "Frame extraction failed."}

    smallFolder = os.path.join(frameFolder, "small")

    predictions = []
    allTopMatches = []

    for frameFile in os.listdir(smallFolder):
        if not frameFile.endswith(".jpg"):
            continue

        framePath = os.path.join(smallFolder, frameFile)

        result = detectGameTopK(framePath, referenceEmbeddings)

        if result["prediction"] == "Unknown Game":
            continue

        predictions.append(result["prediction"])
        allTopMatches.append(
            {
                "frame": framePath,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            }
        )

    ## Retrevial Level Filter Check
    if len(predictions) == 0:
        print("\nCould not confidently detect game.\n")
        return

    # Game Confidence (Vote Distribution Normalization)
    gameCounts = Counter(predictions)
    totalCount = len(predictions)

    voteConfidence = gameCounts.most_common(1)[0][1] / totalCount

    ## Vote Level Filter Check
    if voteConfidence < 0.6:
        print("\nCould not confidently determine the game.\n")
        return

    confidences = {
        game: (count / totalCount) * 100 for game, count in gameCounts.items()
    }

    sortedConfidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)

    # Game Prediction Result
    finalPrediction = gameCounts.most_common(1)[0][0]

    # Best frames from Clip (sorted by confidence)
    sortedFrames = sorted(allTopMatches, key=lambda x: x["confidence"], reverse=True)

    topFrames = sortedFrames[:3]

    influentialFrames = [item["frame"] for item in topFrames]

    print(f"\nDetected Game: {finalPrediction}\n")

    print("Confidence Breakdown:\n")
    for k, v in sortedConfidences:
        print(f"{k}: {v:.2f}%")

    print("\nMost Influential Frames from the Clip: ")
    for ref in influentialFrames:
        print(ref)

    processing_time = "{:.2f}".format(time() - start)
    print(f"\n---> Processing took {processing_time}s <---")

    return {
        "prediction": finalPrediction,
        "confidences": sortedConfidences,
        "influential_frames": influentialFrames,
        "time_taken": processing_time
    }


def detectFrame(path, referenceEmbeddings):
    if not os.path.exists(path):
        print("Image does not exist.\n")
        return {"error": "Image not found."}
    result = detectGameTopK(path, referenceEmbeddings)
    print("IMAGE DETECTION:\n")
    print("\nGame: ", result["prediction"])
    if result["prediction"] != "Unknown Game":
        print("Confidence:  ", result["confidence"])
        return result["prediction"], result["confidence"]

    return result["prediction"], None
