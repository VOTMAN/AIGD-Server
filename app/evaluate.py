import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from detector import detectGameTopK
from embeddings import loadReferenceEmbeddings

UNKNOWN_LABEL = "Unknown Game"
NEGATIVE_CLASS = "NEGATIVE"

referenceEmbeddings = loadReferenceEmbeddings()

TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testing")

def eval():
    if not os.path.isdir(TEST_DIR):
        print("No testing directory")
        return 
    
    y_true = []
    y_pred = []

    for trueGame in os.listdir(TEST_DIR):
        gameRes = []
        gameFolder = os.path.join(TEST_DIR, trueGame)
        for imgName in os.listdir(gameFolder):
            imgPath = os.path.join(gameFolder, imgName)
            res = detectGameTopK(imgPath, referenceEmbeddings)
            gameRes.append(res)
            true_label = UNKNOWN_LABEL if trueGame == NEGATIVE_CLASS else trueGame
            y_true.append(true_label)
            y_pred.append(res["prediction"])

        for item in gameRes:
            if item["prediction"] == "Unknown Game":
                continue
            elif item["prediction"] != trueGame:
                print("\nWRONG PREDICTION:")
                print("True: ", trueGame)
                print("Pred: ", item["prediction"])
                print("Image:", imgPath)
                print("Confidence: ", item["confidence"])

                for match in item["top_matches"]:
                    print(match["game"], round(match["score"], 3))
    
    games = [
        g for g in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, g)) and g != NEGATIVE_CLASS
    ]

    labels = games + ["Unknown Game"]

    overall_accuracy = accuracy_score(y_true, y_pred)
    
    precision = precision_score(
        y_true, y_pred, 
        average=None, 
        labels=games, 
        zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, 
        average=None, 
        labels=games, 
        zero_division=0
    )
    
    f1 = f1_score(
        y_true, y_pred, 
        average=None, 
        labels=games, 
        zero_division=0
    )

    unknown_rate = y_pred.count("Unknown Game") / len(y_pred)

    negative_indices = [i for i, t in enumerate(y_true) if t == UNKNOWN_LABEL]
    correct_rejections = sum(1 for i in negative_indices if y_pred[i] == UNKNOWN_LABEL)
    false_positive_rate = 1 - (correct_rejections / len(negative_indices)) if negative_indices else 0
   
    print(f"\nOverall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Unknown Rate: {unknown_rate * 100:.2f}%")
    print(f"False Postive Rate (NEGATIVE predicted as a game): {false_positive_rate * 100:.2f}%")
    print(f"\n{'Game':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    for i, game in enumerate(games):
        print(f"{game:<20} {precision[i]:>10.2f} {recall[i]:>10.2f} {f1[i]:>10.2f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot()
    plt.show()

eval()   
