import os
import pickle
import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from collections import defaultdict

print("Torch version:", torch.__version__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BASE_DIR, "referenceGames")
CACHE_DIR = os.path.join(BASE_DIR, "cachedEmbeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

# load the model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)
# model.eval()

# tokenizer = open_clip.get_tokenizer('ViT-B-32')

def getImgEmbeddings(imgPath):
    try:
        image = preprocess(
            Image.open(imgPath)
        ).unsqueeze(0)

    except FileNotFoundError:
        print("No such image exitsts")
        return

    with torch.no_grad():
        features = model.encode_image(image)

    features /= features.norm(dim=-1, keepdim=True)

    return features.squeeze(0)

def getTextEmbeddings(text):

    prompt = f"a screenshot of {text}, gameplay"
    processed_text = tokenizer.tokenize(prompt)

    with torch.no_grad():
        features = model.encode_text(processed_text)

    features /= features.norm(dim=-1, keepdim=True)

    return features


def buildReferenceEmbeddings(refFolder = REF_DIR):
    referenceEmbeddings = defaultdict(list)
    gameMeanEmbeddings = {}

    for game in os.listdir(refFolder):
        gameFolder = os.path.join(refFolder, game)

        for imgFile in os.listdir(gameFolder):
            imgPath = os.path.join(gameFolder, imgFile)
            embedding = getImgEmbeddings(imgPath)

            referenceEmbeddings[game].append({
                "embedding": embedding,
                "path": imgPath
            })


    # for game, embeddings in referenceEmbeddings.items():

    #     stacked = torch.stack([
    #         item["embedding"]
    #         for item in embeddings
    #     ])

    #     meanEmbedding = torch.mean(stacked, dim = 0)

    #     meanEmbedding /= meanEmbedding.norm(dim=-1, keepdim=True)
    #     gameMeanEmbeddings[game] = meanEmbedding

    return referenceEmbeddings, gameMeanEmbeddings

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
