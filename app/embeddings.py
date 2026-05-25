import os
import torch
import open_clip
from open_clip import tokenizer
from PIL import Image
from collections import defaultdict

print("Torch version:", torch.__version__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BASE_DIR, "referenceGames")

# load the model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k'
)
# model.eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')

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
    
    # Normalize
    features /= features.norm(dim=-1, keepdim=True)

    return features.squeeze(0)


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
    
    
    for game, embeddings in referenceEmbeddings.items():
        
        stacked = torch.stack([
            item["embedding"]
            for item in embeddings
        ])
        
        meanEmbedding = torch.mean(stacked, dim = 0)
        
        meanEmbedding /= meanEmbedding.norm(dim=-1, keepdim=True)
        gameMeanEmbeddings[game] = meanEmbedding

    return referenceEmbeddings, gameMeanEmbeddings
