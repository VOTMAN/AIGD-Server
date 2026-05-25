from collections import Counter
from embeddings import getImgEmbeddings

def detectGameTopK(queryImgPath, referenceEmbeddings):
    queryEmbedding = getImgEmbeddings(queryImgPath)
    similarities = []

    for game, embedding in referenceEmbeddings.items():
        for item in embedding:
            similarity = (queryEmbedding @ item["embedding"].T).item()

            if similarity < 0.3:
                continue
            
            similarities.append({
                "game": game,
                "score": similarity,
                "path": item["path"]
            })

    similarities.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    TOPK = 5
    topMatches = similarities[:TOPK]

    # print(topMatches)
    top1 = topMatches[0]["score"]
    top2 = topMatches[1]["score"]
    margin = top1 - top2
    # print(top1, top2, margin)

    if top1 < 0.65 or margin < 0.03:
        return {
            "prediction": "Unknown Game",
            "confidence": top1,
            "top_matches": topMatches
        }
    
    votes = Counter()

    for match in topMatches:
        # votes[match["game"]] += 1
        votes[match["game"]] += match["score"]

    # print(votes)
    if not bool(votes):
        return {
            "prediction": "Unknown Game"
        }
    
    predictedGame = votes.most_common(1)[0][0]

    return {
        "prediction": predictedGame,
        "confidence": topMatches[0]["score"],
        "top_matches": topMatches
    }
