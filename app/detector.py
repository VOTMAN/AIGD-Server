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

            similarities.append(
                {"game": game, "score": similarity, "path": item["path"]}
            )

    similarities.sort(key=lambda x: x["score"], reverse=True)

    TOPK = 5
    topMatches = similarities[:TOPK]

    if len(topMatches) < 2:
        return {
            "prediction": "Unknown Game",
            "confidence": 0,
            "top_matches": topMatches,
        }

    top1 = topMatches[0]["score"]
    top1Game = topMatches[0]["game"]

    top2diff = next(
        (
            match["score"] for match in topMatches 
            if top1Game != match["game"]
        ),
        None
    )
    
    if top2diff:
        margin = top1 - top2diff
    else:
        margin = top1

    negativeScore = max(
        (match["score"] for match in topMatches if match["game"] == "NEGATIVE"),
        default=0,
    )

    if (top1 < 0.65 or margin < 0.03) or (negativeScore >= top1 - 0.03):
        return {
            "prediction": "Unknown Game",
            "confidence": 0,
            "top_matches": topMatches,
        }

    votes = Counter()

    for match in topMatches:
        votes[match["game"]] += match["score"]

    if not bool(votes):
        return {"prediction": "Unknown Game"}

    predictedGame = votes.most_common(1)[0][0]

    return {
        "prediction": predictedGame,
        "confidence": topMatches[0]["score"],
        "top_matches": topMatches,
    }
