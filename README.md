# AI Game Detector

Detects which video game is being played in a video clip using AI-powered image embeddings. Drop in a clip or a screenshot, get the game name.

Currently supports **CS2, Elden Ring, Hollow Knight, and Valorant**, with pre-built embeddings shipped directly in the repo.

---

## How It Works

The detector uses **OpenCLIP (ViT-B-32)** to convert images into semantic embeddings — vector representations that capture visual meaning. These are compared against a library of reference screenshots to find the closest match.

### Detection Flow

```
Video Clip / Screenshot
    │
    ▼
Extract a frame every 2 seconds           (video.py)        [Video only]
    │
    ▼
For each frame:
  → Generate embedding via OpenCLIP       (embeddings.py)
  → Compare against reference library
  → Pick best game for that frame         (detector.py)
    │
    ▼
Majority vote across all frames           (main.py)         [Video only]
    │
    ▼
Final Prediction: "Game Name"
```

### Two-Level Voting (Video)

Detection runs at two levels for robustness:

- **Frame level** (`detector.py`) — For each frame, the top 5 most similar reference images are found. Similarity scores are accumulated per game and the highest-scoring game wins that frame.
- **Video level** (`main.py`) — Each frame casts one vote. The game with the most frame-level wins is the final prediction.

This means a single ambiguous frame (menu screen, cutscene, black screen) won't throw off the result — it gets outvoted by consistent gameplay frames.

### Embedding Cache

Pre-built embeddings for all supported games are shipped in `cachedEmbeddings/refEmbed.pkl`. The app loads this automatically on startup — no build step needed.

---

## Project Structure

```
.
├── app/
│   ├── main.py              # CLI entry point + shared detectVideo / detectFrame functions
│   ├── detector.py          # Frame-level game detection
│   ├── embeddings.py        # OpenCLIP model + embedding functions
│   ├── video.py             # Video frame extraction
│   └── utils.py             # Embedding cache helpers + shared detection wrappers
├── backend/
│   ├── main.py              # FastAPI server
│   └──db/
│      ├── db.py                # SQLite database helpers (init, save, get, getAll)
│      └── models.py            # SQLModel table definitions (PredResults)
├── cachedEmbeddings/
│   └── refEmbed.pkl         # Pre-built embeddings (shipped with repo)
├── extractedFrames/         # Auto-generated during detection
│   └── save/                # Persistent influential frames served to the frontend (The server auto delete after 2 hours)
└── requirements.txt

```

---

## Setup

**1. Clone the repo and create a virtual environment**

```bash
git clone <repo-url>
cd aipro
python -m venv myenv
source myenv/bin/activate       # Windows: myenv\Scripts\activate
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. (Optional) Build your own reference embeddings**

If you want to add your own games, create a `referenceGames/` folder at the project root and populate it with screenshots:

```
referenceGames/
└── Your_Game/
    ├── screenshot1.png
    ├── screenshot2.png
    └── ...
```

Then delete `cachedEmbeddings/refEmbed.pkl` — new embeddings will be generated automatically the next time `main.py` runs.

More screenshots per game = better accuracy. Aim for at least 10–20 varied screenshots covering different scenes, HUDs, and gameplay moments.

---

## Usage — CLI

Run from inside the `app/` directory:

```bash
cd app
python main.py
```

You'll be prompted to enter a path to a video or image:

```
=== AI Game Detector ===

Enter video path (or q to quit): ../videoClips/Valorant.mp4

Extracting frames...
Selected Duration: 118.0
Extraction Complete

Detected Game: Valorant

Confidence Breakdown:
Valorant: 100.00%

Most Influential Frames from the Clip:
../extractedFrames/Valorant/small/frame_0003_small.jpg
../extractedFrames/Valorant/small/frame_0010_small.jpg
../extractedFrames/Valorant/small/frame_0021_small.jpg
```

- Videos longer than **3 minutes** are automatically capped at the first 3 minutes.
- Start and end times can be customised via command line arguments:
  ```bash
  python main.py <video_path> <startTime> <endTime>
  # e.g. python main.py ../videoClips/Valorant.mp4 00:30 02:00
  ```
- If no frame clears the similarity threshold, the result is reported as **Unknown Game**.
- Enter `q` to quit.

---

## Usage — Backend (FastAPI)

Run from inside the `backend/` directory:

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Endpoints

#### `POST /api/upload/clip`

Detect the game in a video clip.

**Form fields:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | File | Yes | — | Video file (.mp4, .mov, .avi) |
| `startTime` | String | No | `00:00` | Start time in MM:SS format |
| `endTime` | String | No | None | End time in MM:SS format |

**Response:**

```json
{
  "id": "98a0683c-9ad5-40d9-bbb4-b6fe1d5d7c70"
}
```

Detection result is saved to the database. Use `GET /api/results/{id}` to retrieve it.

---

#### `POST /api/upload/frame`

Detect the game in a single screenshot.

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Image file (.png, .jpg, .jpeg) |

**Response:**

```json
{
  "prediction": "Valorant",
  "confidence": 0.91
}
```

---

#### `GET /api/results/{id}`

Retrieve a stored detection result by ID.

**Response:**

```json
{
  "id": "98a0683c-9ad5-40d9-bbb4-b6fe1d5d7c70",
  "status": "success"
  "clip_name": "Valorant.mp4",
  "prediction": "Valorant",
  "confidences": [["Valorant", 100.0]],
  "frames": [
    "/api/frames/98a0683c-.../frame_0003_medium.jpg",
    "/api/frames/98a0683c-.../frame_0010_medium.jpg",
    "/api/frames/98a0683c-.../frame_0028_medium.jpg"
  ],
  "time_taken": 11.4,
  "created_datetime": "2026-05-24T14:11:15"
}
```

---

#### `GET /api/results/all`

Retrieve all stored detection results, sorted by most recent.

---

#### `GET /api/frames/{id}/{filename}`

Retrieve an extracted frame image by detection ID and filename. Used by the frontend to display influential frames. Frames are cached on the server for 2 hours before cleanup.

---

## Requirements

- Python 3.12
- Node.js 18+
- torch
- open-clip-torch
- opencv-python
- Pillow
- fastapi
- uvicorn
- python-multipart
- sqlmodel

Install Python dependencies:

```bash
pip install -r requirements.txt
```
---

## Future Plans

- [ ] Deduplication — reuse results for previously seen clips
- [ ] Event detection within clips (kills, deaths, aces, explosions, etc.)
- [ ] Expand shipped reference library with more games
