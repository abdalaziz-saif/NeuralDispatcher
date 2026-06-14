# NeuralDispatcher 
### AI-Powered Request Router | RoBERTa | PyTorch | FastAPI | Docker

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)
![Docker](https://img.shields.io/badge/Docker-neuraldispatcher-blue)

---

## What Is NeuralDispatcher?

Every AI system that handles multiple tasks faces the same problem — how does it know what the user actually wants? Instead of guessing, NeuralDispatcher reads any user request and automatically routes it to the right category. It's the brain that sits in front of a multi-agent system and decides who handles what.

Built on a fine-tuned RoBERTa model trained on the Databricks Dolly 15K dataset, served through a FastAPI REST API, and fully containerized with Docker.

---

## Project Structure

```
NeuralDispatcher/
├── main.py                  <- end-to-end ML pipeline
├── config.py                
├── requirements.txt
├── Dockerfile               <- containerizes the FastAPI app
│
├── api/
│   ├── __init__.py
│   └── app.py               <- FastAPI app
│
├── data/
│   ├── dataset.py           <- PyTorch Dataset class
│   └── preprocessing.py     <- data loading, spliting, dataloaders
│
├── model/
│   ├── model.py             <- fine-tune, save, load saved
│   └── trainer.py             training and evaluation Function
│
├── inference/
│   └── dispatcher.py        <- dispatch function
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            <- logging setup
│   ├── metrics.py           <- classification report, confusion matrix
│   ├── exceptions.py        <- custom exceptions
│   └── confusion_matrix.png 
│
└── saved_model/             <- fine-tuned model weights
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── label_map.json
```

---

## How It Works

```
User Input
    ↓
RoBERTa Tokenizer
    ↓
Fine-tuned RoBERTa Model
    ↓
Predicted Category
    ↓
Right Agent Handles It
```

The model classifies any input into one of 8 categories:

| Category | Example Input |
|---|---|
| `open_qa` | "What is the capital of France?" |
| `closed_qa` | "Based on this passage, who wrote it?" |
| `general_qa` | "How does photosynthesis work?" |
| `summarization` | "Summarize this article: ..." |
| `creative_writing` | "Write me a poem about the ocean" |
| `brainstorming` | "Give me startup ideas for 2025" |
| `classification` | "Is this email spam or not?" |
| `information_extraction` | "Extract the dates from this document" |

---

## Technical Overview

### Data & Tokenization
Trained on 15,000 real human instructions from the Databricks Dolly 15K dataset. Each instruction is tokenized using RoBERTa's tokenizer with dynamic padding .

The dataset is imbalanced — some categories appear 3x more than others. This is handled by computing class weights automatically and passing them into the loss function, forcing the model to treat every category equally.

### Fine-Tuning Strategy
Only the last 6 RoBERTa layers and the classification head are trained — the rest stays frozen. This adapts the model to the new task without retraining everything from scratch.

Training runs with mixed precision, a warmup scheduler, and separate learning rates for the backbone and classifier.

### Production-Ready Code
The project is split into clear modules (data, model, inference, API). Logging and custom exceptions are used throughout for easier debugging.

---

## Setup

### Prerequisites
- Python 3.10+
- Docker (optional)

### Install

```bash
git clone https://github.com/yourprofile/NeuralDispatcher.git
cd NeuralDispatcher
pip install -r requirements.txt
```

### Train

```bash
python main.py
```

This downloads the dataset, trains for 6 epochs, evaluates, prints a classification report, saves the model to `saved_model/`, and generates a confusion matrix at `utils/confusion_matrix.png`.

---

## API

### Run Locally

```bash
uvicorn api.app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger documentation.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/dispatch` | Classify a request |

### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/dispatch" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is the capital of Sudan?"}'
```

### Example Response

```json
{
  "text": "What is the capital of Sudan?",
  "label": "open_qa"
}
```

---

## Docker

### Pull from Docker Hub

```bash
docker pull neuraldispatcher
```

### Run

```bash
docker run -p 8000:8000 -v ./saved_model:/app/saved_model --name neuraldispatcher neuraldispatcher
```

### Or build locally

```bash
docker build -t neuraldispatcher .
docker run -p 8000:8000 -v ./saved_model:/app/saved_model --name neuraldispatcher neuraldispatcher
```

### After code changes

```bash
docker stop neuraldispatcher
docker rm neuraldispatcher
docker build -t neuraldispatcher .
docker run -p 8000:8000 -v ./saved_model:/app/saved_model --name neuraldispatcher neuraldispatcher
```

### After retraining model only

```bash
docker restart neuraldispatcher
```

---

The main challenge is that `open_qa`, `closed_qa`, and `general_qa` are genuinely similar in language — a boundary that is hard even for humans to draw clearly. The confusion matrix below shows exactly where the model struggles.


---

## Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model training |
| HuggingFace Transformers | RoBERTa model and tokenizer |
| Databricks Dolly 15K | Training dataset |
| FastAPI | REST API |
| Pydantic | Request/response validation |
| Docker | Containerization |
| Scikit-learn | Evaluation metrics |

---

## License

MIT License — free to use, modify, and distribute.
