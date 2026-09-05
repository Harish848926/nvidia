# Plant Disease Detection & AI Agronomy Advisory (NVIDIA DGX Platform)

An end-to-end deep learning and Retrieval-Augmented Generation (RAG) platform for precision crop foliage pathology, disease classification, and verified treatment recommendations.

---

## Project Structure

```text
plant-disease/
│
├── app.py                 # Main application (Streamlit Web Dashboard + Q&A)
├── train.py               # Train model (DGX CUDA / AMP acceleration)
├── predict.py             # Predict disease from image + RAG advice
├── rag.py                 # RAG knowledge indexing and retrieval engine
├── evaluate.py            # Model evaluation (Accuracy, Top-3, Confusion Matrix)
│
├── data/
│   └── plantvillage/      # Dataset (Train / Val image splits)
│
├── model/
│   └── plant_model.pth    # Trained PyTorch model checkpoint
│
├── documents/             # Agronomy Knowledge Base (Reference Manuals)
│   ├── tomato.txt         # Tomato pathology (Early Blight, Late Blight, Bacterial Spot)
│   ├── rice.txt           # Rice pathology (Blast, Bacterial Blight, Brown Spot)
│   └── potato.txt         # Potato pathology (Early Blight, Late Blight, Black Scurf)
│
├── requirements.txt       # Python dependencies
├── .env                   # Configuration & environment variables
└── README.md              # Project documentation
```

---

## Key Features

1. **Computer Vision Disease Classification**:
   - Neural network backbone with custom classification head fine-tuned for crop leaves.
   - Diagnoses Tomato, Potato, and Rice foliage diseases with confidence scoring and Top-3 differential predictions.
2. **NVIDIA DGX Supercomputer Optimizations**:
   - Automatic Mixed Precision (`torch.amp.autocast`) leveraging NVIDIA Tensor Cores.
   - Multi-worker data loading with `pin_memory=True`.
   - Automatic fallback to CPU if running on non-GPU workstations.
3. **Retrieval-Augmented Generation (RAG)**:
   - Indexes verified agricultural pathology texts in `documents/`.
   - Automatically retrieves exact Visual Symptoms, Organic Controls, Chemical Fungicides, Dosages, and Preventive Measures when a disease is diagnosed.
   - Provides an interactive Q&A interface for farmers to ask agronomic questions.
4. **Interactive Web App**:
   - Multi-page dashboard with drag-and-drop image diagnosis, sample leaf gallery, RAG Q&A chat, and document manual browser.

---

## Installation & Setup

### 1. Install Dependencies
```bash
cd c:/nvidia/plant-disease
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
The provided `.env` is preconfigured for automatic device detection:
```env
DEVICE=auto
MODEL_PATH=model/plant_model.pth
DATA_PATH=data/plantvillage
DOCUMENTS_PATH=documents
APP_PORT=8501
```

---

## Usage

### 1. Launch the Main Application
Run the interactive dashboard:
```bash
python app.py
```
*(Or directly via Streamlit: `streamlit run app.py`)*

Open your browser at **http://localhost:8501**.

### 2. Predict Disease from a Leaf Image (CLI)
To run inference directly on a leaf photo:
```bash
python predict.py --image path/to/leaf.jpg
```
Output includes predicted crop, disease name, confidence score, inference latency, and RAG-retrieved symptoms and treatments.

### 3. Download Real Datasets from Hugging Face
You can stream and download full or partitioned datasets directly from Hugging Face Hub:

```bash
# List recommended Hugging Face datasets
python hf_dataset.py --list-popular

# Download PlantVillage dataset from Hugging Face
python hf_dataset.py --repo faisal-hugging-face/plant-disease --max-per-class 100

# Or train directly from Hugging Face with one command:
python train.py --hf-dataset faisal-hugging-face/plant-disease --epochs 15
```

**Recommended Hugging Face Datasets:**
- `faisal-hugging-face/plant-disease`: Full PlantVillage benchmark
- `flwrlabs/plant-village`: Curated PlantVillage partition
- `beans`: Official lightweight Hugging Face leaf disease dataset
- `ayerr/plant-disease-classification`: Multi-crop high-res disease imagery

### 4. Train the Model on NVIDIA DGX Server
To train or fine-tune the vision model with mixed precision:
```bash
python train.py --epochs 15 --batch-size 32 --lr 0.001
```
The best checkpoint is automatically saved to `model/plant_model.pth`.

### 5. Evaluate the Model
To compute validation accuracy, top-3 accuracy, F1-scores, and generate a classification report:
```bash
python evaluate.py
```
Performance metrics are saved to `model/eval_report.txt`.

### 6. Test RAG Knowledge Engine (CLI)
You can directly query the agronomy manual index:
```bash
python rag.py
```

---

## Supported Crops & Diseases

- **Tomato**: Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*), Bacterial Spot (*Xanthomonas*), Leaf Mold (*Passalora fulva*), Healthy Foliage.
- **Potato**: Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*), Healthy Foliage.
- **Rice**: Leaf Blast (*Magnaporthe oryzae*), Bacterial Leaf Blight (*Xanthomonas oryzae*), Brown Spot (*Bipolaris oryzae*), Healthy Paddy.
