# Indian Cricketer Face Identification

A computer vision and machine learning project that identifies a cricket player from an uploaded face image. The pipeline detects and crops a face, extracts a 512-dimensional FaceNet embedding, and uses a trained classifier to predict one of the players included in the training dataset. If the prediction confidence is below a configurable threshold, the application returns **Unknown**.

The project includes a Streamlit web interface, command-line prediction, model comparison, a PCA + SVM baseline, and evaluation reports.

## Table of contents

- [Project overview](#project-overview)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Run the pipeline](#run-the-pipeline)
- [Launch the Streamlit application](#launch-the-streamlit-application)
- [Make a command-line prediction](#make-a-command-line-prediction)
- [Configuration](#configuration)
- [Evaluation outputs](#evaluation-outputs)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Project overview

The application is designed to:

- Detect a face in an input image and create a standardized face crop.
- Extract facial features using the pretrained FaceNet `InceptionResnetV1` model.
- Train and compare SVM, Logistic Regression, and KNN classifiers using the extracted embeddings.
- Compare the embedding-based approach with a traditional PCA + SVM baseline using grayscale image pixels.
- Display the predicted player, confidence, and alternative matches.
- Return **Unknown** when the prediction confidence is below the configured threshold.

> The model can only predict players represented in its training data. It is not a general-purpose face recognition system.

## How it works

```text
Input image
    |
    v
MTCNN face detection
    |
    v
160 x 160 face crop
    |
    v
FaceNet (InceptionResnetV1)
    |
    v
512-dimensional face embedding
    |
    v
Trained classifier
    |
    +---- Confidence >= threshold ----> Predicted player
    |
    +---- Confidence < threshold -----> Unknown
```

### Pipeline stages

| Stage | Implementation | Purpose |
|---|---|---|
| Dataset download | KaggleHub | Downloads the Indian cricketers dataset |
| Dataset inspection | Custom Python utilities | Discovers image files and their player labels |
| Face detection | MTCNN via `facenet-pytorch` | Detects and crops faces |
| Feature extraction | FaceNet `InceptionResnetV1`, pretrained on VGGFace2 | Converts each face into a 512-dimensional embedding |
| Classifier training | SVM, Logistic Regression, and KNN | Trains and compares classifiers using the embeddings |
| Baseline | Grayscale pixels, PCA, and SVM | Provides a traditional computer-vision baseline |
| Inference | Shared prediction pipeline | Produces a prediction, confidence, and top matches |
| User interface | Streamlit | Allows users to upload an image and view the result |

The SVM and Logistic Regression candidates are tuned using cross-validation. KNN is included in the comparison, but is not selected for deployment because its probability estimates are not suitable for the project's confidence-threshold-based **Unknown** handling.

## Technology stack

- **Language:** Python
- **Face detection and embeddings:** MTCNN, FaceNet, facenet-pytorch
- **Machine learning:** scikit-learn
- **Data processing:** NumPy, pandas
- **Image processing:** Pillow
- **Visualization:** Matplotlib
- **Web application:** Streamlit
- **Dataset download:** KaggleHub
- **Testing:** pytest

## Dataset

This project uses the **Indian Cricketers Dataset** published on Kaggle:

- Dataset: [Indian Cricketers Dataset](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset)
- Kaggle dataset identifier used by the project: `aviratgupta/indian-cricketers-dataset`

The dataset is downloaded locally by `src/download_data.py` through KaggleHub. It is not included in this repository. Please review the dataset's Kaggle page and follow its terms and usage conditions.

## Project structure

```text
cricketer-face-id/
|
|-- app.py                    # Streamlit application
|-- run_all.py                # Runs the complete training pipeline
|-- check_setup.py            # Checks the Python environment and dependencies
|-- config.py                 # Central project configuration
|-- conftest.py               # Pytest configuration
|-- requirements.txt          # Python dependencies
|-- .gitignore
|-- README.md
|
|-- src/
|   |-- __init__.py
|   |-- download_data.py      # Step 1: download dataset
|   |-- inspect_dataset.py    # Step 2: inspect dataset structure
|   |-- preprocess.py         # Step 3: detect and crop faces
|   |-- embeddings.py         # Step 4: extract FaceNet embeddings
|   |-- train.py              # Step 5: train and evaluate classifiers
|   |-- train_baseline.py     # Step 6: PCA + SVM baseline
|   |-- face_embedder.py      # MTCNN and FaceNet wrapper
|   |-- inference.py          # Shared prediction logic
|   |-- predict.py            # Command-line prediction
|   |-- data_utils.py         # Dataset discovery and data utilities
|
|-- tests/
|   |-- __init__.py
|   |-- test_pipeline.py      # Automated pipeline tests
|
|-- data/
|   |-- .gitkeep              # Generated dataset and intermediate data are not committed
|
|-- models/
|   |-- .gitkeep              # Trained model artifacts are not committed
|
|-- reports/
    |-- .gitkeep              # Generated evaluation reports are not committed
```

Dataset files, processed images, embeddings, trained models, and generated reports are created locally when you run the pipeline. They are excluded from Git by the repository's `.gitignore`.

## Getting started

### 1. Requirements

- Python **3.10, 3.11, or 3.12** (64-bit).
- Internet connection for downloading the dataset and pretrained FaceNet weights.
- A CPU is sufficient; a GPU is optional.

The project's current setup instructions target Python 3.10–3.12. Python 3.13 is not listed as supported for this dependency setup.

### 2. Clone the repository

Replace `<your-github-username>` with your GitHub username after publishing the repository.

```bash
git clone https://github.com/<your-github-username>/cricketer-face-id.git
cd cricketer-face-id
```

### 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Check the setup

```bash
python check_setup.py
```

This script checks the Python environment and required packages and initializes the face-model setup. The first run may download pretrained model weights and therefore requires an internet connection.

## Run the pipeline

### Option A: Run all steps

Run the following from the project root:

```bash
python run_all.py
```

This executes the complete pipeline:

1. Download the dataset.
2. Inspect the dataset structure.
3. Detect and crop faces.
4. Generate FaceNet embeddings.
5. Train and evaluate the embedding-based classifiers.
6. Train and evaluate the PCA + SVM baseline.

When the pipeline finishes, launch the application:

```bash
streamlit run app.py
```

### Option B: Skip dataset download

If the dataset has already been downloaded and is available at the path recorded by the project, use:

```bash
python run_all.py --skip-download
```

This skips only the dataset-download step and runs the remaining pipeline stages.

### Option C: Run each stage separately

Run these commands in order from the project root:

| Step | Command | Purpose |
|---|---|---|
| 1 | `python -m src.download_data` | Download the Kaggle dataset |
| 2 | `python -m src.inspect_dataset` | Inspect dataset folders, labels, and image counts |
| 3 | `python -m src.preprocess` | Remove exact duplicate files, detect faces, and save face crops |
| 4 | `python -m src.embeddings` | Generate FaceNet embeddings and handle near-duplicate embeddings |
| 5 | `python -m src.train` | Train, tune, compare, and evaluate classifiers |
| 6 | `python -m src.train_baseline` | Train and evaluate the PCA + SVM baseline |
| 7 | `streamlit run app.py` | Launch the web application |

For a quick preprocessing check, process a limited number of images per player:

```bash
python -m src.preprocess --limit 10
```

This is useful for testing the face-detection stage before processing the full dataset. It does not replace a complete training run.

## Launch the Streamlit application

After the training pipeline has successfully created the trained classifier, run:

```bash
streamlit run app.py
```

The application opens in your browser. Upload a supported image (`JPG`, `JPEG`, `PNG`, or `WEBP`) to view the prediction.

The interface displays the uploaded image and prediction result. It also provides a confidence-threshold setting, the deployed model name, the player classes known to the model, and alternative matches when available.

If the model has not been trained, the application will report that the trained model file is missing. Run the pipeline first.

## Make a command-line prediction

After training, pass an image path to the prediction script:

```bash
python -m src.predict path/to/photo.jpg
```

You can optionally override the confidence threshold:

```bash
python -m src.predict path/to/photo.jpg --threshold 0.60
```

The command-line prediction reports the predicted label, confidence, threshold, and top matches. If no clear face is detected, it reports that no face was found. If confidence is below the threshold, the result is **Unknown**.

## Configuration

Project settings are centralized in `config.py`. Change these values there rather than editing individual pipeline scripts.

| Setting | Default | Description |
|---|---:|---|
| `KAGGLE_DATASET` | `aviratgupta/indian-cricketers-dataset` | Kaggle dataset identifier |
| `FACE_SIZE` | `160` | Face crop size expected by FaceNet |
| `FACE_MARGIN` | `20` | Extra margin around the detected face |
| `MIN_FACE_SIZE` | `40` | Minimum accepted face size in pixels |
| `MIN_DETECTION_PROB` | `0.90` | Minimum face-detection confidence |
| `MAX_IMAGE_SIDE` | `1280` | Maximum image side length before detection |
| `MIN_IMAGES_PER_CLASS` | `20` | Minimum usable face images required for a player class |
| `MAX_CLASSES` | `None` | Optional limit on the number of player classes |
| `TEST_SIZE` | `0.2` | Proportion reserved for the test split |
| `SEED` | `42` | Random seed used by the training utilities |
| `DEDUP_COSINE` | `0.98` | Cosine-similarity threshold used for near-duplicate handling |
| `UNKNOWN_THRESHOLD` | `0.50` | Confidence below which a prediction is returned as Unknown |
| `BASELINE_IMAGE_SIZE` | `64` | Image size used by the PCA + SVM baseline |

To limit the number of player classes used, set `MAX_CLASSES` in `config.py` to a number such as `8`. The default `None` does not impose this class-count limit. Use the dataset inspection command to review the labels available in the downloaded dataset.

## Evaluation outputs

The training scripts create reports in the local `reports/` directory. The exact metrics depend on the dataset contents, preprocessing results, selected classes, and training run.

Important outputs include:

| File | Description |
|---|---|
| `metrics.json` | Summary metrics for the selected embedding-based model |
| `classification_report.txt` | Per-class precision, recall, and F1-score |
| `confusion_matrix.png` | Confusion matrix for the embedding-based classifier |
| `threshold_table.csv` | Coverage and accuracy at different confidence thresholds |
| `misclassified.csv` | Test examples incorrectly classified by the selected model |
| `class_distribution.png` | Class distribution visualization |
| `preprocess_summary.csv` | Per-player preprocessing counts, including detected and missed faces |
| `model_comparison.csv` | Comparison of embedding-based classifiers and the PCA + SVM baseline |
| `baseline_confusion_matrix.png` | Confusion matrix for the PCA + SVM baseline |
| `eigenfaces.png` | Visualization of principal components from the baseline |

Review the generated reports after training before describing model performance in a project report, portfolio, or internship submission. Do not quote an accuracy or other performance value unless it comes from your own completed run.

## Tests

The project includes automated tests for dataset discovery, data utilities, training, and the PCA + SVM baseline.

Install the dependencies, activate the virtual environment, and run:

```bash
python -m pytest -q
```

These tests validate code behavior using test data. Passing the automated tests does not, by itself, establish the model's accuracy on the full Kaggle dataset. Run the complete pipeline and review its evaluation reports to assess real dataset performance.

## Troubleshooting

**The trained model file is missing**

Run the complete pipeline before starting Streamlit:

```bash
python run_all.py
```

**The dataset path is unknown**

Download the dataset first:

```bash
python -m src.download_data
```

**No labelled images are found**

Run the dataset inspection step:

```bash
python -m src.inspect_dataset
```

Check the dataset's folder and image naming structure. The project derives labels from the discovered dataset paths; an unexpected layout may require changes to the dataset discovery utilities.

**No clear face is detected**

Try a sharper, front-facing image with a clearly visible face. Small, blurry, side-facing, or obstructed faces may not be detected.

**Dependency installation fails**

Confirm that you are using Python 3.10, 3.11, or 3.12 in a 64-bit environment. Upgrade pip and retry installation:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Predictions are frequently returned as Unknown**

Review the confidence threshold in `config.py` or the Streamlit sidebar. A lower threshold may increase the number of accepted predictions, but can also allow more incorrect predictions. Use `threshold_table.csv` to understand the coverage-versus-accuracy trade-off before changing the default.

**Some player classes have weak results**

Review `preprocess_summary.csv`, `misclassified.csv`, and `classification_report.txt`. Differences in the number and quality of usable face images can affect per-player performance.

## Limitations and responsible use

- This is an educational project using a dataset of public cricket figures.
- The model only recognizes classes represented in its training data.
- Prediction quality depends on the quality, angle, lighting, and visibility of the face in the input image.
- A confidence score is not a guarantee that a prediction is correct.
- The **Unknown** threshold is a practical rejection rule, not proof that an unfamiliar person has been reliably identified as unknown.
- Do not use this project to identify private individuals, monitor people, or conduct surveillance.
- Follow the dataset provider's terms and applicable laws when using or sharing dataset images.

## Acknowledgements

- [Indian Cricketers Dataset on Kaggle](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)

---

**Project:** Indian Cricketer Face Identification  
**Purpose:** Computer vision and machine learning project  
**Interface:** Streamlit


---

