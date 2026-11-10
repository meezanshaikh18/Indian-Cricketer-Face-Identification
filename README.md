# Indian Cricketer Face Identification

A computer vision and machine learning project that identifies a cricket player from an uploaded face image. The pipeline detects and crops a face, extracts a 512-dimensional FaceNet embedding, and uses a trained classifier to predict one of the players represented in the training dataset. If prediction confidence is below a configurable threshold, the application returns **Unknown**.

The project includes a Streamlit web application, command-line prediction, classifier comparison, a PCA + SVM baseline, and evaluation reports.

## Table of contents

- [Project overview](#project-overview)
- [How it works](#how-it-works)
- [Technology stack](#technology-stack)
- [Dataset](#dataset)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Run the pipeline](#run-the-pipeline)
- [Launch the application](#launch-the-application)
- [Command-line prediction](#command-line-prediction)
- [Configuration](#configuration)
- [Evaluation results](#evaluation-results)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Project overview

The application is designed to:

- Detect a face in an input image and create a standardized face crop.
- Extract facial features using the pretrained FaceNet `InceptionResnetV1` model.
- Train and compare SVM, Logistic Regression, and KNN classifiers using the embeddings.
- Compare the embedding-based approach with a PCA + SVM baseline using grayscale image pixels.
- Display the predicted player, confidence, and alternative matches.
- Return **Unknown** when prediction confidence is below the configured threshold.

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
| Dataset download | KaggleHub | Downloads the Indian Cricketers dataset |
| Dataset inspection | Custom Python utilities | Discovers image files and player labels |
| Face detection | MTCNN via `facenet-pytorch` | Detects and crops faces |
| Feature extraction | FaceNet `InceptionResnetV1`, pretrained on VGGFace2 | Creates 512-dimensional embeddings |
| Classifier training | SVM, Logistic Regression, and KNN | Trains and compares classifiers |
| Baseline | Grayscale pixels, PCA, and SVM | Provides a traditional computer-vision baseline |
| Inference | Shared prediction pipeline | Produces a prediction, confidence, and top matches |
| User interface | Streamlit | Allows image upload and displays results |

The SVM and Logistic Regression candidates are tuned using cross-validation. KNN is included in the comparison, but is not selected for deployment because its probability estimates are not suitable for the project's confidence-threshold-based **Unknown** handling.

## Technology stack

- **Language:** Python
- **Face detection and embeddings:** MTCNN, FaceNet, `facenet-pytorch`
- **Machine learning:** scikit-learn
- **Data processing:** NumPy, pandas
- **Image processing:** Pillow
- **Visualization:** Matplotlib
- **Web application:** Streamlit
- **Dataset download:** KaggleHub
- **Testing:** pytest

## Dataset

This project uses the [Indian Cricketers Dataset on Kaggle](https://www.kaggle.com/datasets/aviratgupta/indian-cricketers-dataset).

- **Kaggle dataset identifier:** `aviratgupta/indian-cricketers-dataset`
- The dataset is downloaded locally by `src/download_data.py` through KaggleHub.
- Dataset images are not included in this repository.

Review the dataset's Kaggle page and follow its terms and usage conditions.

## Project structure

```text
cricketer-face-id/
├── app.py
├── run_all.py
├── check_setup.py
├── config.py
├── conftest.py
├── requirements.txt
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── inspect_dataset.py
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── train.py
│   ├── train_baseline.py
│   ├── face_embedder.py
│   ├── inference.py
│   ├── predict.py
│   └── data_utils.py
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── reports/
    └── .gitkeep
```

Dataset files, processed images, embeddings, trained models, and generated reports are created locally and excluded from Git. The `.gitkeep` files preserve the empty directories in the repository.

## Getting started

### Requirements

- 64-bit Windows, macOS, or Linux.
- Python **3.10, 3.11, or 3.12**. Python 3.12 is the tested setup for this project.
- Internet connection for downloading the dataset and pretrained FaceNet weights.
- CPU is sufficient; a compatible GPU is optional.

### 1. Clone the repository

```bash
git clone https://github.com/meezanshaikh18/indian-cricketer-face-identification.git
cd indian-cricketer-face-identification
```

### 2. Create an environment

#### Windows with Anaconda / Miniconda

```powershell
conda create -n cricket-faceid python=3.12 -y
conda activate cricket-faceid
```

If the environment already exists, activate it instead:

```powershell
conda activate cricket-faceid
```

#### Windows without Conda

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Run these commands after activating the environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**PyArrow compatibility note:** If you use the NumPy 1.x dependency set in this project and encounter an import error stating that PyArrow requires NumPy 2 or later, install the compatible PyArrow version:

```bash
python -m pip install --force-reinstall --no-deps pyarrow==19.0.1
```

### 4. Verify the setup

```bash
python check_setup.py
```

This checks the Python environment and required packages and initializes the face-model setup. The first run may download pretrained model weights, so an internet connection is required.

## Run the pipeline

### Option A: Run all stages

From the project root:

```bash
python run_all.py
```

The complete pipeline:

1. Downloads the dataset.
2. Inspects the dataset structure.
3. Detects and crops faces.
4. Generates FaceNet embeddings.
5. Trains and evaluates embedding-based classifiers.
6. Trains and evaluates the PCA + SVM baseline.

### Option B: Skip dataset download

If the dataset has already been downloaded and is available at the path expected by the project:

```bash
python run_all.py --skip-download
```

This skips **only** the dataset-download stage. It still runs inspection, preprocessing, embedding generation, and training.

### Option C: Run stages individually

Run from the project root, in order:

| Step | Command | Purpose |
|---|---|---|
| 1 | `python -m src.download_data` | Download the Kaggle dataset |
| 2 | `python -m src.inspect_dataset` | Inspect folders, labels, and image counts |
| 3 | `python -m src.preprocess` | Remove exact duplicates, detect faces, and save face crops |
| 4 | `python -m src.embeddings` | Generate FaceNet embeddings and handle near duplicates |
| 5 | `python -m src.train` | Train, tune, compare, and evaluate classifiers |
| 6 | `python -m src.train_baseline` | Train and evaluate the PCA + SVM baseline |

For a limited preprocessing check:

```bash
python -m src.preprocess --limit 10
```

This processes a limited number of images per player for a quick face-detection check. It does not replace a complete training run.

## Launch the application

After the training pipeline has successfully created the model artifacts:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, which you can open in your browser.

Upload a supported image (`JPG`, `JPEG`, `PNG`, or `WEBP`) to view the prediction. The interface displays the prediction result, confidence, confidence-threshold setting, known player classes, and alternative matches when available.

If the model has not been trained, the application will report that the trained model file is missing. Run the pipeline first.

## Command-line prediction

After training, pass an image path to the prediction script:

```bash
python -m src.predict path/to/photo.jpg
```

To override the confidence threshold:

```bash
python -m src.predict path/to/photo.jpg --threshold 0.60
```

The command reports the predicted label, confidence, threshold, and top matches. If no clear face is detected, it reports that no face was found. If confidence is below the threshold, the result is **Unknown**.

## Configuration

Project settings are centralized in `config.py`.

| Setting | Default | Description |
|---|---:|---|
| `KAGGLE_DATASET` | `aviratgupta/indian-cricketers-dataset` | Kaggle dataset identifier |
| `FACE_SIZE` | `160` | Face crop size expected by FaceNet |
| `FACE_MARGIN` | `20` | Extra margin around the detected face |
| `MIN_FACE_SIZE` | `40` | Minimum accepted face size in pixels |
| `MIN_DETECTION_PROB` | `0.90` | Minimum face-detection confidence |
| `MAX_IMAGE_SIDE` | `1280` | Maximum image side length before detection |
| `MIN_IMAGES_PER_CLASS` | `20` | Minimum usable face images required per player |
| `MAX_CLASSES` | `None` | Optional limit on the number of player classes |
| `TEST_SIZE` | `0.2` | Proportion reserved for the test split |
| `SEED` | `42` | Random seed used by training utilities |
| `DEDUP_COSINE` | `0.98` | Cosine-similarity threshold for near-duplicate handling |
| `UNKNOWN_THRESHOLD` | `0.50` | Confidence below which a prediction is returned as Unknown |
| `BASELINE_IMAGE_SIZE` | `64` | Image size used by the PCA + SVM baseline |

Change settings in `config.py` rather than editing individual pipeline scripts. For example, set `MAX_CLASSES` to `8` to limit the number of player classes. The default `None` does not impose this limit.

## Evaluation results

The following results are from one completed run using the project's configured 20-player selection and the downloaded Kaggle dataset. Results can vary with dataset contents, preprocessing, configuration, and random split.

| Metric | Result |
|---|---:|
| Selected embedding model | SVM |
| Players evaluated | 20 |
| Face crops saved | 476 |
| Embeddings saved after near-duplicate handling | 462 |
| SVM test accuracy | 93.5% |
| SVM macro F1-score | 0.934 |
| Logistic Regression test accuracy | 94.6% |
| KNN test accuracy | 94.6% |
| PCA + SVM baseline test accuracy | 31.2% |

The SVM was selected by the project's model-selection procedure for deployment. The classifier comparison and baseline are reported separately; the highest individual test accuracy in this run was not the deployed model.

Generated reports are saved in `reports/` and include:

| File | Description |
|---|---|
| `metrics.json` | Summary metrics for the selected embedding-based model |
| `classification_report.txt` | Per-class precision, recall, and F1-score |
| `confusion_matrix.png` | Confusion matrix for the embedding-based classifier |
| `threshold_table.csv` | Coverage and accuracy at different confidence thresholds |
| `misclassified.csv` | Test examples incorrectly classified by the selected model |
| `class_distribution.png` | Class distribution visualization |
| `preprocess_summary.csv` | Per-player preprocessing counts |
| `model_comparison.csv` | Comparison of embedding-based classifiers and PCA + SVM |
| `baseline_confusion_matrix.png` | Confusion matrix for the PCA + SVM baseline |
| `eigenfaces.png` | Principal-component visualization for the baseline |

Review the generated reports before quoting performance values in a portfolio, report, or internship submission.

## Tests

Run the automated tests from the project root with the environment activated:

```bash
python -m pytest -q
```

The current tested environment completed **9 tests successfully**. Tests validate code behavior using test data; they do not independently establish model accuracy on the full dataset.

## Troubleshooting

### The trained model file is missing

Run the complete pipeline before starting Streamlit:

```bash
python run_all.py
```

### The dataset is not found

Run the dataset download stage:

```bash
python -m src.download_data
```

### No labelled images are found

Inspect the dataset structure:

```bash
python -m src.inspect_dataset
```

The project derives labels from discovered dataset paths. An unexpected folder layout may require changes to the dataset discovery utilities.

### No clear face is detected

Try a sharp, front-facing image with a clearly visible face. Small, blurry, side-facing, or obstructed faces may not be detected.

### Dependency installation or import errors

Confirm that the intended environment is active and that Python is 3.10, 3.11, or 3.12:

```bash
python --version
python -m pip --version
python -m pip check
```

If you use the NumPy 1.x dependency set and see a PyArrow/NumPy compatibility error, apply the PyArrow compatibility command in the dependency-installation section.

### Predictions are frequently returned as Unknown

Review the confidence threshold in `config.py` or the Streamlit sidebar. A lower threshold may increase accepted predictions but can also allow more incorrect predictions. Use `threshold_table.csv` to assess the coverage-versus-accuracy trade-off.

### Some player classes have weak results

Review `preprocess_summary.csv`, `misclassified.csv`, and `classification_report.txt`. The number and quality of usable face images can affect per-player performance.

## Limitations and responsible use

- This is an educational project using a dataset of public cricket figures.
- The model only recognizes classes represented in its training data.
- Prediction quality depends on image quality, angle, lighting, and face visibility.
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

## Connect

I'm always open to connecting with fellow developers, learning new technologies, and exploring opportunities in AI/ML.

- **LinkedIn:** [Meezan Shaikh](https://www.linkedin.com/in/meezanshaikh18/)
- **GitHub:** [meezanshaikh18](https://github.com/meezanshaikh18)

If you find this project interesting, consider giving the repository a star.
