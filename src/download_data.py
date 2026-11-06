"""Step 1 - download the Kaggle dataset and remember where it was saved."""
import kagglehub

import config


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    config.RAW_PATH_FILE.write_text(str(path), encoding="utf-8")
    print("Dataset downloaded to:", path)


if __name__ == "__main__":
    main()
