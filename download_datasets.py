import kagglehub


def download_dataset(handle: str) -> None:
    """Downloads a dataset from Kaggle.

    Args:
        handle: The Kaggle dataset handle
    """
    try:
        local_dir = kagglehub.dataset_download(handle)
        print(f"Downloaded dataset {handle} to {local_dir}")
    except ValueError:
        local_dir = kagglehub.competition_download(handle)
        print(f"Downloaded dataset {handle} to {local_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


# download competition dataset
download_dataset("make-data-count-finding-data-references")

# download datasets for fine-tuning
handles = [
    "conjuring92/mdc-sgkf-folds",
    "conjuring92/mdc-synthetic-mix-v6",
    "conjuring92/mdc-type-mix-v6-ff",
    "conjuring92/mdc-type-synthetic-v1",
    "conjuring92/mdc-type-mix-v4-72b",
    "conjuring92/mdc-filter-mix-v3-pl-ff",
    "conjuring92/mdc-synthetic-input-articles-v1",
]

for handle in handles:
    download_dataset(handle)
