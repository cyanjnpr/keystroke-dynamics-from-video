from resnet import train
import os
import tarfile
import urllib3
import tempfile

# this dataset comes from the following paper:
#
# T. E. de Campos, B. R. Babu and M. Varma. 
# Character recognition in natural images. 
# In Proceedings of the International Conference on Computer Vision Theory and 
# Applications (VISAPP), Lisbon, Portugal, February 2009. 
#
DATASET_URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
# FALLBACK_DATASET_URL = "https://web.archive.org/web/20240711155040if_/https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"

def download_dataset(dataset_dir: str) -> bool:
    success = False
    fd, filename = tempfile.mkstemp()
    print("Downloading the dataset...")
    with urllib3.PoolManager() as http:
        with http.request("GET", DATASET_URL, preload_content=False, decode_content=False) as r:
            if (r.status == 200):
                with open(fd, 'wb') as handle:
                    for chunk in r.stream():
                        handle.write(chunk)
                    success = True
    if success:
        print("Downloaded the dataset")
        print("Extracting...")
        archive = tarfile.open(filename)
        archive.extractall(dataset_dir, filter='data')
        archive.close()
    return success

def dataset_check(dataset_dir: str) -> bool:
    contents = os.listdir(dataset_dir)
    contents = [p for p in contents if not p.startswith(".")]
    if len(contents) == 0:
        print("Dataset not found")
        return download_dataset(dataset_dir)
    return True

if __name__ == "__main__":
    models_path = "models"
    dataset_path = "dataset"
    if dataset_check(dataset_path):
        print("Training...")
        dataset_path = os.path.join(dataset_path, "English", "Fnt")
        train(dataset_path, models_path)
    else:
        print("Failed to download or extract the dataset")

