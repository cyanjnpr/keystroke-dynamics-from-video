from ..resnet import train
import os
import tarfile
import urllib3
import tempfile
import click
from pathlib import Path

# this dataset comes from the following paper:
#
# T. E. de Campos, B. R. Babu and M. Varma. 
# Character recognition in natural images. 
# In Proceedings of the International Conference on Computer Vision Theory and 
# Applications (VISAPP), Lisbon, Portugal, February 2009. 
#
DATASET_URL = "https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"
FALLBACK_DATASET_URL = "https://web.archive.org/web/20240711155040if_/https://info-ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz"

def download_dataset(dataset_dir: str, fallback: bool) -> bool:
    ur = FALLBACK_DATASET_URL if fallback else DATASET_URL
    success = False
    fd, filename = tempfile.mkstemp()
    click.echo("Downloading the dataset...")
    with urllib3.PoolManager() as http:
        with http.request("GET", ur, preload_content=False, decode_content=False) as r:
            if (r.status == 200):
                with open(fd, 'wb') as handle:
                    for chunk in r.stream():
                        handle.write(chunk)
                    success = True
    if success:
        click.echo("Downloaded the dataset")
        click.echo("Extracting...")
        archive = tarfile.open(filename)
        archive.extractall(dataset_dir, filter='data')
        archive.close()
    return success

def dataset_check(dataset_dir: str, fallback: bool) -> bool:
    contents = os.listdir(dataset_dir)
    contents = [p for p in contents if not p.startswith(".")]
    if len(contents) == 0:
        click.echo("Dataset not found")
        return download_dataset(dataset_dir, fallback)
    return True

def train_command(fallback: bool, dataset_path: str, models_path: str):
    models_path = Path(models_path)
    dataset_path = Path(dataset_path)
    if dataset_check(str(dataset_path), fallback):
        click.echo("Training...")
        dataset_path = dataset_path / "English" / "Fnt"
        train(str(dataset_path), str(models_path))
    else:
        click.echo("Failed to download or extract the dataset")

