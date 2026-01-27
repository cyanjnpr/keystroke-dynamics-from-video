from .config import ConfigManager
ConfigManager.set_config_path("default.conf")

import click
from .cmds import (train_command, analyze_command, 
    cbb_command, ibb_command, kunit_command,
    capture_command)

@click.group()
@click.version_option("0.0.1")
@click.pass_context
def cli(ctx):
    pass

@click.command("train")
@click.option("-f", "--fallback", is_flag=True, help="Use the archive link for dataset")
@click.option("-d", "--dataset", type=str, help="Where to save the dataset", default="dataset")
@click.option("-m", "--model", type=str, help="Where to save the model", default="models")
@click.pass_context
def train(ctx, fallback: bool, dataset: str, model: str):
    """
    download dataset and train the ResNet model with hyperparameters matching the one from the paper
    """
    train_command(fallback, dataset, model)

@click.command("analyze")
@click.argument("filename")
@click.argument("dest")
@click.option("-m", "--model", type=str, help="path to model file or models directory", default="models")
@click.pass_context
def analyze(ctx, filename: str, dest: str, model: str):
    """
    extract keystroke dynamics from existing video file
    """
    analyze_command(filename, dest, model)

@click.command("cbb")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def cbb(ctx, filename: str, dest: str):
    """
    detect cursor bounding box in each frame and save to a directory dest
    """
    cbb_command(filename, dest)

@click.command("ibb")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def ibb(ctx, filename: str, dest: str):
    """
    detect isolation bounding box in each frame and save to a directory dest
    """
    ibb_command(filename, dest)

@click.command("kunit")
@click.argument("filename")
@click.argument("dest")
@click.option("-c", "--convexity", is_flag=True, help="Draw convexity of the character")
@click.option("-p", "--predictions", is_flag=True, help="OCR KUnit images")
@click.option("-m", "--model", type=str, help="path to model file or models directory", default="models")
@click.pass_context
def kunit(ctx, filename: str, dest: str, convexity: bool, predictions: bool, model: str):
    """
    detect rightmost character in each isolation bounding box and save to a directory dest
    """
    kunit_command(filename, dest, convexity, predictions, model)

@click.command("capture")
@click.argument("dest")
@click.pass_context
def capture(ctx, dest: str):
    """
    keylogger for capturing real keystroke dynamics
    """
    capture_command(dest)

cli.add_command(train)
cli.add_command(analyze)
cli.add_command(cbb)
cli.add_command(ibb)
cli.add_command(kunit)
cli.add_command(capture)
