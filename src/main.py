from .config import ConfigManager
ConfigManager.set_config_path("default.conf")

import click
from .cmds import (train_command, analyze_command, 
    cbb_command, ibb_command, kunit_command)

@click.group()
@click.version_option("0.0.1")
@click.pass_context
def cli(ctx):
    pass

@click.command("train")
@click.option("-f", "--fallback", is_flag=True, help="Use archive link for dataset")
@click.pass_context
def train(ctx, fallback: bool):
    """
    download dataset and train the ResNet model with hyperparameters matching the one from the paper
    """
    train_command(fallback)

@click.command("analyze")
@click.argument("filename")
@click.argument("dest")
@click.pass_context
def analyze(ctx, filename: str, dest: str):
    """
    extract keystroke dynamics from existing video file
    """
    analyze_command(filename, dest)

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
@click.pass_context
def kunit(ctx, filename: str, dest: str, convexity: bool):
    """
    detect rightmost character in each isolation bounding box and save to a directory dest
    """
    kunit_command(filename, dest, convexity)

cli.add_command(train)
cli.add_command(analyze)
cli.add_command(cbb)
cli.add_command(ibb)
cli.add_command(kunit)
