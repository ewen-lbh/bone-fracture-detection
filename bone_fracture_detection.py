#!/usr/bin/env python3
from os import mkdir
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from pathlib import Path
from shutil import rmtree
from rich import print
from rich.progress import Progress


# _, low, high, = sys.argv


batchesdir = Path("edge-detection/batches")
def detect_edges(filename: str,  low: int, high: int, σ: int=3):
    σ, low, high = map(int, (σ, low, high))

    image = cv2.imread(filename, 0)
    edges = cv2.Canny(image, low, high, apertureSize=σ, L2gradient=True)
    return image, edges




tries = {"start": int(sys.argv[1]), "stop": int(sys.argv[2]), "step": int(sys.argv[3])}
total_steps = len(range(*tries.values()))**2
print(f"[red]cleaning up")
rmtree('edge-detection/batches')
mkdir('edge-detection/batches')
print(f"going with range(*{list(tries.values())})")

with Progress() as progress_bar:
    task = progress_bar.add_task("[blue]Processing...", total=total_steps)
    for high in range(*tries.values()):
        for low in range(*tries.values()):
            plt.suptitle(f"σ={3}, high={high}, low={low}, high×low={high*low}, \nfile is datasets/various/treated/*coude_clair_2.png")
            for i, filename in enumerate([ 'treated/coude_clair_2.png', 'treated/baseline_coude_clair_2.png' ]):
                _, edges = detect_edges(f"datasets/various/{filename}", low, high)
                # plt.subplot(140 + i)
                # plt.title(filename)
                # plt.imshow(img, cmap='gray')
                # i += 1
                # plt.subplot(140 + i)
                plt.subplot(121 + i)
                plt.title(f"Edges of {'healed' if 'baseline' in filename else 'broken'} bone")
                plt.imshow(edges, cmap='gray')
            out_filename = f"edge-detection/batches/{low*high:0{len(str(int(sys.argv[2])**2))}d} low={low:03d} high={high:03d}.png"
            plt.savefig(out_filename)
            progress_bar.update(task, advance=1, description=f"did [yellow]low[/]=[cyan bold]{low:03d}[/] [yellow]high[/]=[cyan bold]{high:03d}[/]")