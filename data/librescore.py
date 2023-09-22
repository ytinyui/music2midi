import argparse
import os
import time
from pathlib import Path
from subprocess import PIPE, Popen

"""
!Important: Use subprocess.run() to run this in Jupyter Notebook.
"""


def run(url, output_dir: str = ".") -> int:
    os.makedirs(output_dir, exist_ok=True)
    # Open a subprocess to run the command
    process = Popen(
        ["npx", "dl-librescore@latest"],
        stdin=PIPE,
        stdout=PIPE,
        text=True,
        cwd=output_dir,
    )
    process.stdin.write(url + "\n")
    process.stdin.flush()
    # Continue
    for line in iter(process.stdout.readline, ""):
        print(line.strip())
        if "Title:" in line:
            # Confirm title
            process.stdin.write("\n")
            process.stdin.flush()

        if "Filetype Selection" in line:
            # Select midi
            process.stdin.write(" \b\n")
            process.stdin.flush()

        if "Done" in line:
            return process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", default=None)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    run(args.url, args.output_dir)
