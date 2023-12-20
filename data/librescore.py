import argparse
import os
import subprocess


def main(url: str, output_dir: str = ".") -> int:
    os.makedirs(output_dir, exist_ok=True)
    # open a subprocess to run the command
    process = subprocess.Popen(
        ["npx", "dl-librescore@latest"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=output_dir,
    )
    process.stdin.write(url + "\n")
    process.stdin.flush()
    for line in iter(process.stdout.readline, ""):
        print(line.strip())
        if "Title:" in line:  # confirm title
            process.stdin.write("\n")
            process.stdin.flush()

        if "Filetype Selection" in line:  # select midi
            process.stdin.write(" \b\n")
            process.stdin.flush()

        if "Output Directory:" in line:  # success
            return process.wait(5)
    raise FileNotFoundError("Score not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", default=None)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()
    main(args.url, args.output_dir)
