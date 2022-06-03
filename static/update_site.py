"""Update the site by importing content from the folder."""
import subprocess
import logging
import argparse
import subprocess
from datetime import datetime
import os


def py_str(cstr):
    return cstr.decode("utf-8")


# list of files to skip
skip_list = set(
    [
        "wheels.html",
        ".gitignore",
        ".nojekyll",
        "CNAME",
    ]
)


def main():
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Deploy a built html to the root.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--site-path", type=str, default="mlc-ai.github.io")
    parser.add_argument("--source-path", type=str, default="mlc-en/_build/html")

    args = parser.parse_args()

    def run_cmd(cmd):
        proc = subprocess.Popen(
            cmd, cwd=args.site_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "cmd error: %s" % cmd
            msg += py_str(out)
            raise RuntimeError(msg)
        return py_str(out)

    run_cmd(["git", "fetch"])
    run_cmd(["git", "checkout", "-B", "main", "origin/main"])
    files = run_cmd(["git", "ls-files"])
    skip_set = set(skip_list)

    for fname in files.split("\n"):
        fname = fname.strip()
        if fname and fname not in skip_set:
            if not args.dry_run:
                run_cmd(["rm", "-rf", fname])
            print("Remove %s" % fname)

    if not args.dry_run:
        os.system("cp -rf %s/* %s" % (args.source_path, args.site_path))
        run_cmd(["git", "add", "--all"])

    if not args.dry_run:
        try:
            run_cmd(["git", "commit", "-am", " Update at %s" % datetime.now()])
        except RuntimeError:
            pass
        run_cmd(["git", "push", "origin", "main"])
        print("Finish updating and push to origin/main ...")


if __name__ == "__main__":
    main()
