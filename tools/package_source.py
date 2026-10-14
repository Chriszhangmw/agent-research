"""Create a ZIP using only the checked source manifest; never recurse over data."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from check_upload import ROOT, check_files, source_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "agent-research-source.zip")
    args = parser.parse_args()
    entries = source_paths()
    problems = check_files(entries)
    if problems:
        raise SystemExit("Refusing to package:\n" + "\n".join(problems))
    output = args.out.resolve()
    if output in [(ROOT / name).resolve() for name in entries]:
        raise SystemExit("ZIP output cannot replace a source file")
    if output.exists():
        raise SystemExit("Refusing to overwrite an existing ZIP; choose a new --out path")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name in entries:
            archive.write(ROOT / name, "agent-research/" + name)
    with ZipFile(output) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {"agent-research/" + name for name in entries}:
            raise SystemExit("ZIP verification failed")
    print(f"Packaged {len(entries)} checked source files: {output.name}")


if __name__ == "__main__":
    main()
