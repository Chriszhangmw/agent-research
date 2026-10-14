"""Check the explicit source manifest and Git index without printing secrets."""
import argparse
import ast
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.txt"
TOKEN = re.compile(r"(?:sk-|gh[pousr]_|github_pat_|AIza)[A-Za-z0-9_\-]{12,}|AKIA[A-Z0-9]{16}")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
CREDENTIAL_URL = re.compile(r"https?://[^/\s:@]+:[^/\s@]+@[^\s\"'<>]+")
PERSONAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]Users[\\/]|/Users/|/home/)[A-Za-z0-9_.-]+")
CREDENTIAL_NAME = re.compile(r"^(?:key|api_key|\w+_api_key|access_token|auth_token|password|secret|\w+_secret)$", re.I)
CREDENTIAL_FIELD = re.compile(r"^(?:authorization|proxy_authorization|x_api_key|api_key|access_token|auth_token|refresh_token|password|secret|\w+_secret)$", re.I)
PLACEHOLDERS = {"", "secret", "secret2", "bearer secret", "test-key", "fake-key", "test-token", "<redacted>", "your-api-key"}
SOURCE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".ipynb", ".txt"}
SPECIAL_FILES = {".gitignore", ".gitattributes"}
CREDENTIAL_FILE = re.compile(
    r"(?:^|/)(?:\.env(?:\..*)?|.*(?:api[-_]?key|secret|credential).*)(?:$|/)", re.I
)


def source_paths():
    entries = [line.strip() for line in MANIFEST.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(entries) != len(set(entries)):
        raise ValueError("duplicate manifest entry")
    for entry in entries:
        relative = PurePosixPath(entry)
        if relative.is_absolute() or ".." in relative.parts or "\\" in entry or ":" in entry:
            raise ValueError("invalid manifest path")
        target = ROOT / entry
        if target.is_symlink() or ROOT not in target.resolve().parents or not target.is_file():
            raise ValueError(f"missing or unsafe source file: {entry}")
        if target.suffix not in SOURCE_SUFFIXES and entry not in SPECIAL_FILES:
            raise ValueError(f"unsupported source file: {entry}")
        if CREDENTIAL_FILE.search(entry):
            raise ValueError(f"credential-related file in manifest: {entry}")
        if any(re.fullmatch(r"(?:data|patients|runs.*|out.*|figdata|__pycache__|\.git)", part) for part in relative.parts):
            raise ValueError(f"artifact directory in manifest: {entry}")
    return entries


def scan_text(text, path):
    problems = []
    for label, pattern in [("credential token", TOKEN), ("private key", PRIVATE_KEY), ("personal path", PERSONAL_PATH)]:
        for match in pattern.finditer(text):
            problems.append(f"{path}:{text.count(chr(10), 0, match.start()) + 1}: {label}")
    for match in CREDENTIAL_URL.finditer(text):
        host = urlsplit(match.group()).hostname or ""
        if path.startswith("test/") and host.endswith(".test"):
            continue  # Deliberately invalid dummy URL used to test rejection.
        problems.append(f"{path}: credential embedded in URL")
    return problems


def scan_python(source, path):
    problems = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: Python syntax error"]
    for node in ast.walk(tree):
        pairs = []
        if isinstance(node, ast.Assign):
            pairs = [(target.id, node.value) for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            pairs = [(node.target.id, node.value)]
        elif isinstance(node, ast.keyword) and node.arg:
            pairs = [(node.arg, node.value)]
        elif isinstance(node, ast.Dict):
            pairs = [(key.value, value) for key, value in zip(node.keys, node.values)
                     if isinstance(key, ast.Constant) and isinstance(key.value, str)
                     and CREDENTIAL_FIELD.fullmatch(key.value.replace("-", "_"))]
        for name, value in pairs:
            if (CREDENTIAL_NAME.fullmatch(name) or CREDENTIAL_FIELD.fullmatch(name.replace("-", "_"))) and isinstance(value, ast.Constant) and isinstance(value.value, str):
                if value.value.lower() not in PLACEHOLDERS:
                    problems.append(f"{path}:{node.lineno}: credential assigned as a literal")
    return problems


def scan_config(value, path):
    problems = []
    if isinstance(value, dict):
        for key, item in value.items():
            if CREDENTIAL_FIELD.fullmatch(str(key).replace("-", "_")) and isinstance(item, str):
                if item.lower() not in PLACEHOLDERS:
                    problems.append(f"{path}: literal credential in configuration")
            problems.extend(scan_config(item, path))
    elif isinstance(value, list):
        for item in value:
            problems.extend(scan_config(item, path))
    return problems


def check_files(entries, read_bytes=None):
    read_bytes = read_bytes or (lambda name: (ROOT / name).read_bytes())
    problems = []
    for name in entries:
        try:
            text = read_bytes(name).decode("utf-8-sig")
        except UnicodeDecodeError:
            problems.append(f"{name}: unexpected binary file")
            continue
        problems.extend(scan_text(text, name))
        if name.endswith(".py"):
            problems.extend(scan_python(text, name))
        elif name.endswith((".yaml", ".yml")):
            import yaml
            problems.extend(scan_config(yaml.safe_load(text), name))
        elif name.endswith(".ipynb"):
            document = json.loads(text)
            for index, cell in enumerate(document.get("cells", [])):
                label = f"{name}:cell-{index}"
                source = "".join(cell.get("source", []))
                problems.extend(scan_text(source, label))
                if cell.get("metadata") or cell.get("attachments"):
                    problems.append(f"{label}: notebook metadata or attachments")
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    problems.append(f"{label}: notebook execution output")
                if cell.get("cell_type") == "code":
                    problems.extend(scan_python(source, label))
            if set(document.get("metadata", {})) - {"kernelspec", "language_info"}:
                problems.append(f"{name}: unexpected notebook metadata")
    return problems


def check_index(entries):
    result = subprocess.run(["git", "ls-files", "--cached", "-z"], cwd=ROOT, check=True, capture_output=True)
    tracked = set(result.stdout.decode("utf-8").split("\0")) - {""}
    problems = [f"{name}: staged file outside source manifest" for name in sorted(tracked - set(entries))]
    problems.extend(f"{name}: manifest file not staged" for name in sorted(set(entries) - tracked))

    def staged_bytes(name):
        return subprocess.run(["git", "show", ":" + name], cwd=ROOT, check=True, capture_output=True).stdout

    problems.extend(check_files(sorted(tracked & set(entries)), staged_bytes))
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="also scan the actual Git index")
    args = parser.parse_args()
    entries = source_paths()
    problems = check_files(entries)
    if args.staged:
        problems.extend(check_index(entries))
    if problems:
        print("Upload check failed; values are withheld:")
        print("\n".join(sorted(set(problems))))
        return 1
    print(f"Upload check passed: {len(entries)} source files; no detected credentials or notebook outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
