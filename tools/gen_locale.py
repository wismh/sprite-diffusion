"""Writes assets/locale/en.strings and uk.strings from the string tables: the window (strings_ui.py), the messages the
program writes (strings_msg.py) and the Help tab (help/help_content.py). Run it after any change of those; then run
help/gen_help.py if the help topics changed."""
import os
import re
import secrets
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, "help"))
import gen_help
from strings_msg import MSG
from strings_ui import UI

root = os.path.join(os.path.dirname(here), "assets", "locale")


def toml_text(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def all_strings():
    table = {}
    for source in (UI, MSG, gen_help.strings()):
        for key, value in source.items():
            if key in table:
                raise ValueError("duplicate key " + key)
            table[key] = value
    return table


def check_keys(table):
    """Every key the code and the documents use must exist."""
    project = os.path.dirname(here)
    used = set()
    for folder, _, files in os.walk(os.path.join(project, "src")):
        for name in files:
            if name.endswith((".cpp", ".h")):
                text = open(os.path.join(folder, name), encoding="utf-8").read()
                used.update(re.findall(r'(?:tr_|say|Message|tr)\(\s*"([a-z_.0-9]+)"', text))
                used.update(re.findall(r'"((?:ds|model|train|gen|insp|filters|process|status|stage)\.[a-z_.0-9]+)"', text))
    for folder, _, files in os.walk(os.path.join(project, "assets", "ui")):
        for name in files:
            if name.endswith(".xml"):
                used.update(re.findall(r"\{tr ([a-zA-Z_.0-9]+)\}", open(os.path.join(folder, name), encoding="utf-8").read()))
    missing = sorted(k for k in used if k not in table and not k.endswith(".") and not k.startswith("stage."))
    if missing:
        raise SystemExit("missing keys: " + ", ".join(missing))


def write_locale(tag, index):
    table = all_strings()
    lines = [f'locale = "{tag}"', ""]
    for key in sorted(table):
        lines += ["[[string]]", f'id = "{key}"', f"text = {toml_text(table[key][index])}", ""]
    path = os.path.join(root, f"{tag}.strings")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    meta = path + ".meta"
    if not os.path.exists(meta):
        with open(meta, "w", encoding="utf-8", newline="\n") as f:
            extra = "source = true\n" if tag == "en" else ""
            f.write(f'guid = "{secrets.token_hex(16)}"\nimporter = "strings"\n{extra}locale = "{tag}"\n')


def main():
    os.makedirs(root, exist_ok=True)
    table = all_strings()
    check_keys(table)
    write_locale("en", 0)
    write_locale("uk", 1)
    print(len(table), "strings")


if __name__ == "__main__":
    main()
