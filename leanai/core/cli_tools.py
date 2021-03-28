"""doc
# leanai.core.cli_tools

> The command line interface for project management with leanai.

You can create a new project with the following command:
```bash
leanai --new_project --name=<name>
```

Also you can add run configurations for vscode for a project with this command:
```bash
leanai --vscode_run_config <name> <module> <arg1> <arg2> ...
```
"""
import sys
import os
import json
import jstyleson
import argparse
import shutil
TEMPLATE_CLONE_URL = "https://github.com/penguinmenac3/leanai-template.git"


def main():
    args = sys.argv[1:]

    valid_tools = ["--new_project", "--vscode_run_config"]
    if args[0] not in valid_tools:
        print("Unknown tool {}. Chose one in {}.".format(args[0], valid_tools))
        return

    if args[0] == "--new_project":
        new_project(args[1:])
    elif args[0] == "--vscode_run_config":
        vscode_run_config(args[1:])


def new_project(args):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('name', type=str, help="Project name")
    args, run_args = parser.parse_known_args(args)
    if os.path.exists(args.name):
        print("Cannot create project. Folder '{}' already exists.".format(args.name))
        return
    os.system("git clone {} {}".format(TEMPLATE_CLONE_URL, args.name))
    os.chdir(args.name)
    shutil.move("template", args.name)
    vscode_run_config(["Train", "{}.config.train".format(args.name)])
    vscode_run_config(["Test", "{}.config.test".format(args.name), "--mode=test"])


def vscode_run_config(args):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('name', type=str, help="The name under which it should show in the run config dropdown.")
    parser.add_argument('module', type=str, help="The name of the module that should be executed, e.g. 'leanai.core.cli'")
    args, run_args = parser.parse_known_args(args)
    if not os.path.exists(".vscode"):
        os.mkdir(".vscode")
        print("Created .vscode folder.")
    if not os.path.exists(os.path.join(".vscode", "launch.json")):
        with open(os.path.join(".vscode", "launch.json"), "w") as f:
            data = {"version": "0.2.0","configurations": []}
            f.write(json.dumps(data, indent=4, sort_keys=True))
            f.write("\n")
        print("Created launch.json.")
    with open(os.path.join(".vscode", "launch.json"), "r") as f:
        data = jstyleson.load(f)
        data["configurations"].append({
            "name": args.name,
            "type": "python",
            "request": "launch",
            "module": args.module,
            "args": run_args
        })
    with open(os.path.join(".vscode", "launch.json"), "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))
        f.write("\n")
    print("Added configuration under name: {}. You may select it in the debug menu now.".format(args.name))

