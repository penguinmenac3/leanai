"""doc
# leanai_launch

> A little tool to help with running the same config over and over.

## workspace.json

In `.leanai/workspace.json` put the following content:
```json
{
    "folders": [
        "leanai"
    ],
    "commands": {
        "Name For My Command": {
            "cmd": [
                "echo",
                "Hello World!"
            ]
        }
    }
}
```

## Launching configs

You can either directly launch a config:
```bash
leanai_launch "Name For My Command"
```
or select from your list interactively:
```bash
leanai_launch
```

"""
import os
import json
import argparse

def _run(cmd):
    print(f"> {cmd}")
    try:
        os.system(cmd)
    except KeyboardInterrupt:
        pass

def launcher(config):
    if not os.path.exists(".leanai/workspace.json"):
        raise FileNotFoundError("You must create a .leanai/workspace.json to use this feature.")
    with open(".leanai/workspace.json", "r") as f:
        workspace = json.loads(f.read())
        commands = workspace["commands"]
    config = ask_for_config(config, commands)
    cmd = commands[config]["cmd"]
    _run(" ".join(cmd))

def ask_for_config(config, commands):
    while config is None:
        print("Please select a command from the list by entering the number:")
        command_list = []
        for idx, command in enumerate(commands.keys()):
            print(f"{idx}: {command}")
            command_list.append(command)
        number = input("Insert number for key: ")
        try:
            number = int(number)
        except:
            print("Your input does not seem to be an integer.")
            continue
        try:
            config = command_list[number]
        except:
            print("You selected an invalid index.")
            continue
    return config

def main():
    parser = argparse.ArgumentParser(description='Launch configs from your workspace.json')
    parser.add_argument('-c', '--config', type=str, default=None, required=False, help='The key of the config in commands to launch.')
    args, other_args = parser.parse_known_args()
    launcher(args.config)

if __name__ == "__main__":
    main()
