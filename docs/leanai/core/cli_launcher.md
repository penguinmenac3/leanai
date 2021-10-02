[Back to Overview](../../README.md)



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



---
### *def* **launcher**(config)

*(no documentation found)*

---
### *def* **ask_for_config**(config, commands)

*(no documentation found)*

---
### *def* **main**()

*(no documentation found)*

