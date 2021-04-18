[Back to Overview](../../README.md)



# leanai_remote

> A little tool to help with running stuff on a cluster or remotely.

## Remote Run (Server or SLURM)

See the help by running.
```bash
leanai_remote --help
```
This will move the code from the workspace to "{user}@{host}:{remote_results_path}/{name}/{version}/src/{repository}" and then execute it.

## Run templates

The script also takes care of filling the templates with the right information to run them on the remote.

**SLURM:**
The `run.template.slurm` is a file in your cwd which must be runnable by sbatch.
It can contain `{remote_results_path}, {repository}, {partition}, {name}, {version}, {user}` which will be replaced by the arguments provided when calling `leanai_remote`.
Note that this script itself can call the run.sh which will be also generated in the slurm case.

**RUN:**
The `run.template.sh` is a file containing the actual code that should be run.
It can contain `{remote_results_path}, {repository}, {name}, {version}, {cmd}, {user}` which will be replaced by the arguments provided when calling `leanai_remote`.


---
### *def* **run**(*self*, name, version_suffix, workspace, repository, remote_results_path, **run**_template, cmd, nodes, gpus, cpus)

*(no documentation found)*

---
### *def* **main**()

*(no documentation found)*

