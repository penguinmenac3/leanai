"""doc
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
"""
import os
import time
import argparse
from leanai.core.experiment import _generate_version


class _RemoteRunner:
    def __init__(self, host, slurm_template, user, dry_run=False, slurm=None):
        self.host = host
        self.slurm_template = slurm_template
        self.dry_run = dry_run
        self.slurm = slurm
        self.user = user

    def run(self, name, version_suffix, workspace, repository, remote_results_path, run_template, cmd, nodes, gpus, cpus):
        display_name = name
        version = _generate_version()
        if version_suffix is not None:
            version = f"{version}_{version_suffix}"
            display_name = f"{name}_{version_suffix}"

        print(f"Preparing: {cmd}")
        def _run(cmd):
            print(f"> {cmd}")
            if not self.dry_run:
                os.system(cmd)

        slurmfile = os.path.join(workspace, repository, "run.slurm")
        if self.slurm is not None:
            with open(slurmfile, "w") as f:
                cpu_mode = "--cpus-per-task" if gpus == 0 else "--cpus-per-gpu"
                f.write(self.slurm_template.format(repository=repository, partition=self.slurm, remote_results_path=remote_results_path, name=name, version=version, user=self.user, display_name=display_name, nodes=nodes, gpus=gpus, cpus=cpus, cpu_mode=cpu_mode))

        runfile = os.path.join(workspace, repository, "run.sh")
        with open(runfile, "w") as f:
            f.write(run_template.format(repository=repository, remote_results_path=remote_results_path, name=name, version=version, cmd=cmd, user=self.user))

        _run(f"ssh {self.user}@{self.host} mkdir -p {remote_results_path}/{name}/{version}/src")
        _run(f"cd {workspace} && tar --exclude=.git --exclude=__pycache__ --exclude=*.egg-info --exclude=docs --exclude=.vscode -cf - . | ssh {self.user}@{self.host} tar -xf - -C {remote_results_path}/{name}/{version}/src")
        
        if os.path.exists(slurmfile):
            os.remove(slurmfile)
        if os.path.exists(runfile):
            os.remove(runfile)

        if self.slurm is not None:
            _run(f"ssh {self.user}@{self.host} sbatch {remote_results_path}/{name}/{version}/src/{repository}/run.slurm")
            print("Waiting for output file...")
            while not os.path.exists(f"{remote_results_path}/{name}/{version}/out.txt"):
                time.sleep(1)
            _run(f"tail -f {remote_results_path}/{name}/{version}/out.txt")
        else:
            _run(f"ssh {self.user}@{self.host} screen -dmS '{version}' bash {remote_results_path}/{name}/{version}/src/{repository}/run.sh")


def main():
    slurm_host = "localhost"
    if "SLURM_HOST" in os.environ:
        slurm_host = os.environ["SLURM_HOST"]
    default_partition = "batch"
    if "SLURM_PARTITION" in os.environ:
        default_partition = os.environ["SLURM_PARTITION"]
    remote_results_path = None
    if "REMOTE_RESULTS_PATH" in os.environ:
        remote_results_path = os.environ["REMOTE_RESULTS_PATH"]
    user = os.environ["USER"]

    parser = argparse.ArgumentParser(description='The main entry point for the script.')
    parser.add_argument('--name', type=str, required=True, help='The name for the experiment.')
    parser.add_argument('--version', type=str, required=False, default=None, help='A suffix that is added to the version if you want. Otherwise it will be just the date without suffix.')
    parser.add_argument('--cmd', type=str, required=False, default="python", help='The command to run.')
    parser.add_argument('--host', type=str, required=False, default=slurm_host, help='The hostname to connect to (e.g. "server.de"), defaults to $SLURM_HOST.')
    parser.add_argument('--user', type=str, required=False, default=user, help='The username used, defaults to $USER.')
    parser.add_argument('--slurm_template', type=str, required=False, default="run.template.slurm", help='The slurm template file.')
    parser.add_argument('--run_template', type=str, required=False, default="run.template.sh", help='The run template file.')
    parser.add_argument('--repository', type=str, default=None, required=False, help='What package should be executed (defaults to cwd).')
    parser.add_argument('--workspace', type=str, default="..", required=False, help='Where the workspace with all git projects is. The entire workspace will be moved to the remote. (defaults to "..", parent directory).')
    parser.add_argument('--partition', type=str, required=False, default=default_partition, help='Select the partition on which to run the code.')
    parser.add_argument('--remote_results_path', type=str, required=(remote_results_path is None), default=remote_results_path, help='The path where on the remote the results live (defaults to $REMOTE_RESULTS_PATH).')
    parser.add_argument('--dry-run', action='store_true', help='This flag will not execute the commands on the cluster but only emulate locally.')
    parser.add_argument('--nodes', type=int, required=False, default=1, help='Select the number of nodes (Default: 1)')
    parser.add_argument('--gpus', type=int, required=False, default=1, help='Select the number of gpus (Default: 1)')
    parser.add_argument('--cpus', type=int, required=False, default=4, help='Select the number of cpus (Default: 4)')
    args, other_args = parser.parse_known_args()

    repository = args.repository
    if repository is None:
        repository = os.getcwd().split("/")[-1]

    name = args.name
    version = args.version

    cmd = f"{args.cmd} {' '.join(other_args)}"

    slurm = None
    if args.host == slurm_host:
        slurm = args.partition
    with open(args.run_template, "r") as f:
        run_template = f.read()

    slurm_template = ""
    if slurm:
        with open(args.slurm_template, "r") as f:
            slurm_template = f.read()

    remote_runner = _RemoteRunner(args.host, slurm_template, user, args.dry_run, slurm=slurm)
    remote_runner.run(name, version, args.workspace, repository, args.remote_results_path, run_template, cmd, args.nodes, args.gpus, args.cpus)


if __name__ == "__main__":
    main()
