#!/usr/bin/env python3
"""Portable SLURM training workflow helper."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TIME_RE = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
NETWORK_COMMAND_RE = re.compile(r"\b(?:curl|wget|pip|conda\s+(?:install|update)|git\s+clone)\b", re.IGNORECASE)


class ConfigurationError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration root must be an object: {path}")
    return data


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    value = mapping.get(key)
    if value is None or value == "":
        raise ConfigurationError(f"Missing {context}.{key}")
    return value


def safe_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ConfigurationError(f"{label} must match {NAME_RE.pattern}")
    return value


def remote_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ConfigurationError(f"{label} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ConfigurationError(f"{label} must not contain '..'")
    return str(path)


def relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise ConfigurationError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ConfigurationError(f"{label} must not contain '..'")
    return str(path)


def positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def quote_command(parts: list[Any]) -> str:
    if not isinstance(parts, list) or not parts or not all(isinstance(item, str) and item for item in parts):
        raise ConfigurationError("Command must be a non-empty array of strings")
    return shlex.join(parts)


def remote_join(workspace: str, value: str) -> str:
    return str(PurePosixPath(workspace) / relative_path(value, "path"))


@dataclass(frozen=True)
class Cluster:
    host: str
    user: str
    port: int
    identity_file: str | None
    workspace: str
    source_dir: str
    data_dir: str
    weights_dir: str
    runs_dir: str
    logs_dir: str
    staging_dir: str
    slurm: dict[str, Any]
    policy: dict[str, Any]

    @property
    def logs_path(self) -> str:
        return str(PurePosixPath(self.workspace) / self.logs_dir)


def parse_cluster(data: dict[str, Any]) -> Cluster:
    connection = require(data, "connection", "cluster")
    remote = require(data, "remote", "cluster")
    slurm = require(data, "slurm", "cluster")
    policy = require(data, "policy", "cluster")
    if not all(isinstance(section, dict) for section in (connection, remote, slurm, policy)):
        raise ConfigurationError("cluster sections must be objects")
    host = require(connection, "host", "connection")
    user = require(connection, "user", "connection")
    if not isinstance(host, str) or not host.strip() or not isinstance(user, str) or not user.strip():
        raise ConfigurationError("connection host and user must be non-empty strings")
    port = connection.get("port", 22)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigurationError("connection.port must be between 1 and 65535")
    workspace = remote_path(require(remote, "workspace", "remote"), "remote.workspace")
    for key in ("source_dir", "data_dir", "weights_dir", "runs_dir", "logs_dir", "staging_dir"):
        relative_path(require(remote, key, "remote"), f"remote.{key}")
    safe_name(require(slurm, "partition", "slurm"), "slurm.partition")
    positive_int(require(slurm, "cpus_per_task", "slurm"), "slurm.cpus_per_task")
    positive_int(require(slurm, "gpus_per_task", "slurm"), "slurm.gpus_per_task")
    if not isinstance(require(slurm, "time_limit", "slurm"), str) or not TIME_RE.fullmatch(slurm["time_limit"]):
        raise ConfigurationError("slurm.time_limit must be D-HH:MM:SS or HH:MM:SS")
    if not isinstance(policy.get("compute_nodes_have_internet"), bool):
        raise ConfigurationError("policy.compute_nodes_have_internet must be boolean")
    return Cluster(
        host=host,
        user=user,
        port=port,
        identity_file=connection.get("identity_file"),
        workspace=workspace,
        source_dir=remote["source_dir"],
        data_dir=remote["data_dir"],
        weights_dir=remote["weights_dir"],
        runs_dir=remote["runs_dir"],
        logs_dir=remote["logs_dir"],
        staging_dir=remote["staging_dir"],
        slurm=slurm,
        policy=policy,
    )


def validate_experiment(data: dict[str, Any], cluster: Cluster) -> dict[str, Any]:
    name = safe_name(require(data, "name", "experiment"), "experiment.name")
    mode = require(data, "mode", "experiment")
    if mode not in {"single", "parallel"}:
        raise ConfigurationError("experiment.mode must be single or parallel")
    inputs = require(data, "inputs", "experiment")
    resources = require(data, "resources", "experiment")
    if not isinstance(inputs, dict) or not isinstance(resources, dict):
        raise ConfigurationError("experiment.inputs and experiment.resources must be objects")
    for key in ("data", "model"):
        relative_path(require(inputs, key, "inputs"), f"inputs.{key}")
    output_dir = relative_path(require(data, "output_dir", "experiment"), "experiment.output_dir")
    for key in ("cpus_per_task", "gpus_per_task"):
        if key in resources:
            positive_int(resources[key], f"resources.{key}")
    if "time_limit" in resources and (not isinstance(resources["time_limit"], str) or not TIME_RE.fullmatch(resources["time_limit"])):
        raise ConfigurationError("resources.time_limit must be D-HH:MM:SS or HH:MM:SS")
    command = require(data, "train_command", "experiment")
    quote_command(command)
    if not cluster.policy["compute_nodes_have_internet"] and NETWORK_COMMAND_RE.search(" ".join(command)):
        raise ConfigurationError("Training command contains a network/setup action while compute nodes have no internet")
    if mode == "parallel":
        trials = require(data, "trials", "experiment")
        if not isinstance(trials, list) or not trials:
            raise ConfigurationError("parallel experiment requires at least one trial")
        names: set[str] = set()
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict):
                raise ConfigurationError(f"trials[{index}] must be an object")
            trial_name = safe_name(require(trial, "name", f"trials[{index}]"), f"trials[{index}].name")
            if trial_name in names:
                raise ConfigurationError(f"Duplicate trial name: {trial_name}")
            names.add(trial_name)
            if not isinstance(trial.get("args", {}), dict):
                raise ConfigurationError(f"trials[{index}].args must be an object")
    return data


def ssh_base(cluster: Cluster) -> list[str]:
    command = ["ssh", "-p", str(cluster.port)]
    if cluster.identity_file:
        command.extend(["-i", os.path.expanduser(cluster.identity_file)])
    command.append(f"{cluster.user}@{cluster.host}")
    return command


def run(command: list[str], dry_run: bool = False, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("+", shlex.join(command))
    if dry_run:
        return None
    return subprocess.run(command, text=True, check=True, capture_output=capture)


def render_template(path: Path, values: dict[str, str]) -> str:
    content = path.read_text(encoding="utf-8")
    for key, value in values.items():
        content = content.replace("{{" + key + "}}", value)
    unresolved = re.findall(r"\{\{[^{}]+\}\}", content)
    if unresolved:
        raise ConfigurationError(f"Unresolved template variables in {path.name}: {', '.join(unresolved)}")
    return content


def resource(cluster: Cluster, experiment: dict[str, Any], key: str) -> Any:
    return experiment.get("resources", {}).get(key, cluster.slurm[key])


def common_script(root: Path, cluster: Cluster) -> str:
    module_loads = "\n".join(f"module load {shlex.quote(item)}" for item in cluster.slurm.get("module_loads", [])) or ":"
    conda_init = cluster.slurm.get("conda_init") or ":"
    conda_env = cluster.slurm.get("conda_env")
    activate = f"source activate {shlex.quote(conda_env)}" if conda_env else ":"
    return render_template(root / "templates" / "common.sh.tpl", {
        "LOG_DIR": shlex.quote(cluster.logs_path),
        "WORKSPACE": shlex.quote(cluster.workspace),
        "MODULE_LOADS": module_loads,
        "CONDA_INIT": conda_init,
        "CONDA_ACTIVATE": activate,
    })


def command_context(cluster: Cluster, experiment: dict[str, Any], trial: dict[str, Any] | None = None) -> dict[str, str]:
    output_dir = remote_join(cluster.workspace, experiment["output_dir"])
    context = {
        "data": remote_join(cluster.workspace, experiment["inputs"]["data"]),
        "model": remote_join(cluster.workspace, experiment["inputs"]["model"]),
        "name": experiment["name"],
        "output_dir": output_dir,
        "output_parent": str(PurePosixPath(output_dir).parent),
        "best_model": str(PurePosixPath(output_dir) / "weights" / "best.pt"),
    }
    if trial:
        trial_output = str(PurePosixPath(output_dir) / trial["name"])
        context.update({"trial_name": trial["name"], "trial_output_dir": trial_output, "trial_output_parent": str(PurePosixPath(trial_output).parent)})
        context.update({str(key): str(value) for key, value in trial.get("args", {}).items()})
    return context


def expand_command(parts: list[str], context: dict[str, str]) -> str:
    try:
        return quote_command([part.format_map(context) for part in parts])
    except KeyError as error:
        raise ConfigurationError(f"Missing command placeholder: {error.args[0]}") from error


def optional_directives(cluster: Cluster) -> str:
    directives: list[str] = []
    if cluster.slurm.get("account"):
        directives.append(f"#SBATCH --account={cluster.slurm['account']}")
    if cluster.slurm.get("qos"):
        directives.append(f"#SBATCH --qos={cluster.slurm['qos']}")
    if cluster.slurm.get("memory"):
        directives.append(f"#SBATCH --mem={cluster.slurm['memory']}")
    return "\n".join(directives)


def input_checks(cluster: Cluster, experiment: dict[str, Any]) -> str:
    paths = [remote_join(cluster.workspace, experiment["inputs"]["data"]), remote_join(cluster.workspace, experiment["inputs"]["model"])]
    lines = [f"test -f {shlex.quote(path)} || {{ echo 'Required file missing: {path}' >&2; exit 3; }}" for path in paths]
    return "\n".join(lines)


def render_job(root: Path, cluster: Cluster, experiment: dict[str, Any]) -> str:
    common = common_script(root, cluster)
    values = {
        "JOB_NAME": safe_name(experiment["name"], "experiment.name"),
        "CPUS_PER_TASK": str(resource(cluster, experiment, "cpus_per_task")),
        "GPUS_PER_TASK": str(resource(cluster, experiment, "gpus_per_task")),
        "PARTITION": cluster.slurm["partition"],
        "TIME_LIMIT": resource(cluster, experiment, "time_limit"),
        "LOG_DIR": cluster.logs_path,
        "OPTIONAL_DIRECTIVES": optional_directives(cluster),
        "COMMON": common,
        "INPUT_CHECKS": input_checks(cluster, experiment),
    }
    if experiment["mode"] == "single":
        values["TRAIN_COMMAND"] = expand_command(experiment["train_command"], command_context(cluster, experiment))
        return render_template(root / "templates" / "train-single.slurm.tpl", values)
    trial_cases: list[str] = []
    for index, trial in enumerate(experiment["trials"]):
        command = expand_command(experiment["train_command"], command_context(cluster, experiment, trial))
        trial_cases.append(f"  {index}) TRAIN_COMMAND={shlex.quote(command)} ;;")
    values["TRIAL_COUNT"] = str(len(experiment["trials"]))
    values["TRIAL_CASES"] = "\n".join(trial_cases)
    return render_template(root / "templates" / "train-parallel.slurm.tpl", values)


def local_dataset_check(dataset: Path, profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    root = dataset / profile.get("root", ".")
    for key in ("train_images", "train_labels", "val_images", "val_labels"):
        value = profile.get(key)
        if not isinstance(value, str):
            errors.append(f"Missing dataset profile field: {key}")
            continue
        path = root / value
        if not path.is_dir():
            errors.append(f"Missing directory: {path}")
    if errors or not profile.get("require_label_pairing", False):
        return errors
    for split in ("train", "val"):
        images = root / profile[f"{split}_images"]
        labels = root / profile[f"{split}_labels"]
        image_stems = {item.stem for item in images.iterdir() if item.is_file()}
        label_stems = {item.stem for item in labels.glob("*.txt") if item.is_file()}
        missing = sorted(image_stems - label_stems)
        if missing:
            errors.append(f"{split}: {len(missing)} image(s) without .txt label")
    return errors


def read_csv_summary(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ConfigurationError(f"CSV contains no rows: {path}")
    fields = set(rows[0])
    metric_aliases = {
        "map50": ("metrics/mAP50(B)", "metrics/mAP50", "mAP50"),
        "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"),
        "precision": ("metrics/precision(B)", "metrics/precision", "precision"),
        "recall": ("metrics/recall(B)", "metrics/recall", "recall"),
    }
    summary: dict[str, Any] = {"file": str(path), "epochs": len(rows)}
    for label, aliases in metric_aliases.items():
        column = next((item for item in aliases if item in fields), None)
        if not column:
            continue
        values = [(index, float(row[column].strip())) for index, row in enumerate(rows) if row.get(column, "").strip()]
        if values:
            best_index, best_value = max(values, key=lambda item: item[1])
            summary[label] = {"column": column, "best": best_value, "best_epoch": best_index, "final": values[-1][1]}
    return summary


def command_doctor(args: argparse.Namespace, cluster: Cluster) -> int:
    missing = [tool for tool in ("ssh", "scp") if shutil.which(tool) is None]
    if missing:
        raise ConfigurationError(f"Required local command(s) not found: {', '.join(missing)}")
    remote = " && ".join([
        f"test -d {shlex.quote(cluster.workspace)}",
        "command -v sbatch",
        "command -v squeue",
        "command -v sacct",
    ])
    run(ssh_base(cluster) + [remote], args.dry_run)
    print("Doctor check completed. Submit a site-approved short GPU probe separately if GPU execution must be verified.")
    return 0


def command_render(args: argparse.Namespace, root: Path, cluster: Cluster, experiment: dict[str, Any]) -> int:
    script = render_job(root, cluster, experiment)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script, encoding="utf-8", newline="\n")
        print(f"Rendered job script: {output}")
    else:
        print(script, end="" if script.endswith("\n") else "\n")
    return 0


def command_sync(args: argparse.Namespace, cluster: Cluster) -> int:
    local = Path(args.source).resolve()
    if not local.is_dir():
        raise ConfigurationError(f"Source directory not found: {local}")
    remote_target = str(PurePosixPath(cluster.workspace) / cluster.source_dir)
    run(ssh_base(cluster) + [f"mkdir -p {shlex.quote(remote_target)}"], args.dry_run)
    command = ["scp", "-P", str(cluster.port), "-r", str(local) + os.sep + ".", f"{cluster.user}@{cluster.host}:{remote_target}/"]
    if cluster.identity_file:
        command[1:1] = ["-i", os.path.expanduser(cluster.identity_file)]
    run(command, args.dry_run)
    return 0


def command_submit(args: argparse.Namespace, cluster: Cluster) -> int:
    local_script = Path(args.script).resolve()
    if not local_script.is_file():
        raise ConfigurationError(f"Rendered script not found: {local_script}")
    remote_dir = str(PurePosixPath(cluster.workspace) / cluster.staging_dir)
    remote_file = str(PurePosixPath(remote_dir) / local_script.name)
    run(ssh_base(cluster) + [f"mkdir -p {shlex.quote(remote_dir)}"], args.dry_run)
    copy = ["scp", "-P", str(cluster.port), str(local_script), f"{cluster.user}@{cluster.host}:{remote_file}"]
    if cluster.identity_file:
        copy[1:1] = ["-i", os.path.expanduser(cluster.identity_file)]
    run(copy, args.dry_run)
    result = run(ssh_base(cluster) + [f"sbatch --parsable {shlex.quote(remote_file)}"], args.dry_run, capture=True)
    if result:
        print(f"Submitted job: {result.stdout.strip()}")
    return 0


def command_status(args: argparse.Namespace, cluster: Cluster) -> int:
    job = safe_name(args.job_id, "job id")
    remote = f"squeue -j {shlex.quote(job)}; scontrol show job {shlex.quote(job)}; sacct -j {shlex.quote(job)} --format=JobID,State,ExitCode,Elapsed,Reason"
    run(ssh_base(cluster) + [remote], args.dry_run)
    return 0


def command_cancel(args: argparse.Namespace, cluster: Cluster) -> int:
    job = safe_name(args.job_id, "job id")
    run(ssh_base(cluster) + [f"scancel {shlex.quote(job)}"], args.dry_run)
    return 0


def command_fetch(args: argparse.Namespace, cluster: Cluster) -> int:
    remote_relative = relative_path(args.remote_path, "remote path")
    local = Path(args.destination).resolve()
    local.mkdir(parents=True, exist_ok=True)
    remote = str(PurePosixPath(cluster.workspace) / remote_relative)
    command = ["scp", "-P", str(cluster.port), "-r", f"{cluster.user}@{cluster.host}:{remote}", str(local)]
    if cluster.identity_file:
        command[1:1] = ["-i", os.path.expanduser(cluster.identity_file)]
    run(command, args.dry_run)
    if args.delete_remote:
        if not cluster.policy.get("allow_remote_delete"):
            raise ConfigurationError("Remote deletion is disabled by cluster policy")
        if args.confirm != remote:
            raise ConfigurationError("Remote deletion requires --confirm with the exact remote path")
        run(ssh_base(cluster) + [f"rm -rf -- {shlex.quote(remote)}"], args.dry_run)
    return 0


def command_dataset(args: argparse.Namespace) -> int:
    profile = load_json(Path(args.profile))
    errors = local_dataset_check(Path(args.dataset), profile)
    if errors:
        print("Dataset validation failed:", *[f"- {error}" for error in errors], sep="\n")
        return 1
    print("Dataset validation passed.")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    paths = [Path(item) for item in args.csv]
    summaries = [read_csv_summary(path) for path in paths]
    summaries.sort(key=lambda item: item.get("map50", {}).get("best", float("-inf")), reverse=True)
    payload = {"experiments": summaries}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Analysis written to: {args.output}")
    else:
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configuration-driven SLURM training workflow helper")
    parser.add_argument("--cluster", type=Path, help="Cluster JSON configuration")
    parser.add_argument("--dry-run", action="store_true", help="Print external commands without running them")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check local tools and login-node SLURM prerequisites")
    render = sub.add_parser("render", help="Render an experiment config to a SLURM script")
    render.add_argument("--experiment", type=Path, required=True)
    render.add_argument("--output")
    sync = sub.add_parser("sync", help="Copy a source directory to the login node")
    sync.add_argument("--source", required=True)
    submit = sub.add_parser("submit", help="Upload a rendered script and submit it")
    submit.add_argument("--script", required=True)
    status = sub.add_parser("status", help="Show queue, detail, and accounting data for one job")
    status.add_argument("job_id")
    cancel = sub.add_parser("cancel", help="Cancel one job")
    cancel.add_argument("job_id")
    fetch = sub.add_parser("fetch", help="Download artifacts from the login node")
    fetch.add_argument("--remote-path", required=True)
    fetch.add_argument("--destination", required=True)
    fetch.add_argument("--delete-remote", action="store_true")
    fetch.add_argument("--confirm")
    dataset = sub.add_parser("check-dataset", help="Validate a local dataset profile")
    dataset.add_argument("--dataset", required=True)
    dataset.add_argument("--profile", required=True)
    analyze = sub.add_parser("analyze", help="Summarize one or more training CSV files")
    analyze.add_argument("csv", nargs="+")
    analyze.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check-dataset":
            return command_dataset(args)
        if args.command == "analyze":
            return command_analyze(args)
        if not args.cluster:
            raise ConfigurationError("--cluster is required for this command")
        root = Path(__file__).resolve().parents[1]
        cluster = parse_cluster(load_json(args.cluster))
        if args.command == "doctor":
            return command_doctor(args, cluster)
        if args.command == "render":
            experiment = validate_experiment(load_json(args.experiment), cluster)
            return command_render(args, root, cluster, experiment)
        if args.command == "sync":
            return command_sync(args, cluster)
        if args.command == "submit":
            return command_submit(args, cluster)
        if args.command == "status":
            return command_status(args, cluster)
        if args.command == "cancel":
            return command_cancel(args, cluster)
        if args.command == "fetch":
            return command_fetch(args, cluster)
        raise AssertionError(f"Unhandled command: {args.command}")
    except (ConfigurationError, OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
