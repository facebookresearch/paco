#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import sys
from pathlib import Path

import submitit
import torch  # noqa import first to avoid https://github.com/pytorch/pytorch/issues/37377
import yaml
from detectron2.engine import default_argument_parser, launch
from detectron2.utils.env import _import_file

DEFAULT_TARGET = Path(__file__).resolve().parent.parent.parent / "tools/train_net.py"
DEFAULT_TARGET_LAZY = (
    Path(__file__).resolve().parent.parent.parent / "tools/lazyconfig_train_net.py"
)


"""
Example:
    python3 ./tools/multi_node_training.py \
        --config-file configs/mask_rcnn_configs/r101_attr_fpn_100_ep.py \
        --num-gpus 8 \
        --num-machines 8 \
        --use-volta32 \
        --name "r101_100_ep" \
        --target "tools/lazyconfig_train_net.py" \
        --job-dir "output/r101_100_ep/"
"""


def parse_args():
    d2_arg_parser = default_argument_parser()
    parser = argparse.ArgumentParser(
        "Submitit for Detectron2", parents=[d2_arg_parser], add_help=False
    )
    parser.add_argument(
        "-p",
        "--partition",
        default="learnfair",
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--timeout", default=60 * 72, type=int, help="Duration of the job"
    )
    parser.add_argument(
        "--job-dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Weights to resume from (.*pth file) or a file (last_checkpoint) "
        "that contains " + "weight file name from the same directory",
    )
    parser.add_argument(
        "--resume-job", default="", type=str, help="resume training from the job"
    )
    parser.add_argument(
        "--use-volta32", action="store_true", help="Big models? Use this"
    )
    parser.add_argument(
        "--name", default="detectron2_job", type=str, help="Name of the job"
    )
    parser.add_argument(
        "--target",
        default="lightning_train_net.py",
        type=str,
        help="The target python file with a main() function to launch. "
        "Default is train_net.py or lazyconfig_train_net.py",  # noqa
    )
    parser.add_argument(
        "--mail",
        default="",
        type=str,
        help="Email this user when the job finishes if specified",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    args = parser.parse_args()
    if args.target is None:
        if is_yacs_cfg(args.config_file):
            args.target = DEFAULT_TARGET
        else:
            args.target = DEFAULT_TARGET_LAZY
    assert os.path.isfile(args.target), args.target
    return args


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/detectron2_experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def is_yacs_cfg(config_file):
    if config_file.endswith(".py"):
        return False
    else:
        with open(config_file) as f:
            obj = yaml.unsafe_load(f.read())
        return "train" not in obj


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.target = os.path.realpath(args.target)

    def __call__(self):
        sys.path.insert(0, os.path.dirname(self.args.target))
        module_name = os.path.splitext(os.path.basename(self.args.target))[0]
        main_module = _import_file(module_name, self.args.target, True)

        socket_name = (
            os.popen("ip r | grep default | awk '{print $5}'").read().strip("\n")
        )
        print(f"[launcher] Setting GLOO and NCCL sockets IFNAME to: {socket_name}")
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name
        os.environ["NCCL_SOCKET_IFNAME"] = socket_name
        os.environ["NCCL_DEBUG"] = "INFO"
        roce_hca = (
            os.popen(
                "ibstat | grep 'Link layer: Ethernet' -B50 | grep \"CA '\" | tail -n1 | awk '{print $2}' "  # noqa
            )
            .read()
            .strip("'\n")
        )
        if roce_hca:
            print(f"[launcher] Disable ROCE HCA {roce_hca}")
            os.environ["NCCL_IB_HCA"] = f"^{roce_hca}"

        hostname_first_node = (
            os.popen("scontrol show hostnames $SLURM_JOB_NODELIST")
            .read()
            .split("\n")[0]
        )
        dist_url = f"tcp://{hostname_first_node}:12399"
        print(f"[launcher] Using the following dist url: {dist_url}")

        self._setup_gpu_args()
        launch(
            main_module.main,
            self.args.num_gpus,
            num_machines=self.args.num_machines,
            machine_rank=self.args.machine_rank,
            dist_url=dist_url,
            args=(self.args,),
        )

    def checkpoint(self):
        self.args.resume = True
        print("[launcher] Requeuing ", self.args)
        return submitit.helpers.DelayedSubmission(type(self)(self.args))

    def _setup_gpu_args(self):

        job_env = submitit.JobEnvironment()
        output_dir = str(self.args.output_dir).replace("%j", str(job_env.job_id))
        if self.args.resume_from is not None:
            if job_env.global_rank == 0:
                p = os.path.join(output_dir, "output")
                os.makedirs(p, exist_ok=True)
                if self.args.resume_from.endswith(".pth"):
                    weights_file = self.args.resume_from
                else:
                    with open(self.args.resume_from) as f:
                        weights_filename = f.read().strip()
                    weights_file = os.path.join(
                        os.path.dirname(self.args.resume_from), weights_filename
                    )
                print(f"[launcher] Copy weights file {weights_file} to {p}")
                shutil.copy(weights_file, p)
                with open(os.path.join(p, "last_checkpoint"), "w") as f:
                    f.write(os.path.basename(weights_file))
            self.args.resume = True
            self.args.resume_from = None
        if is_yacs_cfg(self.args.config_file):
            self.args.opts.extend(["OUTPUT_DIR", os.path.join(output_dir, "output")])
        else:
            self.args.opts.append(
                "train.output_dir=" + os.path.join(output_dir, "output")
            )
        self.args.machine_rank = job_env.global_rank


def main():
    args = parse_args()
    assert args.config_file
    if args.job_dir == "":
        job_dir = get_shared_folder()
        args.job_dir = job_dir / "%j"
    else:
        job_dir = args.job_dir

    if args.resume_job != "":
        assert args.resume_from is None, "Cannot have both resume_job and resume_from!"
        print(f"[launcher] Resuming job {args.resume_job}")
        job_dir_to_resume = os.path.join(
            str(args.job_dir).replace("%j", args.resume_job), "output"
        )
        resume_from = os.path.join(job_dir_to_resume, "last_checkpoint")
        if os.path.isfile(resume_from):
            args.resume_from = resume_from
        name = (
            os.popen(f'sacct -j {args.resume_job} -X --format "JobName%200" -n')
            .read()
            .strip()
        )
        args.name = f"{name}_resumed_from_{args.resume_job}"

    # Note that the folder will depend on the job_id, to easily track
    # experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    # cluster setup is defined by environment variables
    num_gpus_per_node = args.num_gpus
    nodes = args.num_machines
    partition = args.partition
    timeout_min = args.timeout
    kwargs = {}
    if args.use_volta32:
        # "constraint" is deprecated.
        kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["comment"] = args.comment

    executor.update_parameters(
        mem_gb=60 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,
        cpus_per_task=10 * num_gpus_per_node,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name=args.name)
    if args.mail:
        executor.update_parameters(
            additional_parameters={"mail-user": args.mail, "mail-type": "END"}
        )

    args.output_dir = args.job_dir
    trainer = Trainer(args)
    job = executor.submit(trainer)
    print(f"[launcher] Submitted job_id: {job.job_id}, dir: {job.paths.folder}")


if __name__ == "__main__":
    main()
