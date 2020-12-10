from options.train_options import TrainOptions
from train import main
from data import create_dataset
import copy
import submitit
import os


PARTITION = 'learnfair'
ARRAY_PARALLELISM = 512
NUM_CPUS = 32
NUM_GPUS = 8
JOB_TIME = 3*24*60 # 3 days
MAX_JOBS = 100
LOG_DIR = "/checkpoint/hberard/DomainBed/logs"


opt = TrainOptions().parse()
dataset = create_dataset(opt)

os.makedirs(LOG_DIR, exist_ok=True)
executor = submitit.AutoExecutor(folder=LOG_DIR)
executor.update_parameters(slurm_time=JOB_TIME,
        gpus_per_node=NUM_GPUS,
        slurm_array_parallelism=ARRAY_PARALLELISM,
        cpus_per_task=NUM_CPUS,
        slurm_partition=PARTITION)

jobs = []
for i, A in enumerate(dataset.dataset.envs.ENVIRONMENT_NAMES):
    for j, B in enumerate(dataset.dataset.envs.ENVIRONMENT_NAMES):
        if A == B:
            continue
        opt = copy.deepcopy(opt)
        opt.A = i
        opt.B = j
        opt.name = "%s/%s2%s"%(opt.dataset, A, B)
        opt.gpu_ids = range(NUM_GPUS)
        job = executor.submit(main, opt)
        jobs.append(job)
        print(job.job_id)