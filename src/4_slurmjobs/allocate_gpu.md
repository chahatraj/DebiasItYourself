## 1) Allocate GPU

this is the environment we will be working in:
/home/craj/nanotron-env/bin/activate

we allocate gpus using salloc commands, for example:
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40G --time=0-2:00:00
salloc -p contrib-gpuq -q cs_dept --nodes=1 --ntasks-per-node=1 --gres=gpu:3g.40gb:1 --mem=40G --time=0-2:00:00
salloc -p contrib-gpuq -q cs_dept --nodes=1 --ntasks-per-node=1 --gres=gpu:A100.80gb:1 --mem=40G --time=0-4:00:00
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:A100.80gb:1 --mem=90G --time=0-24:00:00

we have contrib-gpuq and gpuq
and we have cs_dept and gpu
the combination of gpuq and gpu is preemptive

we can change the time according to job requirements. 

we do sgpu to check available gpus

we do sacct -X and squeue -u craj to monitor jobs

we submit jobs using sbatch and slurm scripts


## 2) Allocate interactive GPU

Use this allocation pattern:

```bash
salloc -p contrib-gpuq -q cs_dept \
  --nodes=1 --ntasks-per-node=1 \
  --gres=gpu:3g.40gb:1 --mem=40G --time=0-6:00:00
```

Wait for:

`Nodes <hostname> are ready for job`


## 3) Multiple interactive jobs in parallel

If two jobs land on the same host, avoid rendezvous port collision by setting unique ports, for example:

```bash
/home/craj/nanotron-env/bin/python -m torch.distributed.run \
  --nproc-per-node 1 --master_port 29611 \
  run_train.py --config /ABS/PATH/TO/config_B.yaml
```

Use a different port per concurrent job on the same node (e.g. `29611`, `29612`, `29613`).

## 4) Monitor status

```bash
squeue -u craj -o '%.18i %.15P %.30j %.10T %.12M %.6D %R'
```

## 5) Release GPUs when done

```bash
scancel <jobid1> <jobid2> ...
```

## Known gotchas

- `RuntimeError: 0 active drivers`: command ran on login node, not allocated GPU node.
- `EADDRINUSE ... port 29500`: for `torchrun`, set a unique `--master_port`.
- Interactive job can be `RUNNING` after your command exits; always cancel to free GPU.
