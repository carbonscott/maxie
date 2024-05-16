### Locate the `train` directory

```bash
export MAXIE=<your maxie location>
export MAXIE_TRAIN="$MAXIE/train"
```

### Copy configs and python scripts to your working directory

A working directory where you preprocess your data, e.g. mine is

```bash
export PROJ_MAXIE_TRAIN=$PROJWORK/lrn044/foundation_models/results/cwang31/
cd $PROJ_MAXIE_TRAIN
```

Now, copy and paste them

```bash
cp -rv $MAXIE_TRAIN/*config $PROJ_MAXIE_TRAIN/
cp -rv $MAXIE_TRAIN/*py     $PROJ_MAXIE_TRAIN/
```

### Generate JSON files required for training

```
# Optional, if you don't have it set up
export PROJ_MAXIE_PREPROC=$PROJWORK/lrn044/foundation_models/results/cwang31/  # Change it to your preprocess location

# Save a list of yamls to a `dataset.yaml` file
ls $PROJ_MAXIE_PREPROC/outputs/*.yaml | awk '{print "- " $0}' > $PROJ_MAXIE_TRAIN/dataset.yaml
```

```bash
python generate_dataset_in_json.py --yaml $PROJ_MAXIE_TRAIN/dataset.yaml --num_cpus 80 --dir_output $PROJ_MAXIE_TRAIN/experiments/datasets --train_frac 0.8 --seed 42
```

### Launch a job

```bash
python launch_job.py train_config.dataset.path_train=experiments/datasets/dataset.train.json train_config.dataset.path_eval=experiments/datasets/dataset.eval.json train_config.misc.num_gpus=6 train_config.dataset.batch_size=1 train_config.dataset.num_workers=1 train_config.loss.grad_accum_steps=2 train_config.model.name=facebook/vit-mae-base train_config.dataset.seg_size=4 train_config.misc.max_eval_iter=4 train_config.misc.data_dump_on=false train_config.logging.filename_prefix=reduced_batch job=reduced_batch bsub_config.ipc_workers=2 bsub_config.qos=batch bsub_config.walltime=2:00 bsub_config.num_nodes=45 bsub_config.trainer=train.fsdp.py auto_submit=true bsub_config.num_cpus_for_client=4 
```

This should generate a yaml at `$MAXIE_TRAIN/experiments/yaml/my_job.yaml` and a
bsub script at `$MAXIE_TRAIN/experiments/bsub/my_job.bsub`.

You can submit the job with `bsub $MAXIE_TRAIN/experiments/bsub/my_job.bsub`.

Also, if you pass `auto_submit=true` in the `python launch_job.py ...` command,
it should automatically submit the job for you.
