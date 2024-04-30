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
python launch_job.py train_config.dataset.path_train=$MAXIE_TRAIN/experiments/datasets/dataset.json train_config.dataset.path_eval=$MAXIE_TRAIN/experiments/datasets/dataset.json train_config.misc.num_gpus=6 train_config.misc.num_nodes=100 train_config.model.name=facebook/vit-mae-huge job=my_job
```

This should generate a yaml at `$MAXIE_TRAIN/experiments/yaml/my_job.yaml` and a
bsub script at `$MAXIE_TRAIN/experiments/bsub/my_job.bsub`.

You can submit the job with `bsub $MAXIE_TRAIN/experiments/bsub/my_job.bsub`.

Also, if you pass `auto_submit=true` in the `python launch_job.py ...` command,
it should automatically submit the job for you.
