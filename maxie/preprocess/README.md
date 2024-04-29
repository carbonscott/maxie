### Locate repository

```bash
export MAXIE=<your maxie location>
export MAXIE_PREPROC="$MAXIE/maxie/preprocess"
```

### Copy configs and python scripts to your working directory

A working directory where you preprocess your data, e.g. mine is

```bash
export PROJ_MAXIE_PREPROC=$PROJWORK/lrn044/foundation_models/results/cwang31/clean_dataset
```

Now, copy and paste them

```bash
cp -rv $MAXIE_PREPROC/*config $PROJ_MAXIE_PREPROC/
cp -rv $MAXIE_PREPROC/*py     $PROJ_MAXIE_PREPROC/
```

### Launch a job

```bash
OMP_NUM_THREADS=1 python launch_job.py qos=batch bsub_num_cpus=15 num_cpus=10 exp=mfxp1002221 num_runs=69 detector_name=Rayonix job=mfxp1002221 file_bsub_job=mfxp1002221.bsub
```

Please refer to the [metadata table](https://docs.google.com/spreadsheets/d/1uWP6sLZXIZFCX_s03CwtvmjtO7pfUzlq/edit?usp=sharing&ouid=110722147140884792923&rtpof=true&sd=true)
to find out `exp`, `num_runs` and `detector_name`.  You can put the value of `exp` into `job` and `file_bsub_job` too.

An example is shown below.

![](../../figures/preprocess_job.png)
