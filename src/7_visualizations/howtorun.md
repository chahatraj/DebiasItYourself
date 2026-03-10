# How To Run

Run from this folder:

```bash
cd /scratch/craj/diy/src/7_visualizations
```

## 1) Full pipeline (baselines + finetuning + combined)

```bash
python run_plot_pipeline.py
```

## 2) Only build finetuning CSVs

```bash
python run_plot_pipeline.py --prepare-only
```

## 3) Pipeline by mode

Baselines only:

```bash
python run_plot_pipeline.py --mode baselines
```

Finetuning only:

```bash
python run_plot_pipeline.py --mode finetuning
```

Baselines + finetuning plots only:

```bash
python run_plot_pipeline.py --mode combined
```

## 4) Limit datasets in pipeline mode

```bash
python run_plot_pipeline.py --mode all --datasets crowspairs stereoset
```

## 5) Single-dataset plotting mode

Baselines for one dataset:

```bash
python run_plot_pipeline.py --dataset bbq --mode baselines
```

## 6) See all options

```bash
python run_plot_pipeline.py --help
```
