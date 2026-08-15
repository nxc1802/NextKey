# Run official training on Kaggle

Enable a Kaggle **GPU** accelerator and attach the dataset containing the
`processed/` directory. Kaggle mounts attached datasets under
`/kaggle/input/<dataset-slug>`; do not use the Kaggle website path
`datasets/<owner>/<slug>` in a notebook shell command.

```bash
!git clone https://github.com/nxc1802/NextKey.git /kaggle/working/NextKey
!find /kaggle/input -type f -path '*/processed/jdwr_v1/manifest.json' -print
!DATA_ROOT="$(find /kaggle/input -type f -path '*/processed/jdwr_v1/manifest.json' -print -quit | sed 's#/processed/jdwr_v1/manifest.json$##')" && test -n "$DATA_ROOT" && cp -a "$DATA_ROOT/processed" /kaggle/working/NextKey/data/
!cd /kaggle/working/NextKey && test -f data/processed/jdwr_v1/manifest.json && python -m pip install -q -e . && python -c "import torch; assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator'; print(torch.__version__, torch.cuda.get_device_name(0))" && PYTHONPATH=src python -u scripts/train_mvp_chartagger.py --config configs/model/mvp_chartagger_full_kaggle.yaml && PYTHONPATH=src python -u scripts/evaluate_mvp_chartagger.py --config configs/model/mvp_chartagger_full_kaggle.yaml && zip -r /kaggle/working/nextkey-kaggle-results.zip models/checkpoints/mvp-chartagger-full-kaggle.pt models/checkpoints/mvp-chartagger-full-kaggle-vocab.json models/checkpoints/training_history.json experiments/reports/mvp-chartagger-full-kaggle.json experiments/reports/mvp-chartagger-full-kaggle.md experiments/runs/mvp-chartagger-full-kaggle
!ls -lh /kaggle/working/nextkey-kaggle-results.zip
```
