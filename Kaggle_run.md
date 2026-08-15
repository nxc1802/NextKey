# Run official training on Kaggle

Enable a Kaggle **GPU** accelerator and attach a dataset that contains a
`processed/jdwr_v1/manifest.json` file. The runner automatically searches
`data/processed` first, then every attached Kaggle input, and copies the first
matching `processed/` directory when necessary. No dataset path or training
parameters need to be supplied.

```bash
!if [ ! -d /kaggle/working/NextKey/.git ]; then git clone https://github.com/nxc1802/NextKey.git /kaggle/working/NextKey; fi
!git -C /kaggle/working/NextKey pull --ff-only origin main
!cd /kaggle/working/NextKey && python scripts/run_kaggle_training.py
```
