# v0.9 M0 applications (sources only)

Developed against installed TensorLang 0.8.0 using public APIs only,
outside the TensorLang source tree.

Reproduce (from the app directory, with tensorlang installed):

```
python data/generate.py data          # all apps (C: ~512MB zarr)
tensorlang run main.tl                # run the application
python -m pytest tests/ -q           # application tests
python reference/*.py                 # independent native reference
```
