# Reproduction

Install dependencies and the spaCy English model:

```powershell
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the faithful appendix reconstruction:

```powershell
python reproduce/original_pipeline.py
```

The script expects the official UCI TSV files in `data/raw/` and writes all generated
metrics, plots, logs, and environment information under `results/` and `figures/`.
