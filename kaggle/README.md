# Kaggle evidence notebook

The notebook downloads the unmodified official UCI files into `/kaggle/working/data/raw`
using the UCI archive URL, so no Kaggle Dataset attachment is required. If the network is
disabled, place `drugLibTrain_raw.tsv` and `drugLibTest_raw.tsv` in that directory before
running the first data cell. Run all cells, save a notebook version, and keep the generated
`/kaggle/working/results` and `/kaggle/working/figures` files as evidence. The notebook
deliberately uses the UCI Druglib.com files, not a Drugs.com/Kaggle review dataset: those
are different datasets.

Suggested Kaggle notebook title: `Faithful reproduction of Druglib.com side-effect severity thesis`.

The notebook is suitable for GitHub because its Markdown records data provenance,
non-negotiable thesis constraints, corrections, textual analyses (class distribution,
character-length histogram, word cloud, and top terms), outputs, and limitations. Do not
present a Kaggle execution as verification unless it uses the official UCI TSVs and the
saved version shows the output cells.
