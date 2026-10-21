# Data

## UCI (Portugal), included

`dataset.csv` is the UCI *Predict Students' Dropout and Academic Success* dataset
(Realinho, Vieira Martins, Machado and Baptista, 2021; DOI
[10.24432/C5MC89](https://doi.org/10.24432/C5MC89)), redistributed unchanged
under its [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence.
This copy is the comma-separated version distributed on Kaggle; the semicolon-separated
UCI file also works with `src/uci/data.py`.

Please cite the dataset and the data descriptor:

> Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student
> dropout and academic success. *Data*, 7(11), 146. https://doi.org/10.3390/data7110146

## UPV (Spain), download separately

Used for the external validation (`src/upv/`). Not included, because the raw
files are large. Download the three CSV files from Zenodo,
DOI [10.5281/zenodo.17239943](https://doi.org/10.5281/zenodo.17239943)
(CC BY 4.0), and put them here:

```
data/upv/dataset_2018_hash.csv
data/upv/dataset_2021_hash.csv
data/upv/dataset_2022_hash.csv
```

`run_all.py` checks for them and skips the UPV step, with a message, if they
are missing. Their SHA-256 checksums are recorded in `results/run_manifest.json`.

Please cite the dataset and its data descriptor:

> Igualde-Sáez, A., Maheut, J., Garcia-Sabater, J. P., Marin-Garcia, J. A., Turro, C.,
> Despujol Zabala, I., Puche Garcia, S., Alonso, M., Benlloch-Dualde, J. V., &
> Soriano Jiménez, P. P. (2025). *StudentDropoutDataset* (v1) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.17239943
>
> Igualde-Sáez, A., et al. (2025). University student dropout: a longitudinal dataset
> of demographic, socioeconomic, and academic indicators. *Data*, 10(10), 162.
