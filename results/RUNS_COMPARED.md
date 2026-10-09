# numj verdicts across independent runs

Runs: results, results/run2. Cell = `X / numj` median ratio per run, then the verdict. **robust** = same verdict in every run.

| kernel | shape | vs numpy-reuse | vs numpy | vs java-naive | vs java-blocked |
|---|---|---|---|---|---|
| sqdist | [16] | 97.34× / 86.80× **win** (robust) | 233.02× / 217.82× **win** (robust) | 0.70× / 0.65× **lose** (robust) | 1.05× / 1.07× **tie** (robust) |
| sumsq_muladd | [16] | 98.30× / 94.06× **win** (robust) | 222.37× / 176.88× **win** (robust) | 0.63× / 0.66× **lose** (robust) | 1.11× / 1.25× **tie** (robust) |
| sqdist | [1000] | 15.03× / 13.25× **win** (robust) | 35.79× / 33.41× **win** (robust) | 5.98× / 5.49× **win** (robust) | 4.72× / 4.33× **win** (robust) |
| sumsq_muladd | [1000] | 16.10× / 14.93× **win** (robust) | 32.87× / 31.20× **win** (robust) | 4.34× / 4.22× **win** (robust) | 5.37× / 5.17× **win** (robust) |
| sqdist | [100000] | 2.89× / 3.08× **win** (robust) | 31.91× / 35.66× **win** (robust) | 2.91× / 2.98× **win** (robust) | 2.43× / 2.63× **win** (robust) |
| sumsq_muladd | [100000] | 2.68× / 2.46× **win** (robust) | 17.71× / 16.74× **win** (robust) | 1.58× / 1.62× **win** (robust) | 2.29× / 4.70× **win** (robust) |
| sqdist | [10000000] | 2.55× / 2.18× **win** (robust) | 5.76× / 5.43× **win** (robust) | 1.13× / 1.47× **tie** (robust) | 1.08× / 1.57× mixed |
| sumsq_muladd | [10000000] | 3.03× / 0.30× mixed | 6.59× / 0.73× mixed | 0.98× / 0.13× mixed | 0.99× / 0.15× **tie** (robust) |
| normalize_rows | [4×4] | 115.99× / 86.77× **win** (robust) | 139.11× / 98.70× **win** (robust) | 0.56× / 0.74× **lose** (robust) | 0.67× / 0.62× **lose** (robust) |
| normalize_rows | [100×10] | 2.88× / 1.96× **win** (robust) | 3.31× / 2.18× **win** (robust) | 0.38× / 0.34× **lose** (robust) | 0.39× / 0.35× **lose** (robust) |
| normalize_rows | [1000×100] | 2.29× / 2.29× **win** (robust) | 4.90× / 2.80× **win** (robust) | 1.36× / 1.12× **win** (robust) | 1.45× / 2.23× **win** (robust) |
| normalize_rows | [100000×100] | 2.62× / 2.27× **win** (robust) | 4.55× / 3.70× **win** (robust) | 1.02× / 0.95× **tie** (robust) | 1.05× / 1.15× **tie** (robust) |
| sqdist_rows | [1000×16] | 4.89× / 3.80× **win** (robust) | 4.14× / 3.57× **win** (robust) | 1.11× / 1.15× **tie** (robust) | 2.87× / 1.90× **win** (robust) |
| sqdist_rows | [10000×100] | 5.86× / 6.49× **win** (robust) | 17.83× / 21.39× **win** (robust) | 2.17× / 1.59× **win** (robust) | 1.94× / 2.10× **win** (robust) |
| sqdist_rows | [100000×100] | 2.96× / 5.45× **win** (robust) | 5.24× / 10.22× **win** (robust) | 0.89× / 1.32× mixed | 0.91× / 1.44× mixed |

## Tally

- vs **numpy-reuse**: mixed: 1, win: 14
- vs **numpy**: mixed: 1, win: 14
- vs **java-naive**: lose: 4, mixed: 2, tie: 3, win: 6
- vs **java-blocked**: lose: 2, mixed: 2, tie: 4, win: 7
