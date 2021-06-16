# code/report

Store here the scripts that we use to generate tables, figures,
etc. for a report about the benchmark results.

Perhaps some notebooks as well.

## description of the scripts:

### Figure 4: ECFDs

Figures 4a and 4b from Section 5.4 of the article can be generated as follows:

```
python3 figure_4a.py --solid-lines gpy_mle0121 gpy_mle0143 gpy_mle0133
python3 figure_4b.py --solid-lines gpy_mle0131 gpy_mle0132 gpy_mle0133
```

TODO: Merge/simplify these two scripts, since they are almost identical

## Figure 5: AUC vs runtime for restart & multistart

Figures 5a et 5b from Section 5.4 of the article can be generated as follows:

```
python3 figure_5a_auc_restart.py
python3 figure_5b_auc_multistart.py
```

### tmp

- `plot_error_cdf_benchmark.py` : plots the ECDFs of NLL differences (error)

Syntax:
```
python3 plot_error_cdf_benchmark.py --solid-lines gpy_mle0000 stk261_matR2019b_mle1000 --dashed-lines gpy_mle0010 gpy_mle001
```

Used commands:
```
python3 plot_error_cdf_benchmark.py --solid-lines gpy_mle0021 gpy_mle0022 gpy_mle0023 --dashed-lines gpy_mle0031 gpy_mle0032 gpy_mle0033
```
- `error_scatterplot.py` : generates scatterplots of NLL differences (error)

Syntax:
```
python error_scatterplot.py 1 gpy_mle0021 gpy_mle0031
```
- `scatterplot.py` : generates scatterplot for LOO predictions

Syntax:
```
python3 scatterplot.py 2 gpy_mle0133 gpy_mle3021 g10mod 10d
```
- `nll_boxplot.py` : generates boxplots(for multiple output functions)/histogram(for single output function) for NLL differences of *default* & *improved* 
as obtained with LOO

Syntax:
```
python3 nll_boxplot.py 2 gpy_mle3022 gpy_mle3021 borehole 20d
```
- `loo_boxplot.py` : generates boxplots for range(lengthscale) values for LOO

Syntax:
```
python3 loo_boxplot.py 2 gpy_mle0133 g10 3d f_1 
```
- `multiple_loo_boxplot.py` : generates boxplots for range(lengthscale) values of *default* & *improved* as obtained with LOO

Syntax:
```
python3 multiple_loo_boxplot.py 2 gpy_mle3022 gpy_mle3021 borehole 20d f_1 
```
corresponding notebook : `multiple_boxplot.ipynb`
- `restart_area.py` : script for all restart experiments. 

TODO: not organized!

Syntax:
```
python restart_area.py <method_name>^^
```

^^multiple `<method_name>` is given as inputs when we are interested in plotting the restart area for schemes like `[n_1, p]`
