# saferGPMLE/expB

## What you will find in here

This directory provides a framework for producing the other results of
  the article, and in particular the results presented in Section 5.

## Datasets

Datasets are supposed to be located under `./datasets`.

They are actually copied from another repo called test-functions.  When new
datasets are added to `test-functions/data/doe`, the collection datasets in
this repo can be updated using `python3 update_datasets.py`.

## Benchmarks

### First benchmark

For each dataset, find the MLE for the constant-mean Matérn-5/2 anisotropic
model.

The results for this benchmark are stored in the folder ./results/bench1.

### Second benchmark

For each n-dataset, find the MLE for each possible (n-1)-dataset that can be
extracted from it.

Same model as in the first benchmark.

The results for this benchmark are stored in the folder ./results/bench2.

## Method names and numbers

MLE methods will be given names of the form `BLAH1_BLAH2_mleWXYZ` where
* the prefix `BLAH1` indicates which toolbox (package, module, etc.) has been
  used, optionnally with some information about the version number,
* `BLAH2` optionnally provide some additional information about the software
  environment,
* the suffix WXYZ specifies a particular method.

Inside the suffix WXYZ, the first digit indicates the toolbox:
* `0` for GPy,
* ... (other will be added progressively).

Remark: there is a little bit of redundency in this convention since the toolbox
is actually encoded twice, but its seems better than having a single method
number such as `007` refer to different methods depending on the underlying
toolbox.


## Running a benchmark

Syntax:
```
python3 code/bench/py/launcher.py type toolbox i input_data_path output_data_path informations
```

Example:
```
python3 code/bench/py/launcher.py simple gpy 001 datasets results/bench1/data
```

Input arguments explained:
* `type`: indicates whether you want to test estimation on each
  dataset (i.e. 'simple') or on each of the (n-1) datasets obtained by
  removing one point from the original dataset (i.e. 'loo').
* `toolbox`: toolbox that you want to use (gpy for now).
* `i`: index of the method you want to test.
* `intput_data_path`: folder where the datasets are stored.
* `output_data_path`: folder where you want to write the results.
* `informations`: optional, could be provided to specify BLAH2 (see
  Methods names and numbers section).

## Reports

### Plot absolute difference NLL CDF

You can plot the empirical CDF of the NLL differences between the
results of one method and the best known values.

The script is ./code/report/plot_error_cdf_benchmark.py and can be
used like this :

```
python3 plot_error_cdf_benchmark.py --solid-lines gpy_mle0000 stk261_matR2019b_mle1000 --dashed-lines gpy_mle0010 gpy_mle001
```

The script has a sanity check. It will fail if it's run for methods
that don't have exactly the same (dataset, output) entries set. For
example, this message :

```
Traceback (most recent call last):
  File "plot_error_cdf_benchmark.py", line 57, in <module>
    df_pivot[df_pivot['cost'].isnull().any(1)].iloc[0]))
ValueError: Comparison not done on same datasets. First example : 
       optim_type              
cost  best_known                  40.389083
      gpy_mle0000                       NaN
      gpy_mle0010                       NaN
      gpy_mle0011                       NaN
      stk261_matR2019b_mle1000    87.222710
Name: (branin, f, 10d), dtype: float64
```

means that results are missing for methods gpy_mle00{00, 10, 11} for
output 'f' of the size 10d branin dataset.
