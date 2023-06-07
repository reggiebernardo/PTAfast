Python codes for studying the influence of the cosmic variance on SGWB analysis with PTA correlations data ([2304.07040](https://arxiv.org/abs/2304.07040)).

#### Main notebooks: results
> correlations_ng.ipynb - PTA correlations analysis with real data <br />
> correlations_mock.ipynb - PTA correlations analysis with mock data

#### python and yaml files
> correlations_yyy.py - likelihoods setup <br />
> evidence_zzz.py - Bayesian evidence calculations via `ultranest` <br />
> ____.yaml - likelihoods and priors for `cobaya`

Run as e.g., `mpirun -n 12 cobaya-run hd_unc2.yaml` and `mpirun -n 12 python evidence_ng_hd.py`

