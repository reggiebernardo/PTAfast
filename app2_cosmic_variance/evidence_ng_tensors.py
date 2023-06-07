import numpy as np
from mpi4py import MPI
import ultranest
import logging

# Set the logging level to a higher level, such as WARNING or ERROR
logging.getLogger("ultranest").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

# import likelihoods using NANOGrav data

from correlations_another import ll_tensors_unc2_TCV as ll_DataplusTCV
from correlations import ll_tensors_unc2 as ll_TplusCV
from correlations import ll_tensors_unc0 as ll_T0

def A2_unit_transform(u):
    return 20 * u  # Map from unit cube [0, 1] to [0, 10]

def run_ultranest(loglike, likelihood_name):
    # Set up the UltraNest analysis
    sampler = ultranest.ReactiveNestedSampler(["A2", "v"], loglike)

    # Run the UltraNest analysis with additional MCMC steps or convergence criterion
    max_num_improvement_loops = 1000  # Maximum number of iterations without improvement
    tolerance = 0.1  # Convergence criterion (fractional change in evidence)
    sampler.run(max_num_improvement_loops=max_num_improvement_loops, dlogz=tolerance)
    result = sampler.results
    print('A2 =', A2_unit_transform(result['posterior']['mean'][0]), \
          '+/-', A2_unit_transform(result['posterior']['stdev'][0]))
    print('v :', result['posterior'])
    print('logZ =', result['logz'], '+/-', result['logzerr'])
    print()

    # Save the results to a text file
    filename = f'result_ng_{likelihood_name}.txt'
    np.savetxt(filename, np.array(list(result.items()), dtype=object), delimiter='=', fmt='%s')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get LogZ for Tensors
def loglike_T(params):
    A2, v = A2_unit_transform(params[0]), params[1]
    return ll_T0(A2, v)

run_ultranest(loglike_T, 'T0')

# Get LogZ for Data + Tensor CV
def loglike_DataplusTCV(params):
    A2, v = A2_unit_transform(params[0]), params[1]
    return ll_DataplusTCV(A2, v)

run_ultranest(loglike_DataplusTCV, 'DataplusTCV')

# Get LogZ for Tensors + CV
def loglike_TplusCV(params):
    A2, v = A2_unit_transform(params[0]), params[1]
    return ll_TplusCV(A2, v)

run_ultranest(loglike_TplusCV, 'TplusCV')

MPI.Finalize()
