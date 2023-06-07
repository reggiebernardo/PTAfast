import numpy as np
from mpi4py import MPI
import ultranest
import logging

# Set the logging level to a higher level, such as WARNING or ERROR
logging.getLogger("ultranest").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

# import likelihoods using NANOGrav data

from correlations_another import ll_hd_unc2 as ll_DataplusCV
from correlations import ll_hd_unc2 as ll_HDplusCV
from correlations import ll_hd_unc0 as ll_HD0

def A2_unit_transform(u):
    return 10 * u  # Map from unit cube [0, 1] to [0, 10]

def run_ultranest(loglike, likelihood_name):
    # Set up the UltraNest analysis
    sampler = ultranest.ReactiveNestedSampler(["A2"], loglike)

    # Run the UltraNest analysis with additional MCMC steps or convergence criterion
    max_num_improvement_loops = 1000  # Maximum number of iterations without improvement
    tolerance = 0.1  # Convergence criterion (fractional change in evidence)
    sampler.run(max_num_improvement_loops=max_num_improvement_loops, dlogz=tolerance)
    result = sampler.results
    print('A2 =', A2_unit_transform(result['posterior']['mean'][0]), \
          '+/-', A2_unit_transform(result['posterior']['stdev'][0]))
    print('logZ =', result['logz'], '+/-', result['logzerr'])
    print()

    # Save the results to a text file
    filename = f'result_ng_{likelihood_name}.txt'
    np.savetxt(filename, np.array(list(result.items()), dtype=object), delimiter='=', fmt='%s')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get LogZ for HD
def loglike_HD(params):
    A2 = A2_unit_transform(params[0])
    return ll_HD0(A2)

run_ultranest(loglike_HD, 'HD0')

# Get LogZ for HD + CV
def loglike_HDplusCV(params):
    A2 = A2_unit_transform(params[0])
    return ll_HDplusCV(A2)

run_ultranest(loglike_HDplusCV, 'HDplusCV')

# Get LogZ for Data + CV
def loglike_DataplusCV(params):
    A2 = A2_unit_transform(params[0])
    return ll_DataplusCV(A2)

run_ultranest(loglike_DataplusCV, 'DataplusCV')

MPI.Finalize()
