import numpy as np
from mpi4py import MPI
import ultranest
import logging

# Set the logging level to a higher level, such as WARNING or ERROR
logging.getLogger("ultranest").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

from correlations import ll_mon
from correlations import ll_GRN

def A2_unit_transform(u):
    return 10 * u  # Map from unit cube [0, 1] to [0, 2]

def run_ultranest(loglike, likelihood_name, paramlist):
    # Set up the UltraNest analysis
    sampler = ultranest.ReactiveNestedSampler(paramlist, loglike)

    # Run the UltraNest analysis with additional MCMC steps or convergence criterion
    max_num_improvement_loops = 1000  # Maximum number of iterations without improvement
    tolerance = 0.1  # Convergence criterion (fractional change in evidence)
    frac_remain = 0.5  # Fraction of remaining evidence below which the algorithm terminates
    sampler.run(max_num_improvement_loops=max_num_improvement_loops, \
                dlogz=tolerance, \
                frac_remain=frac_remain)
    result = sampler.results
    print('logZ =', result['logz'], '+/-', result['logzerr'])
    print()

    # Save the results to a text file
    filename = f'result_ng_{likelihood_name}.txt'
    np.savetxt(filename, np.array(list(result.items()), dtype=object), delimiter='=', fmt='%s')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Get LogZ for mon
def loglike_mon(params):
    A2 = A2_unit_transform(params[0])
    return ll_mon(A2)

run_ultranest(loglike_mon, 'mon', ["A2"])

# Get LogZ GRN
def linear_map(u):
    a = 0.01
    b = 3.00
    return a + (b - a) * u

def loglike_GRN(params):
    A2 = A2_unit_transform(params[0])
    sigma = linear_map(params[1])
    return ll_GRN(A2, sigma)

run_ultranest(loglike_GRN, 'grn', ["A2", "sigma"])

MPI.Finalize()
