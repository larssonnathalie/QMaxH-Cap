from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
import qiskit

import numpy as np


def makeCostHamiltonian(qubo:QuadraticProgram, prints=True):
    return []

def makeAnsatz(Hc, prints=True)->QAOAAnsatz:
    # use QAOAAnsatz
    return []

def hardwareSetup(prints=True):
    return []

def transpileAnsatz(ansatz:QAOAAnsatz, backend, prints=True): 
    return []

def findParameters(initial_betas, initial_gammas, quantumCircuit, prints=True, plots=True):
    n_layers = len(initial_gammas)
    # Using estimator primitive
    # use parallelism job-mode
    return []


def sampleSolutions(bestParameters, prints=True, plots=True):
    # Using sampler primitive
    # Use single job-mode
    # returns distribution of solutions
    return []