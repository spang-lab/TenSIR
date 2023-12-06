import os
from enum import Enum
from os.path import join

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import typer

import tensir
from paths import AUSTRIA_DATA_CACHE_PATH, RESULTS_DIR
from tensir.uniformization.derivative import log_likelihood_gradient_dataset
from tensir.uniformization.forward import log_likelihood_dataset


class Sampler(str, Enum):
    nuts = "nuts"
    mh = "mh"


class _TenSIRLogLikelihoodGrad(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, data: np.array, n_threads: int = 1) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._n_days = data.shape[0]
        self._n_threads = n_threads

    def perform(self, node, inputs, outputs):
        (theta,) = inputs
        cast_theta = np.asarray(theta, dtype=np.float64)
        grads, _ = log_likelihood_gradient_dataset(self._data, cast_theta, theta_is_log=True, threads=self._n_threads)
        outputs[0][0] = grads * self._n_days


class TenSIRLogLikelihood(pt.Op):
    """A wrapper around the TenSIR loglikelihood, so that
    it can be used in PyMC models.

    This operation expects the log Theta vector of shape (2, ).
    """

    itypes = [pt.dvector]  # (2, )
    otypes = [pt.dscalar]  # scalar, the loglikelihood

    def __init__(self, data: np.ndarray, n_threads) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._n_days = data.shape[0]
        self._n_threads = n_threads
        self._gradop = _TenSIRLogLikelihoodGrad(data, n_threads)

    def _loglikelihood(self, theta: np.ndarray) -> float:
        """Calculates the log-likelihood function."""
        cast_theta = np.asarray(theta, dtype=np.float64)
        score = log_likelihood_dataset(self._data, cast_theta, theta_is_log=True, threads=self._n_threads)
        return float(score) * self._n_days

    def perform(self, node, inputs, outputs):
        """This is the method which is called by the operation.

        It calculates the loglikelihood.

        Note:
            The arguments and the output are PyTensor variables.
            See `_loglikelihood` method for the "true" implementation
        """
        (theta,) = inputs  # Unwrap the inputs

        # Call the log-likelihood function
        logl = self._loglikelihood(theta)

        outputs[0][0] = np.array(logl)  # Wrap the log-likelihood into output

    def grad(self, inputs, g):
        (theta,) = inputs
        tangent_vector = g[0]

        return [tangent_vector * self._gradop(theta)]


def main(
        sampler: Sampler = typer.Option(..., help="Select a sampling algorithm."),
        month: int = typer.Option(
            ..., help="Select a specific month (3-8) from the dataset as the data for the simulation."),
        run: int = typer.Option(
            ..., help="Specify a run. This value only influences the seed (via 'seed = month * 1000 + run') to keep "
                      "the results reproducible."),
        tune: int = typer.Option(..., help="Number of points to tune for."),
        draws: int = typer.Option(..., help="Number of points to create after tuning."),
        threads: int = typer.Option(
            ..., help="Number of concurrent threads to use. Set to 1 to disable concurrency. Vanishing returns for "
                      "threads > 60 (because of how the code is parallelized: 2 parameters for ~30 days)."),
):
    """
    Generate/sample points from the posterior of the SIR model.
    """
    seed = month * 1000 + run

    data_sir = tensir.data.covid19_austria_daily(start=f"2020-{month:02d}-01",
                                                 end=f"2020-{month + 1:02d}-01",
                                                 cache_path=AUSTRIA_DATA_CACHE_PATH)

    np.random.seed(seed)

    tensir_ll = TenSIRLogLikelihood(data_sir, threads)
    with pm.Model():
        theta = pm.Uniform("theta", upper=0, lower=np.log(0.01), shape=(2,))
        pm.Potential("loglikelihood", tensir_ll(theta))
        if sampler == "nuts":
            step = pm.NUTS(max_treedepth=2, early_max_treedepth=2)
        elif sampler == "mh":
            step = pm.Metropolis()
        else:
            raise ValueError(f"Illegal sampler '{sampler}'")
        idata = pm.sample(chains=1, tune=tune, draws=draws, cores=1, discard_tuned_samples=False, step=step)

    out_dir = join(RESULTS_DIR, f"{sampler}-points")
    os.makedirs(out_dir, exist_ok=True)
    idata.to_netcdf(join(out_dir, f"{sampler}-points-{month:02d}-{run:02d}.nc"))


if __name__ == '__main__':
    typer.run(main)
