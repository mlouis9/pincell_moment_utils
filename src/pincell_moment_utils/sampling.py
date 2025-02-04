import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import emcee
from typing import List, Callable

def _global_log_prob(theta, pdf, domain):
    """
    A top-level log-prob function that can be pickled for multiprocessing.
    
    Parameters
    ----------
    theta : array_like
        A 3-element array [x, w, E].
    pdf : callable
        The flux function for a given surface, e.g. ReconstructedFlux(...).
    domain : list of 3 tuples
        [(x_min, x_max), (w_min, w_max), (E_min, E_max)].
        
    Returns
    -------
    float
        The log of the PDF (log(flux)), or -np.inf if out-of-bounds or flux<=0.
    """
    x, w, E = theta
    (x_min, x_max), (w_min, w_max), (e_min, e_max) = domain

    # Domain checks
    if not (x_min <= x <= x_max):
        return -np.inf
    if not (w_min <= w <= w_max):
        return -np.inf
    if not (e_min <= E <= e_max):
        return -np.inf
    
    val = pdf(x, w, E)
    if val <= 0:
        return -np.inf
    return np.log(val)

def _global_log_prob_logE(theta, pdf, domain):
    """
    A log-prob function in (x, w, lnE)-space, picklable for multiprocessing.
    """
    x, w, lnE = theta
    (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) = domain

    # Domain checks
    if not (x_min <= x <= x_max):
        return -np.inf
    if not (w_min <= w <= w_max):
        return -np.inf
    if not (lnE_min <= lnE <= lnE_max):
        return -np.inf

    E = np.exp(lnE)
    # pdf(x, w, E) * Jacobian = pdf(x,w,E)*E
    val = pdf(x, w, E)*E

    if val <= 0:
        return -np.inf
    return np.log(val)

def rejection_sampling_3d_parallel(
    pdf: Callable[[float, float, float], float],
    domain: List[List[float]],
    num_samples: int,
    num_workers: int = None,
) -> np.ndarray:
    """
    Perform parallel rejection sampling for a 3-variable flux function pdf(x,w,E),
    assuming you want to sample uniform in x, w, ln(E).
    
    domain: [ (x_min, x_max), (w_min, w_max), (E_min, E_max) ]
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Estimate bounding constant
    M = _estimate_M_uniform(pdf, domain, num_points=40)

    x_bounds, w_bounds, e_bounds = domain

    # Distribute work
    samples_per_worker = [num_samples // num_workers] * num_workers
    for i in range(num_samples % num_workers):
        samples_per_worker[i] += 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _rejection_sampling_worker_uniform,
                pdf,
                x_bounds,
                w_bounds,
                e_bounds,
                n,
                M,
            )
            for n in samples_per_worker
        ]
        results = [f.result() for f in futures]

    return np.concatenate(results, axis=0)

def _rejection_sampling_worker_uniform(
    pdf: Callable[[float, float, float], float],
    x_bounds: List[float],
    w_bounds: List[float],
    e_bounds: List[float],
    num_samples: int,
    M: float,
) -> np.ndarray:
    """
    Uniform rejection-sampling worker in (x, w, lnE) domain.
    """
    x_min, x_max = x_bounds
    w_min, w_max = w_bounds
    e_min, e_max = e_bounds

    # We sample x ~ Uniform(x_min, x_max),
    #            w ~ Uniform(w_min, w_max),
    #            ln(E) ~ Uniform(ln(e_min), ln(e_max)).
    # Then E = exp(ln(E)).
    # The proposal PDF is then 1/( (x_max-x_min)*(w_max-w_min)* ln(e_max/e_min) ) * 1/E
    # But we incorporate that 1/E factor by sampling in ln(E).
    # We'll simply check acceptance with M.

    local_samples = []
    ln_e_min, ln_e_max = np.log(e_min), np.log(e_max)

    while len(local_samples) < num_samples:
        x = np.random.uniform(x_min, x_max)
        w = np.random.uniform(w_min, w_max)
        lnE = np.random.uniform(ln_e_min, ln_e_max)
        E = np.exp(lnE)

        f_val = pdf(x, w, E)
        # proposal_pdf_val = (1/(x_max - x_min))*(1/(w_max - w_min))*(1/(ln_e_max-ln_e_min))
        # but we also need the Jacobian factor 1/E if we were to do it purely in ln(E).
        # We'll handle it by simply comparing f_val to M * proposal_pdf_val * (1/E).
        # Because it's uniform in ln(E), the effective pdf is: 1/((x_range)*(w_range)*(ln_e_range)).
        # The ratio is f_val / [ M * (proposal_pdf_val*(1/E)) ].

        # We'll compute that ratio directly:
        # proposal_pdf_val*(1/E) = 1/[(x_max - x_min)*(w_max - w_min)*(ln_e_max-ln_e_min)*E]
        # So let's store that in a quick variable:
        proposal_val = 1.0 / ((x_max - x_min)*(w_max - w_min)*(ln_e_max - ln_e_min)*E)

        if np.random.rand() < f_val/(M*proposal_val):
            local_samples.append([x, w, E])

    return np.array(local_samples)

def _estimate_M_uniform(
    pdf: Callable[[float, float, float], float],
    domain: List[List[float]],
    num_points=40
) -> float:
    """
    Estimate bounding constant M by randomly sampling in (x, w, lnE).
    domain = [ (x_min, x_max), (w_min, w_max), (E_min, E_max) ].
    """
    x_min, x_max = domain[0]
    w_min, w_max = domain[1]
    e_min, e_max = domain[2]

    ln_e_min, ln_e_max = np.log(e_min), np.log(e_max)

    samples_x = np.random.uniform(x_min, x_max, num_points)
    samples_w = np.random.uniform(w_min, w_max, num_points)
    samples_lnE = np.random.uniform(ln_e_min, ln_e_max, num_points)
    samples_E = np.exp(samples_lnE)

    # Evaluate pdf at each random point
    # The proposal pdf in ln(E)-space is 1/[ (x_range)*(w_range)*(ln_e_range) ], but
    # we also multiply by 1/E in the acceptance ratio. So effectively, the bounding
    # function is M * [ proposal_pdf_val * (1/E) ].
    # We'll store ratio = pdf(x,w,E)/[proposal_pdf_val*(1/E)] and pick the max.

    x_range = (x_max - x_min)
    w_range = (w_max - w_min)
    ln_e_range = (ln_e_max - ln_e_min)

    proposal_pdf_val = 1.0/(x_range*w_range*ln_e_range)
    
    ratios = []
    for i in range(num_points):
        fx = pdf(samples_x[i], samples_w[i], samples_E[i])
        if fx > 0:
            # ratio = fx / [proposal_pdf_val*(1/E)]
            #        = fx * E / proposal_pdf_val
            r = fx * samples_E[i] / proposal_pdf_val
            ratios.append(r)
        else:
            ratios.append(0)

    # Add a small pad
    return 1.1*max(ratios)

def ensemble(
    pdf: Callable[[float, float, float], float],
    domain: List[List[float]],
    N: int,
    burn_in: int = 1000,
    n_walkers: int = 32,
    progress: bool = False,
    num_cores: int = 1
) -> np.ndarray:
    """
    Ensemble sampler in (x, w, E) space returning exactly N total samples.
    """
    (x_min, x_max), (w_min, w_max), (e_min, e_max) = domain

    # Initialize each walker
    p0 = []
    for _ in range(n_walkers):
        x0 = np.random.uniform(x_min, x_max)
        w0 = np.random.uniform(w_min, w_max)
        e0 = np.random.uniform(e_min, e_max)
        p0.append([x0, w0, e0])
    p0 = np.array(p0)

    pool = None
    if num_cores > 1:
        pool = multiprocessing.Pool(processes=num_cores)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        3,
        _global_log_prob,
        args=(pdf, domain),
        pool=pool
    )

    # Burn-in
    state = sampler.run_mcmc(p0, burn_in, progress=progress)
    sampler.reset()

    # Production
    nsteps = (N + n_walkers - 1)//n_walkers
    sampler.run_mcmc(state, nsteps, progress=progress)

    if pool is not None:
        pool.close()
        pool.join()

    chain = sampler.get_chain(flat=True)  # shape: (nwalkers*nsteps, 3)
    if chain.shape[0] > N:
        chain = chain[:N, :]
    return chain

def ensemble_logE(
    pdf: Callable[[float, float, float], float],
    domain: List[List[float]],
    N: int,
    burn_in: int = 1000,
    progress: bool = False,
    n_walkers: int = 32,
    num_cores: int = 1
) -> np.ndarray:
    """
    Ensemble sampler in (x, w, lnE) space, returning exactly N total samples in real space.
    domain = [ (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) ]
    """
    (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) = domain
    
    # Initialize each walker biased a bit toward the lower end of lnE:
    p0 = []
    for _ in range(n_walkers):
        x0 = np.random.uniform(x_min, x_max)
        w0 = np.random.uniform(w_min, w_max)
        # Here, for example, only sample in first half of lnE range:
        lnE0 = lnE_min + 0.5 * (lnE_max - lnE_min) * np.random.rand()
        p0.append([x0, w0, lnE0])
    p0 = np.array(p0)

    # Use a multiprocessing pool if desired:
    pool = None
    if num_cores > 1:
        pool = multiprocessing.Pool(processes=num_cores)

    # Build the sampler in ln(E)-space with the Jacobian factor:
    sampler = emcee.EnsembleSampler(
        n_walkers,
        3,
        _global_log_prob_logE,  # this multiplies pdf by E internally
        args=(pdf, domain),
        pool=pool
    )

    # Burn-in
    state = sampler.run_mcmc(p0, burn_in, progress=progress)
    sampler.reset()

    # Production
    nsteps = (N + n_walkers - 1)//n_walkers
    sampler.run_mcmc(state, nsteps, progress=progress)

    if pool is not None:
        pool.close()
        pool.join()

    chain = sampler.get_chain(flat=True)  # shape: (nwalkers*nsteps, 3)
    if chain.shape[0] > N:
        chain = chain[:N, :]

    # Convert lnE -> E
    out = []
    for (xx, ww, lnE) in chain:
        out.append([xx, ww, np.exp(lnE)])
    return np.array(out)


def metropolis_hastings(
    pdf: Callable[[float,float,float], float],
    domain: List[List[float]],
    N: int,
    x0: np.ndarray = None,
    proposal_std: np.ndarray = None,
    max_init_tries=500
) -> np.ndarray:
    """
    MH in linear E-space.
    """
    dim = 3
    (x_min, x_max), (w_min, w_max), (e_min, e_max) = domain

    # If x0 is None, find a feasible point with flux>0
    if x0 is None:
        for _ in range(max_init_tries):
            trial = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(w_min, w_max),
                np.random.uniform(e_min, e_max)
            ])
            if pdf(*trial) > 0:
                x0 = trial
                break
        if x0 is None:
            raise ValueError("Could not find positive-flux initial guess for MH.")
    else:
        if pdf(*x0) <= 0:
            raise ValueError("User-supplied x0 has zero/negative PDF. Provide better x0.")

    f_current = pdf(*x0)
    if f_current <= 0:
        raise ValueError("Initial guess has zero/negative PDF. Choose a better x0.")

    if proposal_std is None:
        proposal_std = 0.1*np.array([x_max - x_min, w_max - w_min, e_max - e_min])

    samples = []
    x_current = x0.copy()

    for _ in range(N):
        x_proposed = x_current + np.random.normal(scale=proposal_std, size=dim)

        # Domain check
        if not (x_min <= x_proposed[0] <= x_max) or \
           not (w_min <= x_proposed[1] <= w_max) or \
           not (e_min <= x_proposed[2] <= e_max):
            samples.append(x_current.copy())
            continue

        f_proposed = pdf(*x_proposed)
        if f_proposed <= 0:
            samples.append(x_current.copy())
            continue

        alpha = f_proposed / f_current
        if np.random.rand() < alpha:
            x_current = x_proposed
            f_current = f_proposed

        samples.append(x_current.copy())

    return np.array(samples)

def metropolis_hastings_logE(
    pdf: Callable[[float,float,float], float],
    domain: List[List[float]],
    N: int,
    x0: np.ndarray = None,
    proposal_std: np.ndarray = None,
    max_init_tries=500
) -> np.ndarray:
    """
    MH in log(E) space => domain = [ (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) ].
    pdf_logE(point) = pdf(x, w, E)*E   (Jacobian factor).
    """
    dim = 3

    (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) = domain

    def pdf_logE(point):
        xx, ww, lnE = point
        E = np.exp(lnE)
        return pdf(xx, ww, E)*E

    # find feasible init
    if x0 is None:
        for _ in range(max_init_tries):
            trial = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(w_min, w_max),
                np.random.uniform(lnE_min, lnE_max)
            ])
            val = pdf_logE(trial)
            if val > 0:
                x0 = trial
                break
        if x0 is None:
            raise ValueError("Could not find positive flux*g(E) initial guess for MH (logE).")
    else:
        if pdf_logE(x0) <= 0:
            raise ValueError("User-supplied x0 has zero/negative flux*g(E).")

    f_current = pdf_logE(x0)
    if f_current <= 0:
        raise ValueError("Initial guess has zero/negative flux*g(E).")

    if proposal_std is None:
        proposal_std = [
            0.1*(x_max - x_min),
            0.1*(w_max - w_min),
            0.1*(lnE_max - lnE_min)
        ]

    x_current = x0.copy()
    samples = []

    for _ in range(N):
        x_proposed = x_current + np.random.normal(scale=proposal_std, size=dim)

        # domain check
        if not (x_min <= x_proposed[0] <= x_max) or \
           not (w_min <= x_proposed[1] <= w_max) or \
           not (lnE_min <= x_proposed[2] <= lnE_max):
            samples.append(x_current.copy())
            continue

        f_proposed = pdf_logE(x_proposed)
        if f_proposed <= 0:
            samples.append(x_current.copy())
            continue

        alpha = f_proposed / f_current
        if np.random.rand() < alpha:
            x_current = x_proposed
            f_current = f_proposed

        samples.append(x_current.copy())

    # Convert final chain from lnE to E
    final = []
    for (xx, ww, lnE) in samples:
        final.append([xx, ww, np.exp(lnE)])
    return np.array(final)

def sample_surface_flux(
    pdf: Callable[[float, float, float], float],
    domain: List[List[float]],
    N: int,
    method: str = "rejection",
    use_log_energy: bool = False,
    burn_in: int = 1000,
    n_walkers: int = 32,
    progress: bool = False,
    num_cores: int = np.maximum( multiprocessing.cpu_count(), 1)
) -> np.ndarray:
    """
    A single top-level function that dispatches to the various sampling routines.
    domain can be either:
      -  [ (x_min, x_max), (w_min, w_max), (E_min, E_max) ] if not using log-energy
      -  [ (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) ] if use_log_energy = True
    """
    if method == 'rejection':
        if use_log_energy:
            raise NotImplementedError("Rejection sampler here is written for uniform in lnE internally, so set use_log_energy=False.")
        return rejection_sampling_3d_parallel(pdf, domain, N, num_cores)

    elif method == 'ensemble':
        if not use_log_energy:
            return ensemble(pdf, domain, N, burn_in=burn_in, n_walkers=n_walkers, progress=progress, num_cores=num_cores)
        else:
            return ensemble_logE(pdf, domain, N, burn_in=burn_in, n_walkers=n_walkers, progress=progress, num_cores=num_cores)

    elif method == 'metropolis_hastings':
        if not use_log_energy:
            return metropolis_hastings(pdf, domain, N)
        else:
            return metropolis_hastings_logE(pdf, domain, N)
    else:
        raise ValueError(f"Unrecognized method: {method}")
