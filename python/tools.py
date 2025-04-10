import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, brentq

# finding the zeros of a function
def zeros(fn, arr, *args):
    """
    Find where a function crosses 0. Returns the zeroes of the function.

    Parameters
    ----------
    fn : function
    arr : array of arguments for function
    *args : any other arguments the function may have
    """

    # the reduced function, with only the argument to be solved for (all other arguments fixed):
    def fn_reduced(array): return fn(array, *args)

    # the array of values of the function:
    fn_arr = fn_reduced(arr)

    # looking where the function changes sign...
    sign_change_arr = np.where(np.logical_or((fn_arr[:-1] < 0.) * (fn_arr[1:] > 0.),
                                             (fn_arr[:-1] > 0.) * (fn_arr[1:] < 0.))
                               )[0]

    # or, just in case, where it is exactly 0!
    exact_zeros_arr = np.where(fn_arr == 0.)[0]

    # defining the array of 0-crossings:
    cross_arr = []

    # first, interpolating between the sign changes
    if len(sign_change_arr) > 0:
        for i in range(len(sign_change_arr)):
            cross_arr.append(
                brentq(fn_reduced, arr[sign_change_arr[i]],
                       arr[sign_change_arr[i] + 1])
            )

    # and then adding those places where it is exactly 0
    if len(exact_zeros_arr) > 0:
        for i in range(len(exact_zeros_arr)):
            cross_arr.append(arr[exact_zeros_arr[i]])

    # sorting the crossings in increasing order:
    cross_arr = np.sort(np.array(cross_arr))

    return cross_arr


# function converting k [s/km] -> [1/Mpc]
def k_vel2comov(self, k_vel, z, units='1/Mpc'):
    """
    k : wavenumber in velocity units [s/km]
    z : redshift
    units : units for the output comoving wavenumber, either ('1/Mpc' or 'h/Mpc')
    """
    
    Hz = 100*self.h()*(self.Hubble(z)/self.Hubble(0))
    k_com = k_vel*(Hz/(1+z))
    
    if units == '1/Mpc':
        return k_com
        
    elif units == 'h/Mpc':
        return k_com/self.h()
        
    else:
        raise ValueError("The allowed units for k are '1/Mpc' and 'h/Mpc' only.")


# various function templates
def pl_fn(k, kp, D, n):
    return D*(k/kp)**(n+3)

def log2_fn(k, tau):
    return (np.log(k*tau))**2

def plateau(k, a, b, g ,d):
    return ((1-d)*(1 + (a*k)**b)**g + d)

def osc_fn(k, kp, A, X, phi):
    return (1 + A*np.sin((k-kp)*X + phi))

def damp_osc_fn(k, kp, A, X, phi, km, sig, p):
    
    # arg1 = -(((k - km)/sig)**2)**(p/2)
    # arg2 = 0*(((kp - km)/sig)**2)**(p/2)
    # arg = arg1 + arg2
    # expo = np.exp(arg)
    
    expo = np.exp(-(np.abs((k - km)/sig)**p))
    expop = np.exp(-(np.abs((kp - km)/sig)**p))
    
    return (1 + A*(expo/expop)*np.sin((k-kp)*X + phi))


# Analytic formulas for P(k)
def pk_pl(k, kp, D, n):
    
    pl = pl_fn(k, kp, D, n)
    
    return pl

def pk_pl_log(k, kp, D, n, tau):
    
    pl = pl_fn(k, kp, D, n)
    log = log2_fn(k, tau)/log2_fn(kp, tau)
    
    return pl*log

def pk_pl_log_sup(k, kp, D, n, tau, a, b, g, d):
    
    pl = pl_fn(k, kp, D, n)
    log = log2_fn(k, tau)/log2_fn(kp, tau)
    sup = plateau(k, a, b, g, d)# TODO: now D will not be the amplitude at kp!
    
    return pl*log*sup

def pk_pl_log_sup_osc(k, kp, D, n, tau, a, b, g, d, A, X, phi):
    
    pl = pl_fn(k, kp, D, n)
    log = log2_fn(k, tau)/log2_fn(kp, tau)
    sup = plateau(k, a, b, g, d)# TODO: now D will not be the amplitude at kp!
    osc = osc_fn(k, kp, A, X, phi)
    
    return pl*log*sup*osc

def pk_pl_log_sup_damposc(k, kp, D, n, tau, a, b, g, d, A, X, phi, km, sig, p):
    
    pl = pl_fn(k, kp, D, n)
    log = log2_fn(k, tau)/log2_fn(kp, tau)
    sup = plateau(k, a, b, g, d)# TODO: now D will not be the amplitude at kp!
    damposc = damp_osc_fn(k, kp, A, X, phi, km, sig, p)
    
    return pl*log*sup*damposc



def find_pk_fit(self, kbounds, kstar, zstar, model='power_law', Npts=201):
    """
    kbounds : [s/km]
    kstar : [s/km]
    zstar : 
    model : ('power_law', 'power_law_sup', or 'power_law_sine')
    """
    
    # TODO: self is argument here, fix!
    k_lo = k_vel2comov(self, kbounds[0], zstar, units='1/Mpc') # TODO: check: conversion at zstar!
    k_hi = k_vel2comov(self, kbounds[1], zstar, units='1/Mpc') # TODO: check: conversion at zstar!
    k_pivot = k_vel2comov(self, kstar, zstar, units='1/Mpc')
    
    k_arr = np.logspace(np.log10(k_lo), np.log10(k_hi), Npts)
    pk_arr = np.array([self.pk_lin(k, zstar) for k in k_arr])
    dpk_arr = (k_arr**3)*pk_arr / (2. * np.pi**2.)
    
    if model == 'power_law':
        
        def to_fit(Lk, D, n):
            return np.log10(pk_pl(10**Lk, k_pivot, D, n))
        
        pars, _ = curve_fit(to_fit, np.log10(k_arr), np.log10(dpk_arr), [0.3, -2.3])
        
        return pars
    
    if model == 'power_law_log':
        
        keq = self.k_eq()
        
        def to_fit(Lk, D, n, ft):
            return np.log10(pk_pl_log(10**Lk, k_pivot, D, n, (10**ft)/keq))
        
        pars, _ = curve_fit(to_fit, np.log10(k_arr), np.log10(dpk_arr), [0.3, -2.3, 0], bounds=([0, -5, -2], [10, 5, 2]))
        
        # rescaling dimensionful parameters
        pars[2] = (10**pars[2])/keq
        
        return pars
    
    elif model == 'power_law_log_sup':
        
        keq = self.k_eq()
        tau_drag_twin = self.tau_d_twin()
        delta = 1.-2.*self.r_all_twin()
        
        def to_fit(Lk, D, n, ft, fa, b, g, d):
            return np.log10(pk_pl_log_sup(10**Lk, k_pivot, D, n, (10**ft)/keq, (10**fa)*tau_drag_twin, b, g, d))
        
        # D, n, tau, a, b, g, d
        ref_vals = [0.3, -2.3, 0, 0, 1, -1, delta]
        lower_bounds = [0, -5, -2, -2, 0, -10, 0]
        upper_bounds = [10, 5, 2, 2, 10, 0, 1]
        
        pars, _ = curve_fit(to_fit, np.log10(k_arr), np.log10(dpk_arr), ref_vals, bounds=(lower_bounds, upper_bounds))
        
        # rescaling dimensionful parameters
        pars[2] = (10**pars[2])/keq
        pars[3] = (10**pars[3])*tau_drag_twin
        
        return pars
        
    
    elif model == 'power_law_log_sup_sine':
        
        keq = self.k_eq()
        tau_drag_twin = self.tau_d_twin()
        delta = 1.-2.*self.r_all_twin()
        
        # fitting first to the spectrum without oscillations, in order to get a good guess
        def to_fit(Lk, D, n, ft, fa, b, g, d):
            return np.log10(pk_pl_log_sup(10**Lk, k_pivot, D, n, (10**ft)/keq, (10**fa)*tau_drag_twin, b, g, d))
        
        # D, n, tau, a, b, g, d
        ref_vals = [0.3, -2.3, 0, 0, 1, -1, delta]
        lower_bounds = [0, -5, -2, -2, 0, -10, 0]
        upper_bounds = [10, 5, 2, 2, 10, 0, 1]
        
        pars1, _ = curve_fit(to_fit, np.log10(k_arr), np.log10(dpk_arr), ref_vals, bounds=(lower_bounds, upper_bounds))
        
        # defining new y-array, only oscillations
        new_yarr = dpk_arr/(10**to_fit(np.log10(k_arr), *pars1))
        
        # interpolation of oscillations
        def yosc(k):
            temp_fn = interp1d(np.log10(k_arr), new_yarr - 1)
            return temp_fn(np.log10(k))
        
        # estimate of frequency
        X_guess = 2.*np.pi/(2*np.mean(np.diff(zeros(yosc, k_arr))))
        
        # defining new function, with oscillations
        def to_fit(Lk, A, X, phi):
            return osc_fn((10**Lk), k_pivot, A, X, phi)
        
        # A, X, phi
        ref_vals = [max(np.abs(new_yarr - 1)), X_guess, np.pi]
        lower_bounds = [0, 0.01*X_guess, 0]
        upper_bounds = [1, 100*X_guess, 2.*np.pi]
        
        pars2, _ = curve_fit(to_fit, np.log10(k_arr), new_yarr, ref_vals, bounds=(lower_bounds, upper_bounds))
        
        pars = np.concatenate([pars1, pars2])
        
        # rescaling dimensionful parameters
        pars[2] = (10**pars[2])/keq
        pars[3] = (10**pars[3])*tau_drag_twin
        
        return pars
    
    elif model == 'power_law_log_sup_dampsine':
        
        keq = self.k_eq()
        tau_drag_twin = self.tau_d_twin()
        delta = 1.-2.*self.r_all_twin()
        
        # fitting first to the spectrum without oscillations, in order to get a good guess
        def to_fit(Lk, D, n, ft, fa, b, g, d):
            return np.log10(pk_pl_log_sup(10**Lk, k_pivot, D, n, (10**ft)/keq, (10**fa)*tau_drag_twin, b, g, d))
        
        # D, n, tau, a, b, g, d
        ref_vals = [0.3, -2.3, 0, 0, 1, -1, delta]
        lower_bounds = [0, -5, -2, -2, 0, -10, 0]
        upper_bounds = [10, 5, 2, 2, 10, 0, 1]
        
        pars1, _ = curve_fit(to_fit, np.log10(k_arr), np.log10(dpk_arr), ref_vals, bounds=(lower_bounds, upper_bounds))
        
        # defining new y-array, only oscillations
        new_yarr = dpk_arr/(10**to_fit(np.log10(k_arr), *pars1))
        
        # interpolation of oscillations
        def yosc(k):
            temp_fn = interp1d(np.log10(k_arr), new_yarr - 1)    
            return temp_fn(np.log10(k))
        
        # estimate of frequency
        X_guess = 2.*np.pi/(2*np.mean(np.diff(zeros(yosc, k_arr))))
        
        # defining new function, with oscillations
        def to_fit(Lk, A, X, phi, fk, fs, p):            
            return damp_osc_fn((10**Lk), k_pivot, A, X, phi, (10**fk)/tau_drag_twin, (10**fs)/tau_drag_twin, p)
        
        # A, X, phi, km, sigma, p
        ref_vals = [max(np.abs(new_yarr - 1)), X_guess, np.pi, 0, 0, 2]
        lower_bounds = [0, 0.01*X_guess, 0, -2, -2, 0.5]
        upper_bounds = [1, 100.*X_guess, 2.*np.pi, 2, 2, 4]
        
        pars2, _ = curve_fit(to_fit, np.log10(k_arr), new_yarr, ref_vals, bounds=(lower_bounds, upper_bounds))
        
        pars = np.concatenate([pars1, pars2])
        
        # rescaling dimensionful parameters
        pars[2] = (10**pars[2])/keq
        pars[3] = (10**pars[3])*tau_drag_twin
        pars[10] = (10**pars[10])/tau_drag_twin
        pars[11] = (10**pars[11])/tau_drag_twin
        
        return pars
    
    else:
        raise ValueError("The only models allowed are 'power_law', 'power_law_log', 'power_law_log_sup', 'power_law_log_sup_sine', and 'power_law_log_sup_dampsine'.")



def Dn_fit(self, kbounds, kstar, zstar, model='power_law', Npts=201):
    
    k_lo = k_vel2comov(self, kbounds[0], zstar, units='1/Mpc') # TODO: check: conversion at zstar!
    k_hi = k_vel2comov(self, kbounds[1], zstar, units='1/Mpc') # TODO: check: conversion at zstar!
    k_pivot = k_vel2comov(self, kstar, zstar, units='1/Mpc')
    k_arr = np.logspace(np.log10(k_lo), np.log10(k_hi), Npts)
    
    fits = find_pk_fit(self, kbounds, kstar, zstar, model, Npts)
    
    if model == 'power_law':
        def dpk(k): return pk_pl(k, k_pivot, *fits)
    elif model == 'power_law_log':
        def dpk(k): return pk_pl_log(k, k_pivot, *fits)
    elif model == 'power_law_log_sup':
        def dpk(k): return pk_pl_log_sup(k, k_pivot, *fits)
    elif model == 'power_law_log_sup_sine':
        def dpk(k): return pk_pl_log_sup_osc(k, k_pivot, *fits)
    elif model == 'power_law_log_sup_dampsine':
        def dpk(k): return pk_pl_log_sup_damposc(k, k_pivot, *fits)
    
    n_arr = (k_arr*np.gradient(dpk(k_arr), k_arr)/dpk(k_arr) - 3)
    
    Dfit = dpk(k_pivot)
    nfit = float(interp1d(np.log10(k_arr), n_arr)(np.log10(k_pivot)))
    
    
    return (Dfit, nfit)