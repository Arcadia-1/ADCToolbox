import numpy as np
from scipy.linalg import lstsq
import warnings

# Verified

def calibrate_weight_sine(bits, freq=0, rate=0.5, reltol=1e-12, niter=100, order=1, fsearch=0, verbose=0, nom_weight=None):
    """
    FGCalSine — Foreground calibration using a sinewave input

    This function estimates per-bit weights and a DC offset for an ADC by
    fitting the weighted sum of raw bit columns to a sine series at a given
    (or estimated) normalized frequency Fin/Fs. It optionally performs a
    coarse and fine frequency search to refine the input tone frequency.

    Parameters
    ----------
    bits : ndarray or list of ndarrays
        Binary data as matrix (N rows by M cols, N is data points, M is bitwidth).
        Each row is one sample; each column is a bit/segment.
        Can also be a list of arrays for multi-dataset calibration.
    freq : float or array-like, optional
        Normalized frequency Fin/Fs (0 triggers frequency search), default is 0.
        For multi-dataset: can be array of frequencies per dataset.
    rate : float, optional
        Adaptive rate for frequency updates (0..1), default is 0.5.
    reltol : float, optional
        Relative error tolerance, default is 1e-12.
    niter : int, optional
        Max fine-search iterations, default is 100.
    order : int, optional
        Harmonics exclusion order (1 for no exclusion), default is 1.
    fsearch : int, optional
        Force fine search (1) or not (0), default is 0.
    verbose : int, optional
        Print frequency search progress (1) or not (0), default is 0.
    nom_weight : array-like, optional
        Nominal bit weights (only effective when rank is deficient).
        Default is 2^(M-1) down to 2^0.

    Returns
    -------
    weight : ndarray
        The calibrated weights, normalized by the magnitude of sinewave.
    offset : float
        The calibrated DC offset, normalized by the magnitude of sinewave.
    postCal : ndarray or list of ndarrays
        The signal after calibration.
    ideal : ndarray or list of ndarrays
        The best fitted sinewave.
    err : ndarray or list of ndarrays
        The residue errors after calibration (excluding distortion).
    freqCal : float or ndarray
        The fine tuned frequency from calibration.
    """
    
    # ==========================
    # Multi-dataset (list) path
    # ==========================
    if isinstance(bits, list):
        # Validate shapes and collect per-dataset sizes
        ND = len(bits)  # Number of datasets
        if ND == 0:
            raise ValueError('FGCalSine:EmptyInput - Empty list for bits.')
        
        bits_cell = []
        Nk = np.zeros(ND, dtype=int)
        Mk = np.zeros(ND, dtype=int)
        
        # Process each dataset to ensure correct orientation (samples as rows)
        for k in range(ND):
            Bk = bits[k]
            if Bk.size == 0:
                raise ValueError(f'FGCalSine:EmptyDataset - Dataset {k} is empty.')
            
            nTmp, mTmp = Bk.shape
            # Transpose if rows < cols (assume samples should be rows)
            if nTmp < mTmp:
                Bk = Bk.T
                nTmp, mTmp = Bk.shape
            
            bits_cell.append(Bk)
            Nk[k] = nTmp
            Mk[k] = mTmp
        
        # Ensure all datasets have the same bitwidth
        if not np.all(Mk == Mk[0]):
            raise ValueError('FGCalSine:InconsistentWidth - All datasets must have the same number of columns (bits).')
        
        M_orig = Mk[0]
        
        # Process frequency parameter (scalar or vector)
        if np.isscalar(freq):
            freq = np.ones(ND) * freq
        else:
            freq = np.asarray(freq)
        
        if len(freq) != ND:
            raise ValueError('FGCalSine:FreqLength - Length of freq vector must match number of datasets.')
        
        # Set default nominal weights if not provided
        if nom_weight is None:
            nom_weight = 2.0 ** np.arange(M_orig - 1, -1, -1)
        
        # Per-dataset frequency search (only for unknown entries or if forced)
        for k in range(ND):
            if freq[k] == 0 or fsearch == 1:
                _, _, _, _, _, fk = fg_cal_sine(
                    bits_cell[k], freq=freq[k], fsearch=1, order=order,
                    rate=rate, reltol=reltol, niter=niter, verbose=verbose, nom_weight=nom_weight
                )
                freq[k] = fk
        
        # Build unified dataset by concatenating all datasets vertically
        bits_all = np.vstack(bits_cell)
        Ntot = bits_all.shape[0]
        
        # Initialize link and scaling maps for rank patching
        Lmap = np.arange(M_orig)  # Link from original column to patched column
        Kmap = np.ones(M_orig)    # Scale factor for each column
        
        # Rank deficiency patching routine on concatenated data
        if np.linalg.matrix_rank(np.column_stack([bits_all, np.ones((Ntot, 1))])) < M_orig + 1:
            warnings.warn('Rank deficiency detected across datasets. Try patching...')
            bits_patch_all = np.empty((Ntot, 0))
            LR = []  # Reverse link from patched columns to original indices
            M2 = 0
            
            for i1 in range(M_orig):
                # Check if column is constant (no information)
                if np.max(bits_all[:, i1]) == np.min(bits_all[:, i1]):
                    Lmap[i1] = -1  # Mark as invalid (will use 0 weight)
                    Kmap[i1] = 0
                # Check if column adds new rank
                elif np.linalg.matrix_rank(np.column_stack([np.ones((Ntot, 1)), bits_patch_all, bits_all[:, i1]])) > \
                     np.linalg.matrix_rank(np.column_stack([np.ones((Ntot, 1)), bits_patch_all])):
                    bits_patch_all = np.column_stack([bits_patch_all, bits_all[:, i1]])
                    LR.append(i1)
                    M2 = bits_patch_all.shape[1]
                    Lmap[i1] = M2 - 1
                else:
                    # Try to merge with existing column based on correlation
                    flag = 0
                    for i2 in range(M2):
                        r1 = bits_all[:, i1] - np.mean(bits_all[:, i1])
                        r2 = bits_patch_all[:, i2] - np.mean(bits_patch_all[:, i2])
                        cor = np.mean(r1 * r2) / (np.sqrt(np.mean(r1**2)) * np.sqrt(np.mean(r2**2)))
                        
                        # Perfectly correlated (|correlation| ≈ 1)
                        if abs(abs(cor) - 1) < 1e-3:
                            Lmap[i1] = i2
                            Kmap[i1] = nom_weight[i1] / nom_weight[LR[i2]]
                            # Merge column i1 into column i2
                            bits_patch_all[:, i2] = bits_patch_all[:, i2] + bits_all[:, i1] * Kmap[i1]
                            flag = 1
                            break
                    
                    if flag == 0:
                        Lmap[i1] = -1
                        Kmap[i1] = 0
                        warnings.warn(f'Patch warning: cannot find correlated column for column {i1}. Resulting weight will be zero')
            
            M_patch = bits_patch_all.shape[1]
            
            # Check if patching succeeded
            if np.linalg.matrix_rank(np.column_stack([np.ones((Ntot, 1)), bits_patch_all])) < M_patch + 1:
                raise ValueError('Patch failed: rank still deficient after patching across datasets. Try adjusting nomWeight.')
        else:
            bits_patch_all = bits_all
            M_patch = M_orig
        
        # Column magnitude scaling for numerical conditioning
        max_vals = np.max(np.abs(np.vstack([np.max(bits_patch_all, axis=0), np.min(bits_patch_all, axis=0)])), axis=0)
        MAG = np.floor(np.log10(max_vals))
        MAG[np.isinf(MAG)] = 0
        bits_patch_all = bits_patch_all * (10.0 ** (-MAG))
        
        # Build harmonic basis matrices for all datasets
        numHCols = ND * order  # Total number of harmonic columns
        xc = np.zeros((Ntot, numHCols))  # Cosine basis
        xs = np.zeros((Ntot, numHCols))  # Sine basis
        
        rowStart = 0
        for k in range(ND):
            Nk_k = Nk[k]
            rowEnd = rowStart + Nk_k
            # Phase matrix for harmonics 1 to order
            theta_mat = np.outer(np.arange(Nk_k), freq[k] * np.arange(1, order + 1))
            xc[rowStart:rowEnd, k*order:(k+1)*order] = np.cos(theta_mat * 2 * np.pi)
            xs[rowStart:rowEnd, k*order:(k+1)*order] = np.sin(theta_mat * 2 * np.pi)
            rowStart = rowEnd
        
        # Assumption 1: cosine fundamental is unity basis
        A = np.column_stack([bits_patch_all, np.ones((Ntot, 1)), xc[:, order:], xs])
        b = -xc[:, 0]
        x1, _, _, _ = lstsq(A, b)
        
        # Assumption 2: sine fundamental is unity basis
        A = np.column_stack([bits_patch_all, np.ones((Ntot, 1)), xs[:, order:], xc])
        b = -xs[:, 0]
        x2, _, _, _ = lstsq(A, b)
        
        # Choose solution with smaller residual
        if np.sqrt(np.mean((np.dot(np.column_stack([bits_patch_all, np.ones((Ntot, 1)), xc[:, order:], xs]), x1) + xc[:, 0])**2)) < \
           np.sqrt(np.mean((np.dot(np.column_stack([bits_patch_all, np.ones((Ntot, 1)), xs[:, order:], xc]), x2) + xs[:, 0])**2)):
            x = x1
            sel = 0  # Cosine-based solution
        else:
            x = x2
            sel = 1  # Sine-based solution
        
        # Normalization using dataset-1 quadrature fundamental
        w0 = np.sqrt(1 + x[M_patch + numHCols]**2)
        
        # Map weights back to original columns
        wpatch = (x[:M_patch] / w0) * (10.0 ** (-MAG))
        weight = wpatch[np.maximum(Lmap, 0)] * Kmap  # Use max to handle -1 indices
        weight[Lmap < 0] = 0  # Zero out invalid columns
        
        offset = -x[M_patch] / w0
        
        # Reconstruct per-dataset signals
        postCal = []
        ideal = []
        err = []
        
        # First dataset
        if sel:
            ideal_0 = -(xs[:Nk[0], :order] @ x[M_patch + 1:M_patch + 1 + order] + 
                       xc[:Nk[0], :order] @ x[M_patch + 1 + numHCols:M_patch + 1 + numHCols + order]) / w0
        else:
            ideal_0 = -(xc[:Nk[0], :order] @ x[M_patch + 1:M_patch + 1 + order] + 
                       xs[:Nk[0], :order] @ x[M_patch + 1 + numHCols:M_patch + 1 + numHCols + order]) / w0
        ideal.append(ideal_0)
        
        # Remaining datasets
        rowStart = Nk[0]
        for k in range(1, ND):
            rowEnd = rowStart + Nk[k]
            if sel:
                ideal_k = -(xs[rowStart:rowEnd, k*order:(k+1)*order] @ x[M_patch + k*order:M_patch + (k+1)*order] +
                           xc[rowStart:rowEnd, k*order:(k+1)*order] @ x[M_patch + numHCols + k*order:M_patch + numHCols + (k+1)*order]) / w0
            else:
                ideal_k = -(xc[rowStart:rowEnd, k*order:(k+1)*order] @ x[M_patch + k*order:M_patch + (k+1)*order] +
                           xs[rowStart:rowEnd, k*order:(k+1)*order] @ x[M_patch + numHCols + k*order:M_patch + numHCols + (k+1)*order]) / w0
            ideal.append(ideal_k)
            rowStart = rowEnd
        
        # Compute calibrated output and errors for all datasets
        for k in range(ND):
            postCal.append(weight @ bits_cell[k].T)
            err.append(postCal[k] - offset - ideal[k])
        
        # Enforce positive polarity
        if np.sum(weight) < 0:
            weight = -weight
            offset = -offset
            for k in range(ND):
                postCal[k] = -postCal[k]
                ideal[k] = -ideal[k]
                err[k] = -err[k]
        
        freqCal = freq
        return weight, offset, postCal, ideal, err, freqCal
    
    # ==================
    # Single-dataset path
    # ==================
    N, M = bits.shape  # N: number of samples (rows), M: number of bit columns
    
    # Ensure rows are samples, columns are bits
    if N < M:
        bits = bits.T
        N, M = bits.shape
    
    # Set default nominal weights if not provided
    if nom_weight is None:
        nom_weight = 2.0 ** np.arange(M - 1, -1, -1)
    
    # Ensure order is at least 1
    order = max(int(round(order)), 1)
    
    # Initialize link and scale tables for rank patching
    L = np.arange(M)      # Link from column to its correlated column
    K = np.ones(M)        # Weight ratio to its correlated column
    
    # Rank deficiency patching: merge correlated columns, discard constants
    if np.linalg.matrix_rank(np.column_stack([bits, np.ones((N, 1))])) < M + 1:
        warnings.warn('Rank deficiency detected. Try patching...')
        bits_patch = np.empty((N, 0))
        LR = []  # Reverse link from bits_patch to original column indices
        M2 = 0   # Number of columns in bits_patch
        
        for i1 in range(M):
            # Constant column -> discard
            if np.max(bits[:, i1]) == np.min(bits[:, i1]):
                L[i1] = -1
                K[i1] = 0
            # Column adds rank -> keep it
            elif np.linalg.matrix_rank(np.column_stack([np.ones((N, 1)), bits_patch, bits[:, i1]])) > \
                 np.linalg.matrix_rank(np.column_stack([np.ones((N, 1)), bits_patch])):
                bits_patch = np.column_stack([bits_patch, bits[:, i1]])
                LR.append(i1)
                M2 = bits_patch.shape[1]
                L[i1] = M2 - 1
            else:
                # Try to merge with existing column based on correlation
                flag = 0
                for i2 in range(M2):
                    r1 = bits[:, i1] - np.mean(bits[:, i1])
                    r2 = bits_patch[:, i2] - np.mean(bits_patch[:, i2])
                    cor = np.mean(r1 * r2) / (np.sqrt(np.mean(r1**2)) * np.sqrt(np.mean(r2**2)))
                    
                    # Perfectly correlated (|correlation| ≈ 1)
                    if abs(abs(cor) - 1) < 1e-3:
                        L[i1] = i2
                        K[i1] = nom_weight[i1] / nom_weight[LR[i2]]
                        # Merge i1 into i2
                        bits_patch[:, i2] = bits_patch[:, i2] + bits[:, i1] * nom_weight[i1] / nom_weight[LR[i2]]
                        flag = 1
                        break
                
                if flag == 0:
                    L[i1] = -1
                    K[i1] = 0
                    warnings.warn(f'Patch warning: cannot find the correlated column for column {i1}. The resulting weight will be zero')
        
        M = bits_patch.shape[1]
        
        # Check if patching succeeded
        if np.linalg.matrix_rank(np.column_stack([np.ones((N, 1)), bits_patch])) < M + 1:
            raise ValueError('Patch failed: rank still deficient after patching. This may be fixed by changing nom_weight.')
    else:
        bits_patch = bits
    
    # Pre-scaling columns for numerical conditioning
    max_vals = np.max(np.abs(np.vstack([np.max(bits_patch, axis=0), np.min(bits_patch, axis=0)])), axis=0)
    MAG = np.floor(np.log10(max_vals))
    MAG[np.isinf(MAG)] = 0
    bits_patch = bits_patch * (10.0 ** (-MAG))
    
    # Coarse frequency search if freq not provided (freq==0)
    if freq == 0:
        fsearch = 1
        freq_estimates = []

        # Use first min(M, 5) columns to estimate frequency
        for i1 in range(min(M, 5)):
            if verbose:
                print(f'Freq coarse searching ({i1+1}/5):', end='')
            # Weighted sum of first i1+1 columns
            weighted_sum = bits_patch[:, :i1+1] @ nom_weight[:i1+1]
            freq_est = find_fin(weighted_sum)
            freq_estimates.append(freq_est)
            if verbose:
                print(f' freq = {freq_est}')

        freq = np.median(freq_estimates)
    
    # Build harmonic basis matrices
    theta_mat = np.outer(np.arange(N), freq * np.arange(1, order + 1))
    xc = np.cos(theta_mat * 2 * np.pi)  # Cosine basis
    xs = np.sin(theta_mat * 2 * np.pi)  # Sine basis
    
    # Assumption 1: cosine is the unity basis
    A = np.column_stack([bits_patch, np.ones(N), xc[:, 1:], xs])
    b = -xc[:, 0]
    x1, _, _, _ = lstsq(A, b)
    
    # Assumption 2: sine is the unity basis
    A = np.column_stack([bits_patch, np.ones(N), xs[:, 1:], xc])
    b = -xs[:, 0]
    x2, _, _, _ = lstsq(A, b)
    
    # Choose solution with smaller residual
    if np.sqrt(np.mean((np.column_stack([bits_patch, np.ones(N), xc[:, 1:], xs]) @ x1 + xc[:, 0])**2)) < \
       np.sqrt(np.mean((np.column_stack([bits_patch, np.ones(N), xs[:, 1:], xc]) @ x2 + xs[:, 0])**2)):
        x = x1
        sel = 0  # Cosine-based solution
    else:
        x = x2
        sel = 1  # Sine-based solution
    
    # Optional fine frequency search: iteratively refine frequency
    if fsearch:
        warnings.filterwarnings('ignore')
        delta_f = 0
        time_mat = np.outer(np.arange(N), np.ones(order))  # Time index matrix for derivatives
        
        for ii in range(niter):
            freq = freq + delta_f  # Update frequency
            theta_mat = np.outer(np.arange(N), freq * np.arange(1, order + 1))
            
            xc = np.cos(theta_mat * 2 * np.pi)
            xs = np.sin(theta_mat * 2 * np.pi)
            
            order_mat = np.outer(np.ones(N), np.arange(1, order + 1))  # Scale derivatives by harmonic index
            
            # Build coefficient matrices based on current solution
            if sel:
                KS = np.outer(np.ones(N), np.concatenate([[1], x[M+1:M+order]])) * order_mat
                KC = np.outer(np.ones(N), x[M+order:M+2*order]) * order_mat
            else:
                KC = np.outer(np.ones(N), np.concatenate([[1], x[M+1:M+order]])) * order_mat
                KS = np.outer(np.ones(N), x[M+order:M+2*order]) * order_mat
            
            # Partial derivatives w.r.t. frequency
            xcd = -2 * np.pi * KC * time_mat * np.sin(theta_mat * 2 * np.pi) / N
            xsd = 2 * np.pi * KS * time_mat * np.cos(theta_mat * 2 * np.pi) / N
            
            # Re-solve augmented systems with derivative column
            A = np.column_stack([bits_patch, np.ones(N), xc[:, 1:], xs, np.sum(xcd + xsd, axis=1)])
            b = -xc[:, 0]
            x1, _, _, _ = lstsq(A, b)
            e1 = A @ x1 - b
            
            A = np.column_stack([bits_patch, np.ones(N), xs[:, 1:], xc, np.sum(xcd + xsd, axis=1)])
            b = -xs[:, 0]
            x2, _, _, _ = lstsq(A, b)
            e2 = A @ x2 - b
            
            # Choose better solution
            if np.sqrt(np.mean(e1**2)) < np.sqrt(np.mean(e2**2)):
                x = x1
                sel = 0
            else:
                x = x2
                sel = 1
            
            # Compute frequency correction
            delta_f = x[-1] * rate / N
            relerr = np.sqrt(np.mean((x[-1] / N * A[:, -1])**2)) / np.sqrt(1 + x[M + order]**2)

            if verbose:
                print(f'Freq fine iterating ({ii+1}): freq = {freq}, delta_f = {delta_f}, rel_err = {relerr}')

            # Stop if relative error is below tolerance
            if relerr < reltol:
                break
        
        warnings.filterwarnings('default')
    
    # Compute normalization factor (fundamental magnitude)
    w0 = np.sqrt(1 + x[M + order]**2)
    
    # Compute final weights (undo scaling and normalize)
    weight = (x[:M] / w0) * (10.0 ** (-MAG))
    weight = weight[np.maximum(L, 0)] * K  # Map back to original columns
    weight[L < 0] = 0  # Zero out invalid columns
    
    offset = -x[M] / w0  # Normalized DC offset
    
    postCal = weight @ bits.T  # Calibrated signal
    
    # Reconstruct ideal sinewave
    if sel:
        ideal = -(xs[:, 0] + xs[:, 1:] @ x[M+1:M+order] + xc @ x[M+order:M+2*order]) / w0
    else:
        ideal = -(xc[:, 0] + xc[:, 1:] @ x[M+1:M+order] + xs @ x[M+order:M+2*order]) / w0
    
    err = postCal - offset - ideal  # Residual error
    
    # Enforce positive polarity
    if np.sum(weight) < 0:
        weight = -weight
        offset = -offset
        postCal = -postCal
        ideal = -ideal
        err = -err
    
    freqCal = freq
    
    return weight, offset, postCal, ideal, err, freqCal


def find_fin(signal):
    """
    Helper function to estimate normalized frequency Fin/Fs from a signal.
    Uses FFT to find the dominant frequency component.
    
    Parameters
    ----------
    signal : ndarray
        Input signal (1D array)
    
    Returns
    -------
    float
        Estimated normalized frequency (Fin/Fs)
    """
    # Remove DC component
    signal = signal - np.mean(signal)
    
    # Compute FFT
    N = len(signal)
    fft_result = np.fft.fft(signal)
    fft_mag = np.abs(fft_result[:N//2])
    
    # Find peak frequency (excluding DC at index 0)
    peak_idx = np.argmax(fft_mag[1:]) + 1
    
    # Normalized frequency
    fin_fs = peak_idx / N
    
    return fin_fs