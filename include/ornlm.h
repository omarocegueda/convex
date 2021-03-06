/*! \file ornlm.cpp
	\author Omar Ocegueda
	\brief Optimized (Blockwise) Rician-corrected Nonlocal Means filter. Addapted from Pierrick's code with some low level optimizations
*/
#ifndef ORNLM_H
#define ORNLM_H

/*! \brief Optimized (Blockwise) Rician-corrected Nonlocal Means filter
	\param v specifies the local volume to look for similar patches: (2*v+1)^3. A plausible value is v=3.
	\param patchLen specifies the size of the patches: (2*patchLen+1)^3. A plausible value is patchLen=1.
	\param h specifies the amount of smoothing. Plausible values are between 3% and 20% of the maximum value in the image
*/
void ornlm(double *image, int nslices, int nrows, int ncols, int v, int patchLen, double h, double *filtered);
#endif
