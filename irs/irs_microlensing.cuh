﻿#pragma once

#include "complex.cuh"
#include "star.cuh"


/********************************************************************************
lens equation

\param z -- image plane position
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star

\return w = (1-kappa_smooth)*z + shear*z_bar - theta^2 * sum( m_i / (z-z_i)_bar )
********************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, star<T>* stars, int nstars, T kappasmooth, T gamma, T theta)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*(1-kappa_smooth)*z+gamma*z_bar-starsum_bar*/
	return z * (1.0 - kappasmooth) + gamma * z.conj() - starsum.conj();
}

/************************************************************
complex point in the source plane converted to pixel position

\param p -- image plane position
\param hl -- half length of the source plane recieving region
\param npixels -- number of pixels in array

\return (p + (hl + hl*i)) * npixels / (2 * hl)
************************************************************/
template <typename T>
__device__ Complex<T> point_to_pixel(Complex<T> p, T hl, int npixels)
{
	T x1 = (p.re + hl) * npixels / (2.0f * hl);
	T y1 = (p.im + hl) * npixels / (2.0f * hl);
	return Complex<T>(x1, y1);
}

/**********************************************************************
shoot rays from image plane to source plane

\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappasmooth -- smooth matter convergence
\param gamma -- shear
\param theta -- size of the Einstein radius of a unit mass star
\param hlx -- half length of the image plane shooting region x size
\param hly -- half length of the image plane shooting region y size
\param raysep -- separation between central rays of shooting squares
\param hl -- half length of the source plane recieving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels in array
**********************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(star<T>* stars, int nstars, T kappasmooth, T gamma, T theta, T hlx, T hly, T raysep, T hl, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;
	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < 2.0f * hlx / raysep; i += x_stride)
	{
		for (int j = y_index; j < 2.0f * hly / raysep; j += y_stride)
		{
			/*x = image plane, y = source plane
			pay note in comments, as I may also use x and y to denote
			the usual cartesian coordinates when commenting on something*/
			Complex<T> x[4];
			Complex<T> y[4];

			/*location of central ray in image plane*/
			T cx = -hlx + raysep * (0.5f + i);
			T cy = -hly + raysep * (0.5f + j);

			/*shooting more rays in image plane at center +/- 1/3 * distance
			to next central ray in x and y direction*/
			T dx = raysep / 3.0f;

			x[0] = Complex<T>(cx + dx, cy + dx);
			x[1] = Complex<T>(cx - dx, cy + dx);
			x[2] = Complex<T>(cx - dx, cy - dx);
			x[3] = Complex<T>(cx + dx, cy - dx);

			/*map rays from image plane to source plane*/
			for (int k = 0; k < 4; k++)
			{
				y[k] = complex_image_to_source(x[k], stars, nstars, kappasmooth, gamma, theta);
			}

			/*repurpose cx and cy variables to now represent values in the source plane*/
			cx = (y[0].re + y[1].re + y[2].re + y[3].re) / 4.0f;
			cy = (y[0].im + y[1].im + y[2].im + y[3].im) / 4.0f;

			/*calculate Taylor coefficients of the time delay
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappasmooth and shear to
			be able to calculate down to the 4th derivatives of the time delay
			with our 4 rays shot*/
			T t11 = (1.0f - kappasmooth) + (y[0].re - y[1].re - y[2].re + y[3].re - y[0].im - y[1].im + y[2].im + y[3].im) / (8.0f * dx);
			T t12 = (y[0].re + y[1].re - y[2].re - y[3].re + y[0].im - y[1].im - y[2].im + y[3].im) / (8.0f * dx);

			T t111 = (y[3].im - y[2].im + y[1].im - y[0].im) / (4.0f * dx * dx);
			T t112 = (y[0].re - y[1].re + y[2].re - y[3].re) / (4.0f * dx * dx);

			T t1111 = 3.0f * (8.0f * dx * (1.0f - kappasmooth) - y[0].re + y[1].re + y[2].re - y[3].re - y[0].im - y[1].im + y[2].im + y[3].im) / (8.0f * dx * dx * dx);
			T t1112 = 3.0f * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8.0f * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9.0f;

			T ptx;
			T pty;
			T a11;
			T a12;
			T invmag;
			Complex<T> pt;
			int xpix;
			int ypix;
			for (int k = -13; k < 14; k++)
			{
				for (int l = -13; l < 14; l++)
				{
					ptx = cx + t11 * (dx * k) + t12 * (dx * l)
						+ 0.5f * t111 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) + t112 * (dx * k) * (dx * l)
						+ 1.0f / 6.0f * t1111 * ((dx * k) * (dx * k) * (dx * k) - 3.0f * (dx * k) * (dx * l) * (dx * l))
						+ 1.0f / 6.0f * t1112 * (3.0f * (dx * k) * (dx * k) * (dx * l) - (dx * l) * (dx * l) * (dx * l));

					pty = cy + t12 * (dx * k) + (2.0f * (1.0f - kappasmooth) - t11) * (dx * l)
						+ 0.5f * t112 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) - t111 * (dx * k) * (dx * l)
						+ 1.0f / 6.0f * t1112 * ((dx * k) * (dx * k) * (dx * k) - 3.0f * (dx * k) * (dx * l) * (dx * l))
						- 1.0f / 6.0f * t1111 * (3.0f * (dx * k) * (dx * k) * (dx * l) - (dx * l) * (dx * l) * (dx * l));

					pt = Complex<T>(ptx, pty);
					pt = point_to_pixel(pt, hl, npixels);
					xpix = static_cast<int>(pt.re);
					ypix = static_cast<int>(pt.im);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix = npixels - 1 - ypix;
					if (xpix < 0 || xpix > npixels - 1 || ypix < 0 || ypix > npixels - 1)
					{
						continue;
					}

					a11 = t11 + t111 * (dx * k) + t112 * (dx * l) + 0.5f * t1111 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) + t1112 * (dx * k) * (dx * l);
					a12 = t12 + t112 * (dx * k) - t111 * (dx * l) + 0.5f * t1112 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) - t1111 * (dx * k) * (dx * l);
					invmag = a11 * (2.0f * (1.0f - kappasmooth) - a11) - a12 * a12;

					if (invmag > 0)
					{
						atomicAdd(&(pixmin[ypix * npixels + xpix]), 1);
						atomicAdd(&(pixels[ypix * npixels + xpix]), 1);
					}
					else if (invmag < -0)
					{
						atomicAdd(&(pixsad[ypix * npixels + xpix]), 1);
						atomicAdd(&(pixels[ypix * npixels + xpix]), 1);
					}
					else
					{
						atomicAdd(&(pixmin[ypix * npixels + xpix]), 1);
						atomicAdd(&(pixsad[ypix * npixels + xpix]), 1);
						atomicAdd(&(pixels[ypix * npixels + xpix]), 2);
					}
				}
			}
		}
	}
}

