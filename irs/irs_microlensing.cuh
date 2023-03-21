#pragma once

#include "complex.cuh"
#include "star.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x > 0, 0 if x <= 0
******************************/
template <typename T>
__device__ T heaviside(T x)
{
	if (x > 0)
	{
		return static_cast<T>(1);
	}
	else
	{
		return static_cast<T>(0);
	}
}

/************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within the rectangle
		defined by corner, 0 if it is on the
		border or outside
************************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re < z.re && z.re < corner.re && -corner.im < z.im && z.im < corner.im)
	{
		return static_cast<T>(1);
	}
	else
	{
		return static_cast<T>(0);
	}
}

/********************************************************************
lens equation for a rectangular star field

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses

\return w = (1 - kappa) * z + gamma * z_bar
			- theta^2 * sum(m_i / (z - z_i)_bar) - alpha_smooth
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> c1 = corner - z.conj();
	Complex<T> c2 = corner.conj() - z.conj();
	Complex<T> c3 = -corner - z.conj();
	Complex<T> c4 = -corner.conj() - z.conj();

	Complex<T> alpha_smooth = Complex<T>(0, -kappastar / PI) * (c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log())
		- kappastar * 2 * (corner.re + z.re) * boxcar(z, corner)
		- kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);

	/*(1-kappa)*z+gamma*z_bar-starsum_bar-alpha_smooth*/
	return (1 - kappa) * z + gamma * z.conj() - starsum.conj() - alpha_smooth;
}

/********************************************************************
lens equation for a rectangular star field with approximations

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth

\return w = (1 - kappa) * z + gamma * z_bar
			- theta^2 * sum(m_i / (z - z_i)_bar) - alpha_smooth
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> s1;
	Complex<T> s2;
	Complex<T> s3;
	Complex<T> s4;

	for (int i = 1; i <= taylor; i++)
	{
		s1 += (z.conj() / corner).pow(i) / i;
		s2 += (z.conj() / corner.conj()).pow(i) / i;
		s3 += (z.conj() / -corner).pow(i) / i;
		s4 += (z.conj() / -corner.conj()).pow(i) / i;
	}

	Complex<T> alpha_smooth = ((corner - z.conj()) * (corner.log() - s1) - (corner.conj() - z.conj()) * (corner.conj().log() - s2)
		+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4));
	alpha_smooth *= Complex<T>(0, -kappastar / PI);
	alpha_smooth -= kappastar * 2 * (corner.re + z.re);

	/*(1-kappa)*z+gamma*z_bar-starsum_bar-alpha_smooth*/
	return (1 - kappa) * z + gamma * z.conj() - starsum.conj() - alpha_smooth;
}

/********************************************************************
lens equation for a circular star field

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses

\return w = (1 - kappa + kappastar) * z + gamma * z_bar
			- theta^2 * sum(m_i / (z - z_i)_bar)
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar)
{
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	/*(1-kappa+kappastar)*z+gamma*z_bar-starsum_bar*/
	return (1 - kappa + kappastar) * z + gamma * z.conj() - starsum.conj();
}

/*************************************************************
complex point in the source plane converted to pixel position

\param w -- complex source plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels per side for the source
                  plane receiving region

\return (w + hly * (1 + i)) * npixels / (2 * hly)
*************************************************************/
template <typename T>
__device__ Complex<T> point_to_pixel(Complex<T> w, T hly, int npixels)
{
	return (w + hly * Complex<T>(1, 1)) * npixels / (2 * hly);
}

/***********************************************************************
shoot rays from image plane to source plane for a rectangular star field

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
			     rectangular field of point mass lenses
\param hlx1 -- half length of the image plane shooting region x1 size
\param hlx2 -- half length of the image plane shooting region x2 size
\param raysep -- separation between central rays of shooting squares
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
***********************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, T hlx1, T hlx2, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < 2 * hlx1 / raysep; i += x_stride)
	{
		for (int j = y_index; j < 2 * hlx2 / raysep; j += y_stride)
		{
			/*x = image plane, y = source plane*/
			Complex<T> x[4];
			Complex<T> y[4];

			/*location of central ray in image plane*/
			T x1 = -hlx1 + raysep / 2 + raysep * i;
			T x2 = -hlx2 + raysep / 2 + raysep * j;

			Complex<T> z = Complex<T>(x1, x2);

			/*shooting more rays in image plane at center +/- 1/3 * distance
			to next central ray in x1 and x2 direction*/
			T dx = raysep / 3;

			x[0] = Complex<T>(x1 + dx, x2 + dx);
			x[1] = Complex<T>(x1 - dx, x2 + dx);
			x[2] = Complex<T>(x1 - dx, x2 - dx);
			x[3] = Complex<T>(x1 + dx, x2 - dx);

			/*map rays from image plane to source plane*/
#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, nstars, kappastar, corner);
			}

			/*calculate local Taylor coefficients of the potential
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappa and gamma to be able to
			calculate down to the 4th derivatives of the potential
			with our 4 rays shot*/

			T l_p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T l_p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T l_p11 = (kappa - kappastar * boxcar(z, corner)) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T l_p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T l_p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T l_p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T l_p1111 = 3 * (8 * dx * (kappa - kappastar * boxcar(z, corner) - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T l_p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9;

			T y1;
			T y2;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<T> ypos;
			Complex<int> ypix;
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					T dx1 = dx * k;
					T dx2 = dx * l;

					y1 = dx1 - l_p1 - (l_p11 * dx1 + l_p12 * dx2)
						- (l_p111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p112 * dx1 * dx2) / 2
						- l_p1111 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1112 * (3 * dx1 * dx1 * dx2 - dx2 * dx2 * dx2) / 6;

					y2 = dx2 - l_p2 - (l_p12 * dx1 + (2 * (kappa - kappastar * boxcar(z, corner)) - l_p11) * dx2)
						- (l_p112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p111 * dx1 * dx2) / 2
						- l_p1112 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1111 * (-3 * dx1 * dx1 * dx2 + dx2 * dx2 * dx2) / 6;

					if (y1 <= -hly || y1 >= hly || y2 <= -hly || y2 >= hly)
					{
						continue;
					}

					ypos = Complex<T>(y1, y2);
					ypix = point_to_pixel(ypos, hly, npixels);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix.im = npixels - 1 - ypix.im;

					invmag11 = 1 - l_p11 - (l_p111 * dx1 + l_p112 * dx2)
						- (l_p1111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p1112 * dx1 * dx2) / 2;
					invmag12 = -l_p12 - (l_p112 * dx1 - l_p111 * dx2)
						- (l_p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - kappa + kappastar * boxcar(z, corner)) - invmag11) - invmag12 * invmag12;

					if (invmag > 0)
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else if (invmag < -0)
					{
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 2);
					}
				}
			}
		}
	}
}

/***********************************************************************
shoot rays from image plane to source plane for a rectangular star field
with approximations

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param taylor -- degree of the taylor series for alpha_smooth
\param hlx1 -- half length of the image plane shooting region x1 size
\param hlx2 -- half length of the image plane shooting region x2 size
\param raysep -- separation between central rays of shooting squares
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
***********************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> corner, int taylor, T hlx1, T hlx2, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < 2 * hlx1 / raysep; i += x_stride)
	{
		for (int j = y_index; j < 2 * hlx2 / raysep; j += y_stride)
		{
			/*x = image plane, y = source plane*/
			Complex<T> x[4];
			Complex<T> y[4];

			/*location of central ray in image plane*/
			T x1 = -hlx1 + raysep / 2 + raysep * i;
			T x2 = -hlx2 + raysep / 2 + raysep * j;

			Complex<T> z = Complex<T>(x1, x2);

			/*shooting more rays in image plane at center +/- 1/3 * distance
			to next central ray in x1 and x2 direction*/
			T dx = raysep / 3;

			x[0] = Complex<T>(x1 + dx, x2 + dx);
			x[1] = Complex<T>(x1 - dx, x2 + dx);
			x[2] = Complex<T>(x1 - dx, x2 - dx);
			x[3] = Complex<T>(x1 + dx, x2 - dx);

			/*map rays from image plane to source plane*/
#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, nstars, kappastar, corner, taylor);
			}

			/*calculate local Taylor coefficients of the potential
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappa and gamma to be able to
			calculate down to the 4th derivatives of the potential
			with our 4 rays shot*/

			T l_p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T l_p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T l_p11 = (kappa - kappastar) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T l_p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T l_p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T l_p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T l_p1111 = 3 * (8 * dx * (kappa - kappastar - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T l_p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9;

			T y1;
			T y2;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<T> ypos;
			Complex<int> ypix;
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					T dx1 = dx * k;
					T dx2 = dx * l;

					y1 = dx1 - l_p1 - (l_p11 * dx1 + l_p12 * dx2)
						- (l_p111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p112 * dx1 * dx2) / 2
						- l_p1111 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1112 * (3 * dx1 * dx1 * dx2 - dx2 * dx2 * dx2) / 6;

					y2 = dx2 - l_p2 - (l_p12 * dx1 + (2 * (kappa - kappastar) - l_p11) * dx2)
						- (l_p112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p111 * dx1 * dx2) / 2
						- l_p1112 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1111 * (-3 * dx1 * dx1 * dx2 + dx2 * dx2 * dx2) / 6;

					if (y1 <= -hly || y1 >= hly || y2 <= -hly || y2 >= hly)
					{
						continue;
					}

					ypos = Complex<T>(y1, y2);
					ypix = point_to_pixel(ypos, hly, npixels);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix.im = npixels - 1 - ypix.im;

					invmag11 = 1 - l_p11 - (l_p111 * dx1 + l_p112 * dx2)
						- (l_p1111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p1112 * dx1 * dx2) / 2;
					invmag12 = -l_p12 - (l_p112 * dx1 - l_p111 * dx2)
						- (l_p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - kappa + kappastar) - invmag11) - invmag12 * invmag12;

					if (invmag > 0)
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else if (invmag < -0)
					{
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 2);
					}
				}
			}
		}
	}
}

/**********************************************************************
shoot rays from image plane to source plane for a circular star field

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param hlx1 -- half length of the image plane shooting region x1 size
\param hlx2 -- half length of the image plane shooting region x2 size
\param raysep -- separation between central rays of shooting squares
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
**********************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, T hlx1, T hlx2, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < 2 * hlx1 / raysep; i += x_stride)
	{
		for (int j = y_index; j < 2 * hlx2 / raysep; j += y_stride)
		{
			/*x = image plane, y = source plane*/
			Complex<T> x[4];
			Complex<T> y[4];

			/*location of central ray in image plane*/
			T x1 = -hlx1 + raysep / 2 + raysep * i;
			T x2 = -hlx2 + raysep / 2 + raysep * j;

			/*shooting more rays in image plane at center +/- 1/3 * distance
			to next central ray in x1 and x2 direction*/
			T dx = raysep / 3;

			x[0] = Complex<T>(x1 + dx, x2 + dx);
			x[1] = Complex<T>(x1 - dx, x2 + dx);
			x[2] = Complex<T>(x1 - dx, x2 - dx);
			x[3] = Complex<T>(x1 + dx, x2 - dx);

			/*map rays from image plane to source plane*/
#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, nstars, kappastar);
			}

			/*calculate Taylor coefficients of the potential
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappa and gamma to be able to
			calculate down to the 4th derivatives of the potential
			with our 4 rays shot*/

			T l_p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T l_p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T l_p11 = (kappa - kappastar) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T l_p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T l_p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T l_p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T l_p1111 = 3 * (8 * dx * (kappa - kappastar - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T l_p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9;

			T y1;
			T y2;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<T> ypos;
			Complex<int> ypix;
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					T dx1 = dx * k;
					T dx2 = dx * l;

					y1 = dx1 - l_p1 - (l_p11 * dx1 + l_p12 * dx2)
						- (l_p111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p112 * dx1 * dx2) / 2
						- l_p1111 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1112 * (3 * dx1 * dx1 * dx2 - dx2 * dx2 * dx2) / 6;

					y2 = dx2 - l_p2 - (l_p12 * dx1 + (2 * (kappa - kappastar) - l_p11) * dx2)
						- (l_p112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p111 * dx1 * dx2) / 2
						- l_p1112 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- l_p1111 * (-3 * dx1 * dx1 * dx2 + dx2 * dx2 * dx2) / 6;

					if (y1 <= -hly || y1 >= hly || y2 <= -hly || y2 >= hly)
					{
						continue;
					}

					ypos = Complex<T>(y1, y2);
					ypix = point_to_pixel(ypos, hly, npixels);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix.im = npixels - 1 - ypix.im;

					invmag11 = 1 - l_p11 - (l_p111 * dx1 + l_p112 * dx2)
						- (l_p1111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p1112 * dx1 * dx2) / 2;
					invmag12 = -l_p12 - (l_p112 * dx1 - l_p111 * dx2)
						- (l_p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - kappa + kappastar) - invmag11) - invmag12 * invmag12;

					if (invmag > 0)
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else if (invmag < -0)
					{
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 1);
					}
					else
					{
						if (pixmin) 
						{
							atomicAdd(&(pixmin[ypix.im * npixels + ypix.re]), 1);
						}
						if (pixsad) 
						{
							atomicAdd(&(pixsad[ypix.im * npixels + ypix.re]), 1);
						}
						atomicAdd(&(pixels[ypix.im * npixels + ypix.re]), 2);
					}
				}
			}
		}
	}
}

/**********************************************************************
initialize array of pixels to 0

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
**********************************************************************/
template <typename T>
__global__ void initialize_pixels_kernel(int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			pixels[j * npixels + i] = 0;
		}
	}
}

/**********************************************************************
calculate the minimum and maximum number of rays in the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minrays -- pointer to minimum number of rays
\param maxrays -- pointer to maximum number of rays
**********************************************************************/
template <typename T>
__global__ void histogram_min_max_kernel(int* pixels, int npixels, int* minrays, int* maxrays)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			atomicMin(minrays, pixels[j * npixels + i]);
			atomicMax(maxrays, pixels[j * npixels + i]);
		}
	}
}

/***************************************
initialize histogram values to 0

\param histogram -- pointer to histogram
\param n -- length of histogram
***************************************/
template <typename T>
__global__ void initialize_histogram_kernel(int* histogram, int n)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < n; i += x_stride)
	{
		histogram[i] = 0;
	}
}

/**********************************************************************
calculate the histogram of rays for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minrays -- minimum number of rays
\param histogram -- pointer to histogram
**********************************************************************/
template <typename T>
__global__ void histogram_kernel(int* pixels, int npixels, int minrays, int* histogram)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels; i += x_stride)
	{
		for (int j = y_index; j < npixels; j += y_stride)
		{
			atomicAdd(&(histogram[pixels[j * npixels + i] - minrays]), 1);
		}
	}
}

/**************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	std::ofstream outfile;

	if (fpath.extension() == ".txt")
	{
		outfile.precision(9);
		outfile.open(fname);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}
		for (int i = 0; i < nrows; i++)
		{
			for (int j = 0; j < ncols; j++)
			{
				outfile << vals[i * ncols + j] << " ";
			}
			outfile << "\n";
		}
		outfile.close();
	}
	else if (fpath.extension() == ".bin")
	{
		outfile.open(fname, std::ios_base::binary);

		if (!outfile.is_open())
		{
			std::cerr << "Error. Failed to open file " << fname << "\n";
			return false;
		}

		outfile.write((char*)(&nrows), sizeof(int));
		outfile.write((char*)(&ncols), sizeof(int));
		outfile.write((char*)vals, nrows * ncols * sizeof(T));
		outfile.close();
	}
	else
	{
		std::cerr << "Error. File " << fname << " is not a .bin or .txt file.\n";
		return false;
	}

	return true;
}

/**************************************************************
write histogram

\param histogram -- pointer to histogram
\param n -- length of histogram
\param minrays -- minimum number of rays
\param fname -- location of the file to write to

\return bool -- true if file successfully written, false if not
**************************************************************/
template <typename T>
bool write_histogram(int* histogram, int n, int minrays, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".txt")
	{
		std::cerr << "Error. File " << fname << " is not a .txt file.\n";
		return false;
	}

	std::ofstream outfile;

	outfile.precision(9);
	outfile.open(fname);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return false;
	}
	for (int i = 0; i < n; i++)
	{
		if (histogram[i] != 0)
		{
			outfile << i + minrays << " " << histogram[i] << "\n";
		}
	}
	outfile.close();

	return true;
}

