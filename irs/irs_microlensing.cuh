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
		return static_cast<T> (1);
	}
	else
	{
		return static_cast<T> (0);
	}
}

/*******************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param c -- corner of the rectangular region

\return 1 if z lies within the rectangle 
        defined by c, 0 if it is on the
		border or outside
*******************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> c)
{
	if (-c.re < z.re && z.re < c.re && -c.im < z.im && z.im < c.im)
	{
		return static_cast<T> (1);
	}
	else
	{
		return static_cast<T> (0);
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
\param c -- complex number denoting the corner of
            the rectangular field of point mass lenses

\return w = (1-kappa)*z + gamma*z_bar
            - theta^2 * sum(m_i/(z-z_i)_bar) - alpha_smooth
********************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> c)
{
	T PI = static_cast<T> (3.1415926535898);
	Complex<T> starsum;

	/*sum m_i/(z-z_i)*/
	for (int i = 0; i < nstars; i++)
	{
		starsum += stars[i].mass / (z - stars[i].position);
	}

	/*theta_e^2 * starsum*/
	starsum *= (theta * theta);

	Complex<T> c1 = c.conj() - z.conj();
	Complex<T> c2 = c - z.conj();
	Complex<T> c3 = -c - z.conj();
	Complex<T> c4 = -c.conj() - z.conj();

	Complex<T> alpha_smooth = Complex<T>(0, -kappastar / PI) * (-c1 * c1.log() + c2 * c2.log() + c3 * c3.log() - c4 * c4.log())
		- kappastar * 2 * (c.re + z.re) * boxcar(z, c)
		- kappastar * 4 * c.re * heaviside(c.im + z.im) * heaviside(c.im - z.im) * heaviside(z.re - c.re);

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

\return w = (1-(kappa-kappastar))*z + gamma*z_bar
            - theta^2 * sum(m_i/(z-z_i)_bar)
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

	/*(1-(kappa-kappastar))*z+gamma*z_bar-starsum_bar*/
	return (1 - (kappa - kappastar)) * z + gamma * z.conj() - starsum.conj();
}

/*************************************************************
complex point in the source plane converted to pixel position

\param w -- complex source plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels per side for the source
                  plane receiving region

\return (w + hly(1 + i)) * npixels / (2 * hly)
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
\param c -- complex number denoting the corner of the rectangular field
			of point mass lenses
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
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> c, T hlx1, T hlx2, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
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
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, nstars, kappastar, c);
			}

			/*calculate local Taylor coefficients of the potential
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappa and gamma to be able to
			calculate down to the 4th derivatives of the potential
			with our 4 rays shot*/

			T p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T p11 = (kappa - kappastar) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T p1111 = 3 * (8 * dx * ((kappa - kappastar) - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9;

			T ptx;
			T pty;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<T> pt;
			int xpix;
			int ypix;
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					T dx1 = dx * k;
					T dx2 = dx * l;

					ptx = dx1 - p1 - (p11 * dx1 + p12 * dx2)
						- (p111 * (dx1 * dx1 - dx2 * dx2) + 2 * p112 * dx1 * dx2) / 2
						- p1111 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- p1112 * (3 * dx1 * dx1 * dx2 - dx2 * dx2 * dx2) / 6;

					pty = dx2 - p2 - (p12 * dx1 + (2 * (kappa - kappastar) - p11) * dx2)
						- (p112 * (dx1 * dx1 - dx2 * dx2) - 2 * p111 * dx1 * dx2) / 2
						- p1112 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- p1111 * (-3 * dx1 * dx1 * dx2 + dx2 * dx2 * dx2) / 6;

					pt = Complex<T>(ptx, pty);
					pt = point_to_pixel(pt, hly, npixels);
					xpix = static_cast<int>(pt.re);
					ypix = static_cast<int>(pt.im);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix = npixels - 1 - ypix;
					if (xpix < 0 || xpix > npixels - 1 || ypix < 0 || ypix > npixels - 1)
					{
						continue;
					}

					invmag11 = 1 - p11 - (p111 * dx1 + p112 * dx2)
						- (p1111 * (dx1 * dx1- dx2 * dx2) + 2 * p1112 * dx1 * dx2) / 2;
					invmag12 = -p12 - (p112 * dx1 - p111 * dx2)
						- (p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - (kappa - kappastar)) - invmag11) - invmag12 * invmag12;

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

			T p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T p11 = (kappa - kappastar) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T p1111 = 3 * (8 * dx * ((kappa - kappastar) - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/*divide distance between rays again, by 9
			this gives us an increase in ray density of 27 (our
			initial division of 3, times this one of 9) per unit
			length, so 27^2 per unit area. these rays will use
			Taylor coefficients rather than being directly shot*/
			dx = dx / 9;

			T ptx;
			T pty;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<T> pt;
			int xpix;
			int ypix;
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					T dx1 = dx * k;
					T dx2 = dx * l;

					ptx = dx1 - p1 - (p11 * dx1 + p12 * dx2)
						- (p111 * (dx1 * dx1 - dx2 * dx2) + 2 * p112 * dx1 * dx2) / 2
						- p1111 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- p1112 * (3 * dx1 * dx1 * dx2 - dx2 * dx2 * dx2) / 6;

					pty = dx2 - p2 - (p12 * dx1 + (2 * (kappa - kappastar) - p11) * dx2)
						- (p112 * (dx1 * dx1 - dx2 * dx2) - 2 * p111 * dx1 * dx2) / 2
						- p1112 * (dx1 * dx1 * dx1 - 3 * dx1 * dx2 * dx2) / 6
						- p1111 * (-3 * dx1 * dx1 * dx2 + dx2 * dx2 * dx2) / 6;

					pt = Complex<T>(ptx, pty);
					pt = point_to_pixel(pt, hly, npixels);
					xpix = static_cast<int>(pt.re);
					ypix = static_cast<int>(pt.im);

					/*reverse y coordinate so array forms image in correct orientation*/
					ypix = npixels - 1 - ypix;
					if (xpix < 0 || xpix > npixels - 1 || ypix < 0 || ypix > npixels - 1)
					{
						continue;
					}

					invmag11 = 1 - p11 - (p111 * dx1 + p112 * dx2)
						- (p1111 * (dx1 * dx1 - dx2 * dx2) + 2 * p1112 * dx1 * dx2) / 2;
					invmag12 = -p12 - (p112 * dx1 - p111 * dx2)
						- (p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - (kappa - kappastar)) - invmag11) - invmag12 * invmag12;

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

