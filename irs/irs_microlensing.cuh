#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "complex.cuh"
#include "star.cuh"


/**********************************************************************************************
lens equation

\param z -- image plane position
\param kappa -- macro convergence
\param gamma -- macro shear
\param theta -- size of the Einstein radius of a unit mass
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappastar -- convergence in compact objects
\param shearsmooth -- shear due to smooth mass sheet

\return w = (1-kappa+kappastar)*z + (gamma-shearsmooth)*z_bar - theta^2 * sum(m_i/(z-z_i)_bar )
**********************************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> c)
{
	T PI = 3.1415926535898;
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

	Complex<T> alpha_smooth = Complex<T>(0, -kappastar / PI) * (-c1 * c1.log() + c2 * c2.log() + c3 * c3.log() - c4 * c4.log() + Complex<T>(0, 2.0f * PI * (-c.re - z.re)));


	/*(1-kappa)*z+gamma*z_bar-starsum_bar*/
	return z * (1.0 - kappa) + gamma * z.conj() - starsum.conj() - alpha_smooth;
}

/************************************************************
complex point in the source plane converted to pixel position

\param p -- image plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels in array

\return (p + (hly + hly*i)) * npixels / (2 * hly)
************************************************************/
template <typename T>
__device__ Complex<T> point_to_pixel(Complex<T> p, T hly, int npixels)
{
	T x1 = (p.re + hly) * npixels / (2.0 * hly);
	T y1 = (p.im + hly) * npixels / (2.0 * hly);
	return Complex<T>(x1, y1);
}

/**********************************************************************
shoot rays from image plane to source plane

\param kappa -- macro convergence
\param gamma -- macro shear
\param theta -- size of the Einstein radius of a unit mass
\param stars -- pointer to array of stars
\param nstars -- number of stars in array
\param kappastar -- convergence in compact objects
\param shearsmooth -- shear due to smooth mass sheet
\param hlx1 -- half length of the image plane shooting region x size
\param hlx2 -- half length of the image plane shooting region y size
\param raysep -- separation between central rays of shooting squares
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
**********************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, int nstars, T kappastar, Complex<T> c, T hlx1, T hlx2, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < 2.0f * hlx1 / raysep; i += x_stride)
	{
		for (int j = y_index; j < 2.0f * hlx2 / raysep; j += y_stride)
		{
			/*x = image plane, y = source plane
			pay note in comments, as I may also use x and y to denote
			the usual cartesian coordinates when commenting on something*/
			Complex<T> x[4];
			Complex<T> y[4];

			/*location of central ray in image plane*/
			T cx = -hlx1 + raysep * (0.5f + i);
			T cy = -hlx2 + raysep * (0.5f + j);

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
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, nstars, kappastar, c);
			}

			/*repurpose cx and cy variables to now represent values in the source plane*/
			cx = (y[0].re + y[1].re + y[2].re + y[3].re) / 4.0f;
			cy = (y[0].im + y[1].im + y[2].im + y[3].im) / 4.0f;

			/*calculate Taylor coefficients of the time delay
			relies on symmetries and the fact that there are no higher
			order macro-derivatives than kappasmooth and shear to
			be able to calculate down to the 4th derivatives of the time delay
			with our 4 rays shot*/
			T t11 = (1.0f - kappa + kappastar) + (y[0].re - y[1].re - y[2].re + y[3].re - y[0].im - y[1].im + y[2].im + y[3].im) / (8.0f * dx);
			T t12 = (y[0].re + y[1].re - y[2].re - y[3].re + y[0].im - y[1].im - y[2].im + y[3].im) / (8.0f * dx);

			T t111 = (y[3].im - y[2].im + y[1].im - y[0].im) / (4.0f * dx * dx);
			T t112 = (y[0].re - y[1].re + y[2].re - y[3].re) / (4.0f * dx * dx);

			T t1111 = 3.0f * (8.0f * dx * (1.0f - kappa + kappastar) - y[0].re + y[1].re + y[2].re - y[3].re - y[0].im - y[1].im + y[2].im + y[3].im) / (8.0f * dx * dx * dx);
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
			for (int k = -13; k <= 13; k++)
			{
				for (int l = -13; l <= 13; l++)
				{
					ptx = cx + t11 * (dx * k) + t12 * (dx * l)
						+ 0.5f * t111 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) + t112 * (dx * k) * (dx * l)
						+ 1.0f / 6.0f * t1111 * ((dx * k) * (dx * k) * (dx * k) - 3.0f * (dx * k) * (dx * l) * (dx * l))
						+ 1.0f / 6.0f * t1112 * (3.0f * (dx * k) * (dx * k) * (dx * l) - (dx * l) * (dx * l) * (dx * l));

					pty = cy + t12 * (dx * k) + (2.0f * (1.0f - kappa + kappastar) - t11) * (dx * l)
						+ 0.5f * t112 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) - t111 * (dx * k) * (dx * l)
						+ 1.0f / 6.0f * t1112 * ((dx * k) * (dx * k) * (dx * k) - 3.0f * (dx * k) * (dx * l) * (dx * l))
						- 1.0f / 6.0f * t1111 * (3.0f * (dx * k) * (dx * k) * (dx * l) - (dx * l) * (dx * l) * (dx * l));

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

					a11 = t11 + t111 * (dx * k) + t112 * (dx * l) + 0.5f * t1111 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) + t1112 * (dx * k) * (dx * l);
					a12 = t12 + t112 * (dx * k) - t111 * (dx * l) + 0.5f * t1112 * ((dx * k) * (dx * k) - (dx * l) * (dx * l)) - t1111 * (dx * k) * (dx * l);
					invmag = a11 * (2.0f * (1.0f - kappa + kappastar) - a11) - a12 * a12;

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

