#pragma once

#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************************************************************
number of rays that will be shot in each x1 and x2 direction using taylor
coefficients is equal to 2 * HALF_NUM_RESAMPLED_RAYS + 1
******************************************************************************/
const int HALF_NUM_RESAMPLED_RAYS = 30;
const int NUM_RESAMPLED_RAYS = 2 * HALF_NUM_RESAMPLED_RAYS + 1;

/******************************************************************************
calculate the deflection angle due to nearby stars for a node

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- node within which to calculate the deflection angle

\return alpha_star = theta^2 * sum(m_i / (z - z_i)_bar)
******************************************************************************/
template <typename T>
__device__ Complex<T> star_deflection(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> alpha_star_bar;

	/******************************************************************************
	theta^2 * sum(m_i / (z - z_i))
	******************************************************************************/
	for (int i = 0; i < node->numstars; i++)
	{
		alpha_star_bar += stars[node->stars + i].mass / (z - stars[node->stars + i].position);
	}
	for (int j = 0; j < node->numneighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			alpha_star_bar += stars[neighbor->stars + i].mass / (z - stars[neighbor->stars + i].position);
		}
	}
	alpha_star_bar *= (theta * theta);

	return alpha_star_bar.conj();
}

/******************************************************************************
calculate the deflection angle due to far away stars for a node

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param node -- node within which to calculate the deflection angle

\return alpha_local = theta^2 * sum(i * a_i * (z - z_0) ^ (i - 1))
           where a_i are coefficients of the lensing potential
******************************************************************************/
template <typename T>
__device__ Complex<T> local_deflection(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> alpha_local_bar;
	Complex<T> dz = (z - node->center);

	for (int i = node->expansion_order - 1; i >= 0; i--)
	{
		alpha_local_bar *= dz;
		alpha_local_bar += node->local_coeffs[i + 1] * (i + 1);
	}
	alpha_local_bar *= (theta * theta);

	return alpha_local_bar.conj();
}

/******************************************************************************
calculate the deflection angle due to smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> smooth_deflection(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = 3.1415926535898;
	Complex<T> alpha_smooth;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> s1;
			Complex<T> s2;
			Complex<T> s3;
			Complex<T> s4;

			for (int i = 1; i <= taylor_smooth; i++)
			{
				s1 += (z.conj() / corner).pow(i) / i;
				s2 += (z.conj() / corner.conj()).pow(i) / i;
				s3 += (z.conj() / -corner).pow(i) / i;
				s4 += (z.conj() / -corner.conj()).pow(i) / i;
			}

			alpha_smooth = ((corner - z.conj()) * (corner.log() - s1) - (corner.conj() - z.conj()) * (corner.conj().log() - s2)
				+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4));
			alpha_smooth *= Complex<T>(0, -kappastar / PI);
			alpha_smooth -= kappastar * 2 * (corner.re + z.re);
		}
		else
		{
			Complex<T> c1 = corner - z.conj();
			Complex<T> c2 = corner.conj() - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			/******************************************************************************
			assumes rays shot lie within the rectangle of stars, thus removing any boxcar
			and heaviside functions
			******************************************************************************/
			alpha_smooth = (c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log());
			alpha_smooth *= Complex<T>(0, -kappastar / PI);
			alpha_smooth -= kappastar * 2 * (corner.re + z.re);
		}
	}
	else
	{
		alpha_smooth = -kappastar * z;
	}

	return alpha_smooth;
}

/******************************************************************************
lens equation from image plane to source plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param nstars -- number of point mass lenses in array
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return w = (1 - kappa) * z + gamma * z_bar 
            - alpha_star - alpha_local - alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> complex_image_to_source(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> alpha_star = star_deflection(z, theta, stars, node);
	Complex<T> alpha_local = local_deflection(z, theta, node);
	Complex<T> alpha_smooth = smooth_deflection(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	(1 - kappa) * z + gamma * z_bar - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return (1 - kappa) * z + gamma * z.conj() - alpha_star - alpha_local - alpha_smooth;
}

/******************************************************************************
complex point in the source plane converted to pixel position

\param w -- complex source plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels per side for the source plane receiving
				  region

\return (w + hly * (1 + i)) * npixels / (2 * hly)
******************************************************************************/
template <typename T, typename U>
__device__ Complex<T> point_to_pixel(Complex<U> w, U hly, int npixels)
{
	Complex<T> result((w + hly * Complex<U>(1, 1)) * npixels / (2 * hly));
	return result;
}

/******************************************************************************
shoot rays from image plane to source plane

\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param nodes -- pointer to tree
\param level -- level at which to access the nodes
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param hlx -- half length of the image plane shooting region
\param numrayblocks -- number of ray blocks for the image plane shooting region
\param raysep -- separation between central rays of shooting squares
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
******************************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth,
	Complex<T> hlx, Complex<int> numrayblocks, T raysep, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < numrayblocks.re; i += x_stride)
	{
		for (int j = y_index; j < numrayblocks.im; j += y_stride)
		{
			/******************************************************************************
			x = image plane, y = source plane
			******************************************************************************/
			Complex<T> x[4];
			Complex<T> y[4];

			/******************************************************************************
			location of central ray in image plane and nearest node
			******************************************************************************/
			Complex<T> z = -hlx + raysep / 2 * Complex<T>(1, 1) + raysep * Complex<T>(i, j);
			TreeNode<T>* node = treenode::get_nearest_node(z, root);

			/******************************************************************************
			shooting rays in image plane at center +/- 1/2 * distance to next central ray
			in x1 and x2 direction
			******************************************************************************/
			T dx = raysep / 2;

			x[0] = z + Complex<T>(dx, dx);
			x[1] = z + Complex<T>(-dx, dx);
			x[2] = z + Complex<T>(-dx, -dx);
			x[3] = z + Complex<T>(dx, -dx);

			/******************************************************************************
			map rays from image plane to source plane
			******************************************************************************/
#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				y[k] = complex_image_to_source(x[k], kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
			}

			/******************************************************************************
			calculate local Taylor coefficients of the potential
			relies on symmetries and the fact that there are no higher order
			 macro-derivatives than kappa and gamma to be able to calculate down to the 4th
			 derivatives of the potential with our 4 rays shot
			assumes rays shot lie within the rectangle of stars, thus removing any boxcar
			 and heaviside functions
			******************************************************************************/

			T l_p1 = (y[0].re + y[1].re + y[2].re + y[3].re) / -4;
			T l_p2 = (y[0].im + y[1].im + y[2].im + y[3].im) / -4;

			T l_p11 = (kappa - kappastar) + (-y[0].re + y[1].re + y[2].re - y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx);
			T l_p12 = (-y[0].re - y[1].re + y[2].re + y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx);

			T l_p111 = (y[0].im - y[1].im + y[2].im - y[3].im) / (4 * dx * dx);
			T l_p112 = (-y[0].re + y[1].re - y[2].re + y[3].re) / (4 * dx * dx);

			T l_p1111 = 3 * (8 * dx * (kappa - kappastar - 1) + y[0].re - y[1].re - y[2].re + y[3].re + y[0].im + y[1].im - y[2].im - y[3].im) / (8 * dx * dx * dx);
			T l_p1112 = -3 * (y[0].re + y[1].re - y[2].re - y[3].re - y[0].im + y[1].im + y[2].im - y[3].im) / (8 * dx * dx * dx);

			/******************************************************************************
			divide distance between rays by NUM_RESAMPLED_RAYS
			this gives us an increase in ray density of NUM_RESAMPLED_RAYS per unit length,
			 so NUM_RESAMPLED_RAYS^2 per unit area
			these rays will use Taylor coefficients rather than being directly shot
			******************************************************************************/
			dx = raysep / NUM_RESAMPLED_RAYS;

			T dx1;
			T dx2;
			T y1;
			T y2;
			T invmag11;
			T invmag12;
			T invmag;
			Complex<int> ypix;
			for (int k = -HALF_NUM_RESAMPLED_RAYS; k <= HALF_NUM_RESAMPLED_RAYS; k++)
			{
				for (int l = -HALF_NUM_RESAMPLED_RAYS; l <= HALF_NUM_RESAMPLED_RAYS; l++)
				{
					dx1 = dx * k;
					dx2 = dx * l;

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

					ypix = point_to_pixel<int, T>(Complex<T>(y1, y2), hly, npixels);

					/******************************************************************************
					account for possible rounding issues when converting to integer pixels
					******************************************************************************/
					if (ypix.re == npixels)
					{
						ypix.re--;
					}
					if (ypix.im == npixels)
					{
						ypix.im--;
					}

					invmag11 = 1 - l_p11 - (l_p111 * dx1 + l_p112 * dx2)
						- (l_p1111 * (dx1 * dx1 - dx2 * dx2) + 2 * l_p1112 * dx1 * dx2) / 2;
					invmag12 = -l_p12 - (l_p112 * dx1 - l_p111 * dx2)
						- (l_p1112 * (dx1 * dx1 - dx2 * dx2) - 2 * l_p1111 * dx1 * dx2) / 2;
					invmag = invmag11 * (2 * (1 - kappa + kappastar) - invmag11) - invmag12 * invmag12;

					/******************************************************************************
					reverse y coordinate so array forms image in correct orientation
					******************************************************************************/
					ypix.im = npixels - 1 - ypix.im;

					if (invmag > 0)
					{
						if (pixmin)
						{
							atomicAdd(&pixmin[ypix.im * npixels + ypix.re], 1);
						}
						atomicAdd(&pixels[ypix.im * npixels + ypix.re], 1);
					}
					else if (invmag < -0)
					{
						if (pixsad)
						{
							atomicAdd(&pixsad[ypix.im * npixels + ypix.re], 1);
						}
						atomicAdd(&pixels[ypix.im * npixels + ypix.re], 1);
					}
					else
					{
						if (pixmin)
						{
							atomicAdd(&pixmin[ypix.im * npixels + ypix.re], 1);
						}
						if (pixsad)
						{
							atomicAdd(&pixsad[ypix.im * npixels + ypix.re], 1);
						}
						atomicAdd(&pixels[ypix.im * npixels + ypix.re], 2);
					}
				}
			}
		}
	}
}


/******************************************************************************
initialize array of pixels to 0

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
******************************************************************************/
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

/******************************************************************************
calculate the minimum and maximum number of rays in the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minrays -- pointer to minimum number of rays
\param maxrays -- pointer to maximum number of rays
******************************************************************************/
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

/******************************************************************************
initialize histogram values to 0

\param histogram -- pointer to histogram
\param n -- length of histogram
******************************************************************************/
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

/******************************************************************************
calculate the histogram of rays for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param minrays -- minimum number of rays
\param histogram -- pointer to histogram
******************************************************************************/
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
			atomicAdd(&histogram[pixels[j * npixels + i] - minrays], 1);
		}
	}
}

/******************************************************************************
write array of values to disk

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_array(T* vals, int nrows, int ncols, const std::string& fname)
{
	std::filesystem::path fpath = fname;

	if (fpath.extension() != ".bin")
	{
		std::cerr << "Error. File " << fname << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;

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

	return true;
}

/******************************************************************************
write histogram

\param histogram -- pointer to histogram
\param n -- length of histogram
\param minrays -- minimum number of rays
\param fname -- location of the file to write to

\return bool -- true if file is successfully written, false if not
******************************************************************************/
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

