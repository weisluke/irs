#pragma once

#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


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
           where a_i are coefficients of the lensing potential in units of the
           node size
******************************************************************************/
template <typename T>
__device__ Complex<T> local_deflection(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> alpha_local_bar;
	Complex<T> dz = (z - node->center) / node->half_length;

	for (int i = node->expansion_order; i >= 1; i--)
	{
		alpha_local_bar *= dz;
		alpha_local_bar += node->local_coeffs[i] * i;
	}
	alpha_local_bar *= (theta * theta);
	/******************************************************************************
	account for node size 
	******************************************************************************/
	alpha_local_bar /= node->half_length;

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
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> alpha_smooth;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> s1;
			Complex<T> s2;
			Complex<T> s3;
			Complex<T> s4;

			for (int i = taylor_smooth; i >= 1; i--)
			{
				s1 += 1.0 / i;
				s2 += 1.0 / i;
				s3 += 1.0 / i;
				s4 += 1.0 / i;

				s1 *= (z.conj() / corner);
				s2 *= (z.conj() / corner.conj());
				s3 *= (z.conj() / -corner);
				s4 *= (z.conj() / -corner.conj());
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
\param root -- pointer to root node
\param num_rays_factor -- log2(number of rays per unit half length)
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
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, int num_rays_factor, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth,
	Complex<T> hlx, Complex<int> numrayblocks, T hly, int* pixmin, int* pixsad, int* pixels, int npixels)
{
	__shared__ Complex<T> block_half_length;
	__shared__ Complex<T> block_center;
	__shared__ TreeNode<T> node[1];
	__shared__ int nstars;
	__shared__ star<T> tmp_stars[treenode::MAX_NUM_STARS_DIRECT];

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		block_half_length = Complex<T>(hlx.re / numrayblocks.re, hlx.im / numrayblocks.im);
	}
	__syncthreads();

	for (int l = blockIdx.y; l < numrayblocks.im; l += gridDim.y)
	{
		for (int k = blockIdx.x; k < numrayblocks.re; k += gridDim.x)
		{
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				block_center = -hlx + block_half_length + 2 * Complex<T>(block_half_length.re * k, block_half_length.im * l);
				*node = *(treenode::get_nearest_node(block_center, root));
				nstars = 0;
			}
			__syncthreads();
			if (threadIdx.x == 0)
			{
				for (int j = threadIdx.y; j < node->numstars; j += blockDim.y)
				{
					tmp_stars[atomicAdd(&nstars, 1)] = stars[node->stars + j];
				}
			}
			for (int i = threadIdx.x; i < node->numneighbors; i += blockDim.x)
			{
				TreeNode<T>* neighbor = node->neighbors[i];
				for (int j = threadIdx.y; j < neighbor->numstars; j += blockDim.y)
				{
					tmp_stars[atomicAdd(&nstars, 1)] = stars[neighbor->stars + j];
				}
			}
			__syncthreads();

			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				node->numneighbors = 0;
				node->stars = 0;
				node->numstars = nstars;
			}
			__syncthreads();

			int num_rays = (2 << num_rays_factor);
			Complex<T> ray_half_sep = block_half_length / num_rays;
			Complex<int> ypix;
			Complex<T> z;
			Complex<T> w;
			for (int j = threadIdx.y; j < num_rays; j += blockDim.y)
			{
				for (int i = threadIdx.x; i < num_rays; i += blockDim.x)
				{
					z = block_center - block_half_length + ray_half_sep + 2 * Complex<T>(ray_half_sep.re * i, ray_half_sep.im * j);
					w = complex_image_to_source(z, kappa, gamma, theta, tmp_stars, kappastar, node, rectangular, corner, approx, taylor_smooth);

					/******************************************************************************
					if the ray landed outside the receiving region
					******************************************************************************/
					if (w.re < -hly || w.re > hly || w.im < -hly || w.im > hly)
					{
						continue;
					}

					ypix = point_to_pixel<int, T>(w, hly, npixels);

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

					/******************************************************************************
					reverse y coordinate so array forms image in correct orientation
					******************************************************************************/
					ypix.im = npixels - 1 - ypix.im;

					atomicAdd(&pixels[ypix.im * npixels + ypix.re], 1);

				}
			}
			__syncthreads();
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

