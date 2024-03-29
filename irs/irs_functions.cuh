#pragma once

#include "alpha_local.cuh"
#include "alpha_smooth.cuh"
#include "alpha_star.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************************************************************
lens equation from image plane to source plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
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
	Complex<T> a_star = alpha_star<T>(z, theta, stars, node);
	Complex<T> a_local = alpha_local<T>(z, theta, node);
	Complex<T> a_smooth = alpha_smooth<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	(1 - kappa) * z + gamma * z_bar - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return (1 - kappa) * z + gamma * z.conj() - a_star - a_local - a_smooth;
}

/******************************************************************************
magnification at a point in the image plane

\param z -- complex image plane position
\param kappa -- total convergence
\param gamma -- external shear
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param node -- node within which to calculate the deflection angle
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
						approximate

\return mu = ( (dw / dz)^2 - dw/dz * (dw/dz)bar ) ^ -1
******************************************************************************/
template <typename T>
__device__ T magnification(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d_a_star_d_zbar = d_alpha_star_d_zbar<T>(z, theta, stars, node);
	Complex<T> d_a_local_d_zbar = d_alpha_local_d_zbar<T>(z, theta, node);
	T d_a_smooth_d_z = d_alpha_smooth_d_z<T>(z, kappastar, rectangular, corner, approx);
	Complex<T> d_a_smooth_d_zbar = d_alpha_smooth_d_zbar<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	T d_w_d_z = (1 - kappa) - d_a_smooth_d_z;
	Complex<T> d_w_d_zbar = gamma - d_a_star_d_zbar - d_a_local_d_zbar - d_a_smooth_d_zbar;

	T mu_inv = d_w_d_z * d_w_d_z - d_w_d_zbar.abs() * d_w_d_zbar.abs();

	return 1 / mu_inv;
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
\param center_x -- center of the image plane shooting region
\param hlx -- half length of the image plane shooting region
\param numrayblocks -- number of ray blocks for the image plane shooting region
\param raysep -- separation between central rays of shooting squares
\param center_y -- center of the source plane receiving region
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
******************************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, int num_rays_factor, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth,
	Complex<T> center_x, Complex<T> hlx, Complex<int> numrayblocks, 
	Complex<T> center_y, T hly, int* pixmin, int* pixsad, int* pixels, int npixels, int* percentage)
{
	Complex<T> block_half_length = Complex<T>(hlx.re / numrayblocks.re, hlx.im / numrayblocks.im);
	
	extern __shared__ int shared_memory[];
	TreeNode<T>* node = reinterpret_cast<TreeNode<T>*>(&shared_memory[0]);
	star<T>* tmp_stars = reinterpret_cast<star<T>*>(&node[1]);
	__shared__ int nstars;

	for (int l = blockIdx.y; l < numrayblocks.im; l += gridDim.y)
	{
		for (int k = blockIdx.x; k < numrayblocks.re; k += gridDim.x)
		{
			Complex<T> block_center = center_x - hlx + block_half_length + 2 * Complex<T>(block_half_length.re * k, block_half_length.im * l);
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
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
					w = complex_image_to_source<T>(z, kappa, gamma, theta, tmp_stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
					
					/******************************************************************************
					if the ray location is the same as a star position, we will have a nan returned
					******************************************************************************/
					if (isnan(w.re) || isnan(w.im))
					{
						continue;
					}

					/******************************************************************************
					shift ray position relative to center
					******************************************************************************/
					w -= center_y;

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

					if (pixmin && pixsad)
					{
						T mu = magnification<T>(z, kappa, gamma, theta, tmp_stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
						if (mu >= 0)
						{
							atomicAdd(&pixmin[ypix.im * npixels + ypix.re], 1);
						}
						else
						{
							atomicAdd(&pixsad[ypix.im * npixels + ypix.re], 1);
						}
					}
					atomicAdd(&pixels[ypix.im * npixels + ypix.re], 1);

				}
			}
			__syncthreads();
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				int p = atomicAdd(percentage, 1);
				if (p * 100 / (numrayblocks.re * numrayblocks.im) > (p - 1) * 100 / (numrayblocks.re * numrayblocks.im))
				{
					device_print_progress(p, numrayblocks.re * numrayblocks.im);
				}
			}
		}
	}
}


/******************************************************************************
initialize array of values to 0

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
******************************************************************************/
template <typename T>
__global__ void initialize_array_kernel(int* vals, int nrows, int ncols)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < ncols; i += x_stride)
	{
		for (int j = y_index; j < nrows; j += y_stride)
		{
			vals[j * ncols + i] = 0;
		}
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

