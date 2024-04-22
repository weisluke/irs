#pragma once

#include "alpha_local.cuh"
#include "alpha_smooth.cuh"
#include "alpha_star.cuh"
#include "array_functions.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util.cuh"


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
	Complex<T> center_y, Complex<T> hly, int* pixmin, int* pixsad, int* pixels, Complex<int> npixels, int* percentage)
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
			for (int i = threadIdx.x; i < node->num_neighbors; i += blockDim.x)
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
				node->num_neighbors = 0;
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
					if (w.re < -hly.re || w.re > hly.re || w.im < -hly.im || w.im > hly.im)
					{
						continue;
					}

					ypix = point_to_pixel<int, T>(w, hly, npixels);

					/******************************************************************************
					account for possible rounding issues when converting to integer pixels
					******************************************************************************/
					if (ypix.re == npixels.re)
					{
						ypix.re--;
					}
					if (ypix.im == npixels.im)
					{
						ypix.im--;
					}

					/******************************************************************************
					reverse y coordinate so array forms image in correct orientation
					******************************************************************************/
					ypix.im = npixels.im - 1 - ypix.im;

					if (pixmin && pixsad)
					{
						T mu = magnification<T>(z, kappa, gamma, theta, tmp_stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
						if (mu >= 0)
						{
							atomicAdd(&pixmin[ypix.im * npixels.re + ypix.re], 1);
						}
						else
						{
							atomicAdd(&pixsad[ypix.im * npixels.re + ypix.re], 1);
						}
					}
					atomicAdd(&pixels[ypix.im * npixels.re + ypix.re], 1);

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

