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
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if 
                        approximate
\param ray_half_sep -- half separation between central rays of shooting squares
\param num_ray_threads -- number of threads of rays for the image plane shooting region
\param center_x -- center of the image plane shooting region
\param hlx -- half length of the image plane shooting region
\param center_y -- center of the source plane receiving region
\param hly -- half length of the source plane receiving region
\param pixmin -- pointer to array of positive parity pixels
\param pixsad -- pointer to array of negative parity pixels
\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
******************************************************************************/
template <typename T>
__global__ void shoot_rays_kernel(T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* root, 
	int rectangular, Complex<T> corner, int approx, int taylor_smooth,
	Complex<T> ray_half_sep, Complex<int> num_ray_threads, Complex<T> center_x, Complex<T> hlx, 
	Complex<T> center_y, Complex<T> hly, int* pixmin, int* pixsad, int* pixels, Complex<int> npixels, unsigned long long int* percentage)
{
	for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < num_ray_threads.im; j += blockDim.y * gridDim.y)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_ray_threads.re; i += blockDim.x * gridDim.x)
		{
			Complex<T> z = -hlx + ray_half_sep + 2 * Complex<T>(ray_half_sep.re * i, ray_half_sep.im * j);
			TreeNode<T>* node = treenode::get_nearest_node(z, root);
			Complex<T> w = complex_image_to_source<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
			
			/******************************************************************************
			if the ray location is the same as a star position, we will have a nan returned
			******************************************************************************/
			if (isnan(w.re) || isnan(w.im))
			{
				if (threadIdx.x == 0 && threadIdx.y == 0)
				{
					unsigned long long int p = atomicAdd(percentage, 1);
					unsigned long long int imax = ((num_ray_threads.re - 1) / blockDim.x + 1);
					imax *= ((num_ray_threads.im - 1) / blockDim.y + 1);
					if (p * 100 / imax > (p - 1) * 100 / imax)
					{
						device_print_progress(p, imax);
					}
				}
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
				if (threadIdx.x == 0 && threadIdx.y == 0)
				{
					unsigned long long int p = atomicAdd(percentage, 1);
					unsigned long long int imax = ((num_ray_threads.re - 1) / blockDim.x + 1);
					imax *= ((num_ray_threads.im - 1) / blockDim.y + 1);
					if (p * 100 / imax > (p - 1) * 100 / imax)
					{
						device_print_progress(p, imax);
					}
				}
				continue;
			}

			Complex<int> ypix = point_to_pixel<int, T>(w, hly, npixels);

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
				T mu = magnification<T>(z, kappa, gamma, theta, stars, kappastar, node, rectangular, corner, approx, taylor_smooth);
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
			
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				unsigned long long int p = atomicAdd(percentage, 1);
				unsigned long long int imax = ((num_ray_threads.re - 1) / blockDim.x + 1);
				imax *= ((num_ray_threads.im - 1) / blockDim.y + 1);
				if (p * 100 / imax > (p - 1) * 100 / imax)
				{
					device_print_progress(p, imax);
				}
			}
		}
	}
}

