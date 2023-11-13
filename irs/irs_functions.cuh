#pragma once

#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"
#include "util.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************************************************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x > 0, 0 if x <= 0
******************************************************************************/
template <typename T>
__device__ T heaviside(T x)
{
	if (x > 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within the rectangle defined by corner, 0 if it is on the
		border or outside
******************************************************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re < z.re && z.re < corner.re && -corner.im < z.im && z.im < corner.im)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

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
calculate the deriviative of the deflection angle due to nearby stars for a
node with respect to zbar

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- node within which to calculate the deflection angle

\return d_alpha_star_d_zbar = -theta^2 * sum(m_i / (z - z_i)_bar^2)
******************************************************************************/
template <typename T>
__device__ Complex<T> d_star_deflection_d_zbar(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> d_alpha_star_bar_d_z;

	/******************************************************************************
	theta^2 * sum(m_i / (z - z_i))
	******************************************************************************/
	for (int i = 0; i < node->numstars; i++)
	{
		d_alpha_star_bar_d_z += stars[node->stars + i].mass / ((z - stars[node->stars + i].position) * (z - stars[node->stars + i].position));
	}
	for (int j = 0; j < node->numneighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			d_alpha_star_bar_d_z += stars[neighbor->stars + i].mass / ((z - stars[neighbor->stars + i].position) * (z - stars[neighbor->stars + i].position));
		}
	}
	d_alpha_star_bar_d_z *= -(theta * theta);

	return d_alpha_star_bar_d_z.conj();
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
calculate the derivative of the deflection angle due to far away stars for a
node with respect to zbar

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param node -- node within which to calculate the deflection angle

\return alpha_local = theta^2 * sum(i * a_i * (z - z_0) ^ (i - 1))
		   where a_i are coefficients of the lensing potential in units of the
		   node size
******************************************************************************/
template <typename T>
__device__ Complex<T> d_local_deflection_d_zbar(Complex<T> z, T theta, TreeNode<T>* node)
{
	Complex<T> d_alpha_local_bar_dz;
	Complex<T> dz = (z - node->center) / node->half_length;

	for (int i = node->expansion_order; i >= 2; i--)
	{
		d_alpha_local_bar_dz *= dz;
		d_alpha_local_bar_dz += node->local_coeffs[i] * i * (i - 1);
	}
	d_alpha_local_bar_dz *= (theta * theta);
	/******************************************************************************
	account for node size
	******************************************************************************/
	d_alpha_local_bar_dz /= (node->half_length * node->half_length);

	return d_alpha_local_bar_dz.conj();
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
				s1 *= i;
				s1 += 1;
				s1 /= i;

				s2 *= i;
				s2 += 1;
				s2 /= i;

				s3 *= i;
				s3 += 1;
				s3 /= i;

				s4 *= i;
				s4 += 1;
				s4 /= i;

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

			alpha_smooth = (c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log());
			alpha_smooth *= Complex<T>(0, -kappastar / PI);
			alpha_smooth -= kappastar * 2 * (corner.re + z.re) * boxcar(z, corner);
			alpha_smooth -= kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);
		}
	}
	else
	{
		alpha_smooth = -kappastar * z;
	}

	return alpha_smooth;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to z

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not

\return d_alpha_smooth_d_z
******************************************************************************/
template <typename T>
__device__ T d_smooth_deflection_d_z(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx)
{
	T d_alpha_smooth_d_z = -kappastar;

	if (rectangular && !approx)
	{
		d_alpha_smooth_d_z *= boxcar(z, corner);
	}

	return d_alpha_smooth_d_z;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to zbar

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return d_alpha_smooth_d_zbar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_smooth_deflection_d_zbar(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> d_alpha_smooth_d_zbar;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r1 = z.conj() / corner;
			Complex<T> r2 = z.conj() / corner.conj();

			Complex<T> s1;
			Complex<T> s2;

			for (int i = (taylor_smooth % 2 == 0 ? taylor_smooth : taylor_smooth - 1); i >= 2; i -= 2)
			{
				s1 *= i;
				s1 += 1;
				s1 /= i;

				s2 *= i;
				s2 += 1;
				s2 /= i;

				s1 *= (r1 * r1);
				s2 *= (r2 * r2);
			}
			d_alpha_smooth_d_zbar += s1 - s2;
			d_alpha_smooth_d_zbar *= 2;

			if (taylor_smooth % 2 == 0)
			{
				d_alpha_smooth_d_zbar += r1.pow(taylor_smooth) * 2;
				d_alpha_smooth_d_zbar -= r2.pow(taylor_smooth) * 2;
			}

			d_alpha_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_alpha_smooth_d_zbar += kappastar - 4 * kappastar * corner.arg() / PI;
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d_alpha_smooth_d_zbar = (c1.log() - c2.log() - c3.log() + c4.log());
			d_alpha_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_alpha_smooth_d_zbar -= kappastar * boxcar(z, corner);
		}
	}

	return d_alpha_smooth_d_zbar;
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
	Complex<T> alpha_star = star_deflection<T>(z, theta, stars, node);
	Complex<T> alpha_local = local_deflection<T>(z, theta, node);
	Complex<T> alpha_smooth = smooth_deflection<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	/******************************************************************************
	(1 - kappa) * z + gamma * z_bar - alpha_star - alpha_local - alpha_smooth
	******************************************************************************/
	return (1 - kappa) * z + gamma * z.conj() - alpha_star - alpha_local - alpha_smooth;
}

/******************************************************************************
magnification at a point in the image plane

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

\return mu = ( (dw / dz)^2 - dw/dz * (dw/dz)bar ) ^ -1
******************************************************************************/
template <typename T>
__device__ T magnification(Complex<T> z, T kappa, T gamma, T theta, star<T>* stars, T kappastar, TreeNode<T>* node,
	int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	Complex<T> d_alpha_star_d_zbar = d_star_deflection_d_zbar<T>(z, theta, stars, node);
	Complex<T> d_alpha_local_d_zbar = d_local_deflection_d_zbar<T>(z, theta, node);
	T d_alpha_smooth_d_z = d_smooth_deflection_d_z<T>(z, kappastar, rectangular, corner, approx);
	Complex<T> d_alpha_smooth_d_zbar = d_smooth_deflection_d_zbar<T>(z, kappastar, rectangular, corner, approx, taylor_smooth);

	T d_w_d_z = (1 - kappa) - d_alpha_smooth_d_z;
	Complex<T> d_w_d_zbar = gamma - d_alpha_star_d_zbar - d_alpha_local_d_zbar - d_alpha_smooth_d_zbar;

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
	Complex<T> hlx, Complex<int> numrayblocks, T hly, int* pixmin, int* pixsad, int* pixels, int npixels, int* percentage)
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
			Complex<T> block_center = -hlx + block_half_length + 2 * Complex<T>(block_half_length.re * k, block_half_length.im * l);
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

