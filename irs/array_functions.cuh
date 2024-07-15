#pragma once

#include "complex.cuh"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>


/******************************************************************************
initialize array of values to 0

\param vals -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
******************************************************************************/
template <typename T>
__global__ void initialize_array_kernel(T* vals, int nrows, int ncols)
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
transpose array

\param z1 -- pointer to array of values
\param nrows -- number of rows in array
\param ncols -- number of columns in array
\param z2 -- pointer to transposed array of values
******************************************************************************/
template <typename T>
__global__ void transpose_array_kernel(T* z1, int nrows, int ncols, T* z2)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int a = x_index; a < nrows * ncols; a += x_stride)
	{
		int col = a % ncols;
		int row = (a - col) / ncols;

		z2[col * nrows + row] = z1[a];
	}
}

/******************************************************************************
complex point in the source plane converted to pixel position

\param w -- complex source plane position
\param hly -- half length of the source plane receiving region
\param npixels -- number of pixels per side for the source plane receiving
				  region

\return (w + hly) * npixels / (2 * hly)
******************************************************************************/
template <typename T, typename U>
__device__ Complex<T> point_to_pixel(Complex<U> w, Complex<U> hly, Complex<int> npixels)
{
	Complex<T> result((w + hly).re * npixels.re / (2 * hly.re), (w + hly).im * npixels.im / (2 * hly.im));
	return result;
}

/******************************************************************************
add two arrays together

\param arr1 -- pointer to array of values
\param arr2 -- pointer to array of values
\param arr3 -- pointer to array of sum
\param nrows -- number of rows in array
\param ncols -- number of columns in array
******************************************************************************/
template <typename T>
__global__ void add_arrays_kernel(T* arr1, T* arr2, T* arr3, int nrows, int ncols)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < ncols; i += x_stride)
	{
		for (int j = y_index; j < nrows; j += y_stride)
		{
			arr3[j * ncols + i] = arr1[j * ncols + i] + arr2[j * ncols + i];
		}
	}
}

/******************************************************************************
calculate the histogram of rays for the pixel array

\param pixels -- pointer to array of pixels
\param npixels -- number of pixels for one side of the receiving square
\param hist_min -- minimum value in the histogram
\param histogram -- pointer to histogram
\param factor -- factor by which to multiply the pixel values before casting
                 to integers for the histogram
******************************************************************************/
template <typename T>
__global__ void histogram_kernel(T* pixels, Complex<int> npixels, int hist_min, int* histogram, int factor = 1)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	int y_index = blockIdx.y * blockDim.y + threadIdx.y;
	int y_stride = blockDim.y * gridDim.y;

	for (int i = x_index; i < npixels.re; i += x_stride)
	{
		for (int j = y_index; j < npixels.im; j += y_stride)
		{
			atomicAdd(&histogram[static_cast<int>(pixels[j * npixels.re + i] * factor + 0.5 - hist_min)], 1);
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
bool write_histogram(T* histogram, int n, int minrays, const std::string& fname)
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

