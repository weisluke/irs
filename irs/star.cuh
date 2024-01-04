﻿#pragma once

#include "complex.cuh"
#include "mass_function.cuh"
#include "util.cuh"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <cmath>
#include <cstdint> //for std::uintmax_t
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits> //for std::numeric_limits
#include <new>
#include <string>
#include <system_error> //for std::error_code


/******************************************************************************
structure to hold position and mass of a star
******************************************************************************/
template <typename T>
struct star
{
	Complex<T> position;
	T mass;
};


/******************************************************************************
initialize curand states for random star field generation

\param states -- pointer to array of curand states
\param nstars -- number of states (stars) to initialize
\param seed -- random seed to use
******************************************************************************/
template<typename T>
__global__ void initialize_curand_states_kernel(curandState* states, int nstars, int seed)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	for (int i = x_index; i < nstars; i += x_stride)
	{
		curand_init(seed, i, 0, &states[i]);
	}
}

/******************************************************************************
generate random star field

\param states -- pointer to array of curand states
\param stars -- pointer to array of point mass lenses
\param nstars -- number of stars to generate
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param mf -- enum for the mass function
\param msolar -- solar mass in arbitrary units
\param mL -- lower mass cutoff for the distribution in arbitrary units
\param mH -- upper mass cutoff for the distribution in arbitrary units
******************************************************************************/
template <typename T>
__global__ void generate_star_field_kernel(curandState* states, star<T>* stars, int nstars, int rectangular, Complex<T> corner, 
	massfunctions::massfunction mf = massfunctions::equal, T msolar = 1, T mL = 0.01, T mH = 50)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

	const T PI = static_cast<T>(3.1415926535898);

	for (int i = x_index; i < nstars; i += x_stride)
	{
		T x1;
		T x2;

		if (rectangular)
		{
			/******************************************************************************
			random positions in the range [-corner, corner]
			******************************************************************************/
			x1 = curand_uniform_double(&states[i]) * 2 * corner.re - corner.re;
			x2 = curand_uniform_double(&states[i]) * 2 * corner.im - corner.im;
		}
		else
		{
			/******************************************************************************
			random angle
			******************************************************************************/
			T a = curand_uniform_double(&states[i]) * 2 * PI;
			/******************************************************************************
			random radius uses square root of random number as numbers need to be evenly
			dispersed in 2-D space
			******************************************************************************/
			T r = sqrt(curand_uniform_double(&states[i])) * corner.abs();

			/******************************************************************************
			transform to Cartesian coordinates
			******************************************************************************/
			x1 = r * cos(a);
			x2 = r * sin(a);
		}

		stars[i].position = Complex<T>(x1, x2);
		stars[i].mass = MassFunction<T>(mf).mass(curand_uniform_double(&states[i]), msolar, mL, mH);
	}
}

/******************************************************************************
determines star field parameters from the given array

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param m_low -- lower mass cutoff
\param m_up -- upper mass cutoff
\param meanmass -- mean mass <m>
\param meanmass2 -- mean mass squared <m^2>
******************************************************************************/
template <typename T>
void calculate_star_params(int nstars, int rectangular, Complex<T> corner, T theta, star<T>* stars,
	T& kappastar, T& m_low, T& m_up, T& meanmass, T& meanmass2)
{
	const T PI = static_cast<T>(3.1415926535898);

	m_low = std::numeric_limits<T>::max();
	m_up = std::numeric_limits<T>::min();

	T mtot = 0;
	T m2tot = 0;

	for (int i = 0; i < nstars; i++)
	{
		mtot += stars[i].mass;
		m2tot += stars[i].mass * stars[i].mass;
		m_low = std::min(m_low, stars[i].mass);
		m_up = std::max(m_up, stars[i].mass);
	}
	meanmass = mtot / nstars;
	meanmass2 = m2tot / nstars;

	if (rectangular)
	{
		kappastar = mtot * PI * theta * theta / (4 * corner.re * corner.im);
	}
	else
	{
		kappastar = mtot * theta * theta / (corner.abs() * corner.abs());
	}
}

/******************************************************************************
read binary star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file_bin(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".bin")
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin file.\n";
		return false;
	}

	std::error_code err;
	std::uintmax_t fsize = std::filesystem::file_size(starfile, err);

	if (err)
	{
		std::cerr << "Error determining size of star input file " << starfile << "\n";
		return false;
	}

	std::ifstream infile;
	infile.open(starfile, std::ios_base::binary);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	/******************************************************************************
	first item in the file is the number of stars
	******************************************************************************/
	infile.read((char*)(&nstars), sizeof(int));
	if (nstars < 1)
	{
		std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
		return false;
	}

	/******************************************************************************
	allocate memory for stars if needed
	******************************************************************************/
	if (stars == nullptr)
	{
		cudaMallocManaged(&stars, nstars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
	}

	/******************************************************************************
	second item in the file is whether the star field is rectangular
	******************************************************************************/
	infile.read((char*)(&rectangular), sizeof(int));
	if (rectangular != 0 && rectangular != 1)
	{
		std::cerr << "Error. Invalid rectangular input. rectangular must be 1 (rectangular) or 0 (circular).\n";
		return false;
	}


	/******************************************************************************
	objects in the file are nstars + rectangular + corner + theta + stars
	******************************************************************************/
	if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<T>) + sizeof(T) +  nstars * sizeof(star<T>))
	{
		/******************************************************************************
		third item in the file is the corner of the star field
		******************************************************************************/
		infile.read((char*)(&corner), sizeof(Complex<T>));
		/******************************************************************************
		fourth item in the file is the size of the Einstein radius of a unit mass point
		lens
		******************************************************************************/
		infile.read((char*)(&theta), sizeof(T));

		infile.read((char*)stars, nstars * sizeof(star<T>));
	}
	else if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<float>) + sizeof(float) + nstars * sizeof(star<float>))
	{
		Complex<float> temp_corner;
		infile.read((char*)(&temp_corner), sizeof(Complex<float>));
		corner = temp_corner;

		float temp_theta;
		infile.read((char*)(&temp_theta), sizeof(float));
		theta = static_cast<T>(temp_theta);

		star<float>* temp_stars = new (std::nothrow) star<float>[nstars];
		if (!temp_stars)
		{
			std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
			return false;
		}
		infile.read((char*)temp_stars, nstars * sizeof(star<float>));
		for (int i = 0; i < nstars; i++)
		{
			stars[i].position = temp_stars[i].position;
			stars[i].mass = static_cast<T>(temp_stars[i].mass);
		}
		delete[] temp_stars;
		temp_stars = nullptr;
	}
	else if (fsize == sizeof(int) + sizeof(int) + sizeof(Complex<double>) + sizeof(double) + nstars * sizeof(star<double>))
	{
		Complex<double> temp_corner;
		infile.read((char*)(&temp_corner), sizeof(Complex<double>));
		corner = temp_corner;

		double temp_theta;
		infile.read((char*)(&temp_theta), sizeof(double));
		theta = static_cast<T>(temp_theta);

		star<double>* temp_stars = new (std::nothrow) star<double>[nstars];
		if (!temp_stars)
		{
			std::cerr << "Error. Memory allocation for *temp_stars failed.\n";
			return false;
		}
		infile.read((char*)temp_stars, nstars * sizeof(star<double>));
		for (int i = 0; i < nstars; i++)
		{
			stars[i].position = temp_stars[i].position;
			stars[i].mass = static_cast<T>(temp_stars[i].mass);
		}
		delete[] temp_stars;
		temp_stars = nullptr;
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " does not contain validly formatted single or double precision stars and accompanying information.\n";
		return false;
	}

	infile.close();

	return true;
}

/******************************************************************************
read txt star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file_txt(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".txt")
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .txt file.\n";
		return false;
	}

	/******************************************************************************
	set parameters that will be modified based on star input file
	******************************************************************************/
	nstars = 0;
	T x1, x2, m;

	std::ifstream infile;
	infile.open(starfile);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	while (infile >> x1 >> x2 >> m)
	{
		nstars++;
	}
	infile.close();

	if (nstars < 1)
	{
		std::cerr << "Error. Invalid num_stars input. num_stars must be an integer > 0\n";
		return false;
	}


	/******************************************************************************
	allocate memory for stars if needed
	******************************************************************************/
	if (stars == nullptr)
	{
		cudaMallocManaged(&stars, nstars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
	}


	/******************************************************************************
	set parameters that will be modified based on star input file
	******************************************************************************/
	T max_x1 = 0;
	T max_x2 = 0;
	T max_rad = 0;
	T total_mass = 0;
	T I_stars = 0;

	std::ifstream infile;
	infile.open(starfile);
	if (!infile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	for	(int i = 0; i < nstars; i++)
	{
		infile >> x1 >> x2 >> m;
		stars[i].position = Complex<T>(x1, x2);
		stars[i].mass = m;
		total_mass += m;
		I_stars += m * (x1 * x1 + x2 * x2); //moment of inertia of a point mass
		max_x1 = std::max(max_x1, std::abs(x1));
		max_x2 = std::max(max_x2, std::abs(x2));
		max_rad = std::max(max_rad, std::sqrt(x1 * x1 + x2 * x2));
	}
	infile.close();

	/******************************************************************************
	determine whether a star field is circular or rectangular by using the moment
	of inertia
	******************************************************************************/
	T I_circ = total_mass * max_rad * max_rad / 2; //moment of inertia of a disk of radius max_rad
	T I_rect = total_mass * (max_x1 * max_x1 + max_x2 * max_x2) / 3; //moment of inertia of a rectangle with corner (max_x1, max_x2)

	if (I_stars / I_circ > I_stars / I_rect)
	{
		rectangular = 0;
		corner.re = max_rad;
		corner.im = 0;
	}
	else
	{
		rectangular = 1;
		corner.re = max_x1;
		corner.im = max_x2;
	}

	theta = 1; //assume all positions are in units of the Einstein radius of a unit mass
	
	return true;
}

/******************************************************************************
read star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param kappastar -- convergence in point mass lenses
\param m_low -- lower mass cutoff
\param m_up -- upper mass cutoff
\param meanmass -- mean mass <m>
\param meanmass2 -- mean mass squared <m^2>
\param starfile -- location of the star field file

\return bool -- true if file is successfully read, false if not
******************************************************************************/
template <typename T>
bool read_star_file(int& nstars, int& rectangular, Complex<T>& corner, T& theta, star<T>*& stars,
	T& kappastar, T& m_low, T& m_up, T& meanmass, T& meanmass2, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() == ".bin")
	{
		if !(read_star_file_bin(nstars, rectangular, corner, theta, stars, starfile))
		{
			return false;
		}
	}
	else if (starpath.extension() == ".txt")
	{
		if !(read_star_file_txt(nstars, rectangular, corner, theta, stars, starfile))
		{
			return false;
		}
	}
	else
	{
		std::cerr << "Error. Star input file " << starfile << " is not a .bin or .txt file.\n";
		return false;
	}

	calculate_star_params<T>(nstars, rectangular, corner, theta, stars, kappastar, m_low, m_up, meanmass, meanmass2);

	return true;
}

/******************************************************************************
write star field file

\param nstars -- number of point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the field of point mass
				 lenses
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param starfile -- location of the star field file

\return bool -- true if file is successfully written, false if not
******************************************************************************/
template <typename T>
bool write_star_file(int nstars, int rectangular, Complex<T> corner, T theta, star<T>* stars, const std::string& starfile)
{
	std::filesystem::path starpath = starfile;

	if (starpath.extension() != ".bin")
	{
		std::cerr << "Error. Star file " << starfile << " is not a .bin file.\n";
		return false;
	}

	std::ofstream outfile;
	outfile.open(starfile, std::ios_base::binary);

	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << starfile << "\n";
		return false;
	}

	outfile.write((char*)(&nstars), sizeof(int));
	outfile.write((char*)(&rectangular), sizeof(int));
	outfile.write((char*)(&corner), sizeof(Complex<T>));
	outfile.write((char*)(&theta), sizeof(T));
	outfile.write((char*)stars, nstars * sizeof(star<T>));
	outfile.close();

	return true;
}

