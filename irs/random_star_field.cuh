#pragma once

#include "complex.cuh"
#include "star.cuh"

#include <chrono>
#include <cmath>
#include <random>


/**************************************************
generate random star field

\param stars -- pointer to array of stars
\param nstars -- number of stars to generate
\param rad -- radius within which to generate stars
\param mass -- mass for each star
\param seed -- random seed to use. defaults to seed
			   generated based on current time

\return seed -- the random seed used
**************************************************/
template <typename T>
int generate_star_field(star<T>* stars, int nstars, T rad, T mass, int seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
	const T PI = 3.1415926535898;

	/*random number generator seeded according to current time*/
	std::mt19937 gen(seed);

	/*uniform distribution to pick real values between 0 and 1*/
	std::uniform_real_distribution<T> dis(0, 1);

	/*variables to hold randomly chosen angle and radius*/
	T a, r;

	for (int i = 0; i < nstars; i++)
	{
		/*random angle and radius*/
		a = dis(gen) * 2.0 * PI;
		/*radius uses square root of random number
		as numbers need to be evenly dispersed in 2-D space*/
		r = std::sqrt(dis(gen)) * rad;

		/*transform to Cartesian coordinates*/
		stars[i].position = Complex<T>(r * std::cos(a), r * std::sin(a));
		stars[i].mass = mass;
	}

	return seed;
}

