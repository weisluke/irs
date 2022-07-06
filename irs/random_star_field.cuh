#pragma once

#include "complex.cuh"
#include "star.cuh"

#include <chrono>
#include <cmath>
#include <random>


/**********************************************************
generate random star field

\param stars -- pointer to array of stars
\param nstars -- number of stars to generate
\param hlx1 -- side length 1 within which to generate stars
\param hlx2 -- side length 2 within which to generate stars
\param mass -- mass for each star
\param seed -- random seed to use. defaults to seed
			   generated based on current time

\return seed -- the random seed used
**********************************************************/
template <typename T>
int generate_star_field(star<T>* stars, int nstars, T hlx1, T hlx2, T mass, int seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()))
{
	/*random number generator seeded according to the provided seed*/
	std::mt19937 gen(seed);

	/*uniform distribution to pick real values between 0 and 1*/
	std::uniform_real_distribution<T> dis(0, 1);

	/*variables to hold randomly chosen x1 and x2*/
	T x1, x2;

	for (int i = 0; i < nstars; i++)
	{
		x1 = dis(gen) * 2.0 * hlx1 - hlx1;
		x2 = dis(gen) * 2.0 * hlx2 - hlx2;

		stars[i].position = Complex<T>(x1, x2);
		stars[i].mass = mass;
	}

	return seed;
}

