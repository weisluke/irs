#pragma once


namespace massfunctions{
	
/******************************************************************************
template class for handling the mass functions of point mass lenses
******************************************************************************/
template <typename T>
class MassFunction
{

public:

	/******************************************************************************
	calculate mass drawn from a mass function given a probability.

	\param p -- number drawn uniformly in [0,1]
	******************************************************************************/
	__host__ __device__ virtual T mass(T p, ...) {} = 0;

	/******************************************************************************
	calculate <mass> for a mass function
	******************************************************************************/
	__host__ __device__ virtual T mean_mass(...) {} = 0;

	/******************************************************************************
	calculate <mass^2> for a mass function
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2(...) {} = 0;

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2_ln_mass(...) {} = 0;

};

}

