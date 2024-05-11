#pragma once

#include "mass_function_base.cuh"


namespace massfunctions{
    
/******************************************************************************
template class for handling equal mass point mass lenses
******************************************************************************/
template <typename T>
class Equal : public MassFunction<T>
{
	
public:

    /******************************************************************************
	calculate mass drawn from a mass function given a probability.

	\param p -- number drawn uniformly in [0,1]

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T mass(T p, ...)
	{
		return 1;
	}

	/******************************************************************************
	calculate <mass> for a mass function

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T mean_mass(...)
	{
		return 1;
	}

	/******************************************************************************
	calculate <mass^2> for a mass function

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T mean_mass2(...)
	{
		return 1;
	}

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function

    \return 0, as all masses are the same
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2_ln_mass(...)
    {
        return 0;
    }

};

}

