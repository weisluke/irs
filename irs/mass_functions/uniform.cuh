#pragma once

#include "power_law.cuh"


namespace massfunctions{
    
/******************************************************************************
template class for handling point mass lenses with a uniform distribution
******************************************************************************/
template <typename T>
class Uniform : public PowerLaw<T>
{
	
public:

	/******************************************************************************
	a uniform distribution is a power law with slope 0
	no further functions needed as they are not modified from the base class
	******************************************************************************/
	__host__ __device__ Uniform() : PowerLaw<T>(0) {};

};

}

