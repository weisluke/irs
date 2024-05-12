#pragma once

#include "power_law.cuh"


namespace massfunctions{
    
/******************************************************************************
template class for handling point mass lenses following a salpeter distribution
******************************************************************************/
template <typename T>
class Salpeter : public PowerLaw<T>
{	
	
public:

	/******************************************************************************
	a salpeter distribution is a power law with slope -2.35 (though this can be
	changed if necessary since the slope is public)
	no further functions needed as they are not modified from the base class
	******************************************************************************/
	__host__ __device__ Salpeter(T a = -2.35) : PowerLaw<T>(a) {};

};

}

