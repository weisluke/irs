#pragma once

#include "power_law.cuh"


namespace massfunctions{
    
/******************************************************************************
template class for handling equal mass point mass lenses
******************************************************************************/
template <typename T>
class Uniform : public PowerLaw<T>
{

private:
	T slope = static_cast<T>(0);
	
public:

    /******************************************************************************
	calculate mass drawn from a mass function given a probability.

	\param p -- number drawn uniformly in [0,1]
	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mass(T p, T m_lower, T m_upper, ...)
	{
		return PowerLaw::mass(p, m_lower, m_upper, slope);
	}

	/******************************************************************************
	calculate <mass> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass(T m_lower, T m_upper, ...)
	{
		return PowerLaw::mean_mass(m_lower, m_upper, slope);
	}

	/******************************************************************************
	calculate <mass^2> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass2(T m_lower, T m_upper, ...)
	{
		return PowerLaw::mean_mass2(m_lower, m_upper, slope);
	}

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2_ln_mass(T m_lower, T m_upper, ...)
    {
		return PowerLaw::mean_mass2_ln_mass(m_lower, m_upper, slope);
    }

};

}

