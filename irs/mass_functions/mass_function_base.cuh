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
	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ virtual T mass(T p, T m_lower, T m_upper)
	{
		return 0;
	}

	/******************************************************************************
	calculate <mass> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ virtual T mean_mass(T m_lower, T m_upper)
	{
		return 0;
	}

	/******************************************************************************
	calculate <mass^2> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2(T m_lower, T m_upper)
	{
		return 0;
	}

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2_ln_mass(T m_lower, T m_upper)
	{
		return 0;
	}

};

}

