#pragma once

#include "mass_function_base.cuh"

#include <cmath>


namespace massfunctions{
    
/******************************************************************************
template class for handling a population of point mass lenses consisting of two
different masses which each contribute equally to the convergence
******************************************************************************/
template <typename T>
class OpticalDepth : public MassFunction<T>
{
	
public:

    /******************************************************************************
	calculate mass drawn from a mass function given a probability.

	\param p -- number drawn uniformly in [0,1]
	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param m_solar -- solar mass in arbitrary units
	******************************************************************************/
	__host__ __device__ T mass(T p, T m_lower, T m_upper, T m_solar) override
	{
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		if (p <= m_upper / (m_lower + m_upper))
		{
			return m_lower;
		}
		return m_upper;
	}

	/******************************************************************************
	calculate <mass> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param m_solar -- solar mass in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass(T m_lower, T m_upper, T m_solar) override
	{
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		return 2 * m_lower * m_upper / (m_lower + m_upper);
	}

	/******************************************************************************
	calculate <mass^2> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param m_solar -- solar mass in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass2(T m_lower, T m_upper, T m_solar) override
	{
		if (m_lower == m_upper)
		{
			return m_lower * m_lower;
		}

		return m_lower * m_upper;
	}

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param m_solar -- solar mass in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass2_ln_mass(T m_lower, T m_upper, T m_solar) override
    {
		if (m_lower == m_upper)
		{
			return m_lower * m_lower * std::log(m_lower);
		}

		return m_lower * m_upper * (m_lower * std::log(m_lower) + m_upper * std::log(m_upper)) / (m_lower + m_upper);
    }

};

}

