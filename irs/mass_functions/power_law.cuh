#pragma once

#include "mass_function_base.cuh"

#include <cmath>


namespace massfunctions{
    
/******************************************************************************
template class for handling point mass lenses following a power law spectrum
******************************************************************************/
template <typename T>
class PowerLaw : public MassFunction<T>
{

protected:
	/******************************************************************************
	calculate the integral of b * x^a from x1 to x2
	assumes 0 < x1 <= x2

	\param x1 -- lower bound of the integral
	\param x2 -- upper bound of the integral
	\param a -- exponent of x
	\param b -- coefficient of x^a

	\return b * (x2 ^ (a + 1) - x1 ^ (a + 1)) / (a + 1) if a =/= -1
			b * (log(x2) - log(x1)) if a == -1
	******************************************************************************/
	__host__ __device__ T power_integral(T x1, T x2, T a, T b = 1)
	{
		T result;
#ifdef CUDA_ARCH
		if (a != -1)
		{
			result = b * (pow(x2, a + 1) - pow(x1, a + 1)) / (a + 1);
		}
		else
		{
			result = b * (log(x2) - log(x1));
		}
#else
		if (a != -1)
		{
			result = b * (std::pow(x2, a + 1) - std::pow(x1, a + 1)) / (a + 1);
		}
		else
		{
			result = b * (std::log(x2) - std::log(x1));
		}
#endif
		return result;
	}

	/******************************************************************************
	calculate the integral of b * x^a * ln(x) from x1 to x2
	assumes 0 < x1 <= x2

	\param x1 -- lower bound of the integral
	\param x2 -- upper bound of the integral
	\param a -- exponent of x
	\param b -- coefficient of x^a

	\return b * (x2 ^ (a + 1) * ((a + 1) * log(x2) - 1) 
	             - x1 ^ (a + 1) * ((a + 1) * log(x1) - 1)) / (a + 1)^2 if a =/= -1
			b * (log(x2)^2 - log(x1)^2) / 2 if a == -1
	******************************************************************************/
	__host__ __device__ T power_log_integral(T x1, T x2, T a, T b = 1)
	{
		T result;
#ifdef CUDA_ARCH
		if (a != -1)
		{
			result = b * (pow(x2, a + 1) * ((a + 1) * log(x2) - 1) - pow(x1, a + 1) * ((a + 1) * log(x1) - 1)) / ((a + 1) * (a + 1));
		}
		else
		{
			result = b * (log(x2) * log(x2) - log(x1) * log(x1)) / 2;
		}
#else
		if (a != -1)
		{
			result = b * (std::pow(x2, a + 1) * ((a + 1) * std::log(x2) - 1) - std::pow(x1, a + 1) * ((a + 1) * std::log(x1) - 1)) / ((a + 1) * (a + 1));
		}
		else
		{
			result = b * (std::log(x2) * std::log(x2) - std::log(x1) * std::log(x1)) / 2;
		}
#endif
		return result;
	}

	/******************************************************************************
	calculate the value of x such that the integral of b * x^a from x1 to x = p
	assumes 0 < x1

	\param p -- value of the integral
	\param x1 -- lower bound of the integral
	\param a -- exponent of x
	\param b -- coefficient of x^a

	\return (p * (a + 1) / b + x1 ^ (a + 1)) ^ (1 / (a + 1)) if a =/= -1
			x1 * e ^ (p / b) if a == -1
	******************************************************************************/
	__host__ __device__ T invert_power_integral(T p, T x1, T a, T b)
	{
		T result;
#ifdef CUDA_ARCH
		if (a != -1)
		{
			result = pow(p * (a + 1) / b + pow(x1, a + 1), 1 / (a + 1));
		}
		else
		{
			result = x1 * exp(p / b);
		}
#else
		if (a != -1)
		{
			result = std::pow(p * (a + 1) / b + std::pow(x1, a + 1), 1 / (a + 1));
		}
		else
		{
			result = x1 * std::exp(p / b);
		}
#endif
		return result;
	}
	

public:

	/******************************************************************************
	calculate mass drawn from a mass function given a probability.

	\param p -- number drawn uniformly in [0,1]
	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param slope -- slope for the distribution
	******************************************************************************/
	__host__ __device__ virtual T mass(T p, T m_lower, T m_upper, T slope, ...)
    {
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		integrate over entire range of masses to get normalization factor
		******************************************************************************/
		T b = 1 / power_integral(m_lower, m_upper, slope);

		/******************************************************************************
		determine the mass
		******************************************************************************/
		T m = invert_power_integral(p, m_lower, slope, b);

		return m;
    }

	/******************************************************************************
	calculate <mass> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param slope -- slope for the distribution
	******************************************************************************/
	__host__ __device__ virtual T mean_mass(T m_lower, T m_upper, T slope, ...)
    {
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		integrate over entire range of masses to get normalization factor
		******************************************************************************/
		T b = 1 / power_integral(m_lower, m_upper, slope);

		/******************************************************************************
		determine <m>
		******************************************************************************/
		T m = power_integral(m_lower, m_upper, slope + 1, b);

		return m;
    }

	/******************************************************************************
	calculate <mass^2> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param slope -- slope for the distribution
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2(T m_lower, T m_upper, T slope, ...)
    {
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		integrate over entire range of masses to get normalization factor
		******************************************************************************/
		T b = 1 / power_integral(m_lower, m_upper, slope);

		/******************************************************************************
		determine <m^2>
		******************************************************************************/
		T m = power_integral(m_lower, m_upper, slope + 2, b);

		return m;
    }

	/******************************************************************************
	calculate <mass^2 ln(mass)> for a mass function

	\param m_lower -- lower mass cutoff for the distribution in arbitrary units
	\param m__upper -- upper mass cutoff for the distribution in arbitrary units
	\param slope -- slope for the distribution
	******************************************************************************/
	__host__ __device__ virtual T mean_mass2_ln_mass(T m_lower, T m_upper, T slope, ...)
    {
		if (m_lower == m_upper)
		{
			return m_lower;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		integrate over entire range of masses to get normalization factor
		******************************************************************************/
		T b = 1 / power_integral(m_lower, m_upper, slope);

		/******************************************************************************
		determine <m^2 * ln(m)>
		******************************************************************************/
		T m = power_log_integral(m_lower, m_upper, slope + 2, b);

		return m;

    }

};

}

