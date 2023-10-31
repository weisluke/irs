#pragma once

#include <cmath>
#include <map>
#include <string>


namespace massfunctions
{

	/******************************************************************************
	enum to hold type of mass function

	equal -- all objects of an equal mass
	uniform -- objects distributed uniformly in some range
	salpeter -- objects follow a Salpeter mass distribution in some range
	kroupa -- objects follow a Kroupa mass distribution in some range
	******************************************************************************/
	enum massfunction
	{
		equal,
		uniform,
		salpeter,
		kroupa,
	};

	const std::map<std::string, massfunction> MASS_FUNCTIONS
	{
		{"equal", equal},
		{"uniform", uniform},
		{"salpeter", salpeter},
		{"kroupa", kroupa}
	};

}


/******************************************************************************
template class for handling the mass functions of point mass lenses
******************************************************************************/
template <typename T>
class MassFunction
{
	/******************************************************************************
	calculate the integral of b * x^a from x1 to x2
	assumes 0 < x1 <= x2

	\param x1 -- lower bound of the integral
	\param x2 -- upper bound of the integral
	\param a -- exponent of x
	\param b -- coefficient of x

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
	calculate the value of x such that the integral of b * x^a from x1 to x = p
	assumes 0 < x1

	\param p -- value of the integral
	\param x1 -- lower bound of the integral
	\param a -- exponent of x
	\param b -- coefficient of x

	\return (p * (a + 1) / b + x1 ^ (a + 1)) ^ (1 / (a + 1)) if a =/= -1
			x1 * e ^ (p / b) if a == -1
	******************************************************************************/
	__host__ __device__ T invert_power_integral(T p, T x1, T a, T b)
	{
		T result;
#ifdef CUDA_ARCH
		if (a != -1)
		{
			result = pow(p * (1 + a) / b + pow(x1, 1 + a), 1 / (1 + a));
		}
		else
		{
			result = x1 * exp(p / b);
		}
#else
		if (a != -1)
		{
			result = std::pow(p * (1 + a) / b + std::pow(x1, 1 + a), 1 / (1 + a));
		}
		else
		{
			result = x1 * std::exp(p / b);
		}
#endif
		return result;
	}

public:
	massfunctions::massfunction mass_function;

	/******************************************************************************
	default constructor initializes the mass function to equal
	******************************************************************************/
	__host__ __device__ MassFunction(massfunctions::massfunction mf = massfunctions::equal)
	{
		mass_function = mf;
	}

	/******************************************************************************
	copy constructors
	******************************************************************************/
	template <typename U> __host__ __device__ MassFunction(const MassFunction<U>& mf)
	{
		mass_function = mf.mass_function;
	}

	/******************************************************************************
	assignment operators
	******************************************************************************/
	template <typename U> __host__ __device__ MassFunction& operator=(const MassFunction<U>& mf)
	{
		mass_function = mf.mass_function;
		return *this;
	}

	/******************************************************************************
	EQUAL MASS FUNCTION
	******************************************************************************/

	/******************************************************************************
	calculate mass for an equal mass function given a probability.

	\param p -- number drawn uniformly in [0,1]

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T equal_mass(T p)
	{
		return 1;
	}

	/******************************************************************************
	calculate <mass> for an equal mass function

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T mean_equal_mass()
	{
		return 1;
	}

	/******************************************************************************
	calculate <mass^2> for an equal mass function

	\return 1, as mass can be arbitrarily scaled
	******************************************************************************/
	__host__ __device__ T mean_equal_mass2()
	{
		return 1;
	}

	/******************************************************************************
	UNIFORM MASS FUNCTION
	******************************************************************************/

	/******************************************************************************
	calculate mass for a uniform mass function given a probability

	\param p -- number drawn uniformly in [0,1]
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T uniform_mass(T p, T mL = 0.01, T mH = 50)
	{
		return p * (mH - mL) + mL;
	}

	/******************************************************************************
	calculate <mass> for a uniform mass function

	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_uniform_mass(T mL = 0.01, T mH = 50)
	{
		return (mH + mL) / 2;
	}

	/******************************************************************************
	calculate <mass^2> for a uniform mass function

	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_uniform_mass2(T mL = 0.01, T mH = 50)
	{
		return (mH * mH + mH * mL + mL * mL) / 3;
	}

	/******************************************************************************
	SALPETER MASS FUNCTION
	******************************************************************************/

	/******************************************************************************
	calculate mass for a Salpeter mass function given a probability

	\param p -- number drawn uniformly in [0,1]
	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a -- slope for the distribution
	******************************************************************************/
	__host__ __device__ T salpeter_mass(T p, T msolar = 1, T mL = 0.01, T mH = 50, T a = -2.35)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probability, integrating over entire range of masses
		this is missing a factor of b
		******************************************************************************/
		T p_L_H = power_integral(m_lower, m_upper, a);

		/******************************************************************************
		determine b based on normalization condition
		******************************************************************************/
		T b = 1 / p_L_H;

		/******************************************************************************
		determine the mass
		******************************************************************************/
		T m = invert_power_integral(p, m_lower, a, b);

		return m * msolar;
	}

	/******************************************************************************
	calculate <mass> for a Salpeter mass function

	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a -- slope for the distribution
	******************************************************************************/
	__host__ __device__ T mean_salpeter_mass(T msolar = 1, T mL = 0.01, T mH = 50, T a = -2.35)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probability, integrating over entire range of masses
		this is missing a factor of b
		******************************************************************************/
		T p_L_H = power_integral(m_lower, m_upper, a);

		/******************************************************************************
		determine b based on normalization condition
		******************************************************************************/
		T b = 1 / p_L_H;

		/******************************************************************************
		determine <m>
		******************************************************************************/
		T m = power_integral(m_lower, m_upper, a + 1, b);

		return m * msolar;
	}

	/******************************************************************************
	calculate <mass^2> for a Salpeter mass function

	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a -- slope for the distribution
	******************************************************************************/
	__host__ __device__ T mean_salpeter_mass2(T msolar = 1, T mL = 0.01, T mH = 50, T a = -2.35)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probability, integrating over entire range of masses
		this is missing a factor of b
		******************************************************************************/
		T p_L_H = power_integral(m_lower, m_upper, a);

		/******************************************************************************
		determine b based on normalization condition
		******************************************************************************/
		T b = 1 / p_L_H;

		/******************************************************************************
		determine <m^2>
		******************************************************************************/
		T m = power_integral(m_lower, m_upper, a + 2, b);

		return m * msolar * msolar;
	}

	/******************************************************************************
	KROUPA MASS FUNCTION
	******************************************************************************/

	/******************************************************************************
	calculate mass for a Kroupa mass function given a probability

	\param p -- number drawn uniformly in [0,1]
	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a1 -- first slope for the distribution
	\param m1 -- cutoff for first slope in solar mass units
	\param a2 -- second slope for the distribution
	\param m2 -- cutoff for second slope in solar mass units
	\param a3 -- third slope for the distribution
	******************************************************************************/
	__host__ __device__ T kroupa_mass(T p, T msolar = 1, T mL = 0.01, T mH = 50, T a1 = -0.3, T m1 = 0.08, T a2 = -1.3, T m2 = 0.5, T a3 = -2.3)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		for mL <= m <= m1, p(m) = b1 * m^a1
		for m1 <= m <= m2, p(m) = b2 * m^a2
		for m2 <= m <= mH, p(m) = b2 * m^a3
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probabilities, integrating over entire range of masses
		these are missing factors of b1
		******************************************************************************/
		T p_0_1 = 0;
		T p_1_2 = 0;
		T p_2_3 = 0;

		if (m_upper < m1)
		{
			p_0_1 = power_integral(m_lower, m_upper, a1);
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m_upper, a2) * pow(m1, a1 - a2);
			}
			else
			{
				p_1_2 = power_integral(m_lower, m_upper, a2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m2, a2) * pow(m1, a1 - a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m1, a1 - a2) * pow(m2, a2 - a3);
			}
			else if (m_lower < m2)
			{
				p_1_2 = power_integral(m_lower, m2, a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m2, a2 - a3);
			}
			else
			{
				p_2_3 = power_integral(m_lower, m_upper, a3);
			}

		}

		/******************************************************************************
		determine b factors based on normalization condition and finish scaling
		probabilities
		******************************************************************************/
		T b1 = 0;
		T b2 = 0;
		T b3 = 0;

		if (m_upper < m1)
		{
			b1 = 1 / p_0_1;
			p_0_1 *= b1;
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2);
				p_0_1 *= b1;
				p_1_2 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
#endif
			}
			else
			{
				b2 = 1 / p_1_2;
				p_1_2 *= b2;
			}
		}
		else
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2 + p_2_3);
				p_0_1 *= b1;
				p_1_2 *= b1;
				p_2_3 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
				b3 = b2 * pow(m2, a2 - a3);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else if (m_lower < m2)
			{
				b2 = 1 / (p_1_2 + p_2_3);
				p_1_2 *= b2;
				p_2_3 *= b2;
#ifdef CUDA_ARCH
				b3 = b2 * pow(m2, a2 - a3);
#else
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else
			{
				b3 = 1 / p_2_3;
				p_2_3 *= b3;
			}

		}

		/******************************************************************************
		determine the mass
		******************************************************************************/
		T m;

		if (p < p_0_1)
		{
			m = invert_power_integral(p, m_lower, a1, b1);
		}
		else if (p < (p_0_1 + p_1_2))
		{
			if (m_lower < m1)
			{
				m = invert_power_integral(p - p_0_1, m1, a2, b2);
			}
			else
			{
				m = invert_power_integral(p, m_lower, a2, b2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				m = invert_power_integral(p - p_0_1 - p_1_2, m2, a3, b3);
			}
			else if (m_lower < m2)
			{
				m = invert_power_integral(p - p_1_2, m2, a3, b3);
			}
			else
			{
				m = invert_power_integral(p, m_lower, a3, b3);
			}
		}

		return m * msolar;
	}

	/******************************************************************************
	calculate <mass> for a Kroupa mass function

	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a1 -- first slope for the distribution
	\param m1 -- cutoff for first slope in solar mass units
	\param a2 -- second slope for the distribution
	\param m2 -- cutoff for second slope in solar mass units
	\param a3 -- third slope for the distribution
	******************************************************************************/
	__host__ __device__ T mean_kroupa_mass(T msolar = 1, T mL = 0.01, T mH = 50, T a1 = -0.3, T m1 = 0.08, T a2 = -1.3, T m2 = 0.5, T a3 = -2.3)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		for mL <= m <= m1, p(m) = b1 * m^a1
		for m1 <= m <= m2, p(m) = b2 * m^a2
		for m2 <= m <= mH, p(m) = b2 * m^a3
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probabilities, integrating over entire range of masses
		these are missing factors of b1
		******************************************************************************/
		T p_0_1 = 0;
		T p_1_2 = 0;
		T p_2_3 = 0;

		if (m_upper < m1)
		{
			p_0_1 = power_integral(m_lower, m_upper, a1);
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m_upper, a2) * pow(m1, a1 - a2);
			}
			else
			{
				p_1_2 = power_integral(m_lower, m_upper, a2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m2, a2) * pow(m1, a1 - a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m1, a1 - a2) * pow(m2, a2 - a3);
			}
			else if (m_lower < m2)
			{
				p_1_2 = power_integral(m_lower, m2, a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m2, a2 - a3);
			}
			else
			{
				p_2_3 = power_integral(m_lower, m_upper, a3);
			}

		}

		/******************************************************************************
		determine b factors based on normalization condition and finish scaling
		probabilities
		******************************************************************************/
		T b1 = 0;
		T b2 = 0;
		T b3 = 0;

		if (m_upper < m1)
		{
			b1 = 1 / p_0_1;
			p_0_1 *= b1;
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2);
				p_0_1 *= b1;
				p_1_2 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
#endif
			}
			else
			{
				b2 = 1 / p_1_2;
				p_1_2 *= b2;
			}
		}
		else
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2 + p_2_3);
				p_0_1 *= b1;
				p_1_2 *= b1;
				p_2_3 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
				b3 = b2 * pow(m2, a2 - a3);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else if (m_lower < m2)
			{
				b2 = 1 / (p_1_2 + p_2_3);
				p_1_2 *= b2;
				p_2_3 *= b2;
#ifdef CUDA_ARCH
				b3 = b2 * pow(m2, a2 - a3);
#else
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else
			{
				b3 = 1 / p_2_3;
				p_2_3 *= b3;
			}

		}

		/******************************************************************************
		determine <m>
		******************************************************************************/
		T m;

		if (m_upper < m1)
		{
			m = power_integral(m_lower, m_upper, a1 + 1, b1);
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				m = power_integral(m_lower, m1, a1 + 1, b1)
					+ power_integral(m1, m_upper, a2 + 1, b2);
			}
			else
			{
				m = power_integral(m_lower, m_upper, a2 + 1, b2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				m = power_integral(m_lower, m1, a1 + 1, b1)
					+ power_integral(m1, m2, a2 + 1, b2)
					+ power_integral(m2, m_upper, a3 + 1, b3);
			}
			else if (m_lower < m2)
			{
				m = power_integral(m_lower, m2, a2 + 1, b2)
					+ power_integral(m2, m_upper, a3 + 1, b3);
			}
			else
			{
				m = power_integral(m_lower, m_upper, a3 + 1, b3);
			}
		}

		return m * msolar;
	}

	/******************************************************************************
	calculate <mass^2> for a Kroupa mass function

	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	\param a1 -- first slope for the distribution
	\param m1 -- cutoff for first slope in solar mass units
	\param a2 -- second slope for the distribution
	\param m2 -- cutoff for second slope in solar mass units
	\param a3 -- third slope for the distribution
	******************************************************************************/
	__host__ __device__ T mean_kroupa_mass2(T msolar = 1, T mL = 0.01, T mH = 50, T a1 = -0.3, T m1 = 0.08, T a2 = -1.3, T m2 = 0.5, T a3 = -2.3)
	{
		if (mL == mH)
		{
			return mL;
		}

		/******************************************************************************
		p(m) = b * m^a
		for mL <= m <= m1, p(m) = b1 * m^a1
		for m1 <= m <= m2, p(m) = b2 * m^a2
		for m2 <= m <= mH, p(m) = b2 * m^a3
		******************************************************************************/

		/******************************************************************************
		convert upper and lower bounds into solar mass units
		******************************************************************************/
		T m_lower = mL / msolar;
		T m_upper = mH / msolar;

		/******************************************************************************
		determine probabilities, integrating over entire range of masses
		these are missing factors of b1
		******************************************************************************/
		T p_0_1 = 0;
		T p_1_2 = 0;
		T p_2_3 = 0;

		if (m_upper < m1)
		{
			p_0_1 = power_integral(m_lower, m_upper, a1);
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m_upper, a2) * pow(m1, a1 - a2);
			}
			else
			{
				p_1_2 = power_integral(m_lower, m_upper, a2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				p_0_1 = power_integral(m_lower, m1, a1);
				p_1_2 = power_integral(m1, m2, a2) * pow(m1, a1 - a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m1, a1 - a2) * pow(m2, a2 - a3);
			}
			else if (m_lower < m2)
			{
				p_1_2 = power_integral(m_lower, m2, a2);
				p_2_3 = power_integral(m2, m_upper, a3) * pow(m2, a2 - a3);
			}
			else
			{
				p_2_3 = power_integral(m_lower, m_upper, a3);
			}

		}

		/******************************************************************************
		determine b factors based on normalization condition and finish scaling
		probabilities
		******************************************************************************/
		T b1 = 0;
		T b2 = 0;
		T b3 = 0;

		if (m_upper < m1)
		{
			b1 = 1 / p_0_1;
			p_0_1 *= b1;
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2);
				p_0_1 *= b1;
				p_1_2 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
#endif
			}
			else
			{
				b2 = 1 / p_1_2;
				p_1_2 *= b2;
			}
		}
		else
		{
			if (m_lower < m1)
			{
				b1 = 1 / (p_0_1 + p_1_2 + p_2_3);
				p_0_1 *= b1;
				p_1_2 *= b1;
				p_2_3 *= b1;
#ifdef CUDA_ARCH
				b2 = b1 * pow(m1, a1 - a2);
				b3 = b2 * pow(m2, a2 - a3);
#else
				b2 = b1 * std::pow(m1, a1 - a2);
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else if (m_lower < m2)
			{
				b2 = 1 / (p_1_2 + p_2_3);
				p_1_2 *= b2;
				p_2_3 *= b2;
#ifdef CUDA_ARCH
				b3 = b2 * pow(m2, a2 - a3);
#else
				b3 = b2 * std::pow(m2, a2 - a3);
#endif
			}
			else
			{
				b3 = 1 / p_2_3;
				p_2_3 *= b3;
			}

		}

		/******************************************************************************
		determine <m^2>
		******************************************************************************/
		T m;

		if (m_upper < m1)
		{
			m = power_integral(m_lower, m_upper, a1 + 2, b1);
		}
		else if (m_upper < m2)
		{
			if (m_lower < m1)
			{
				m = power_integral(m_lower, m1, a1 + 2, b1)
					+ power_integral(m1, m_upper, a2 + 2, b2);
			}
			else
			{
				m = power_integral(m_lower, m_upper, a2 + 2, b2);
			}
		}
		else
		{
			if (m_lower < m1)
			{
				m = power_integral(m_lower, m1, a1 + 2, b1)
					+ power_integral(m1, m2, a2 + 2, b2)
					+ power_integral(m2, m_upper, a3 + 2, b3);
			}
			else if (m_lower < m2)
			{
				m = power_integral(m_lower, m2, a2 + 2, b2)
					+ power_integral(m2, m_upper, a3 + 2, b3);
			}
			else
			{
				m = power_integral(m_lower, m_upper, a3 + 2, b3);
			}
		}

		return m * msolar * msolar;
	}

	/******************************************************************************
	MASS FUNCTION
	******************************************************************************/

	/******************************************************************************
	calculate mass given a probability

	\param p -- number drawn uniformly in [0,1]
	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mass(T p, T msolar = 1, T mL = 0.01, T mH = 50)
	{
		switch (mass_function)
		{
		case massfunctions::equal:
			return equal_mass(p);
		case massfunctions::uniform:
			return uniform_mass(p, mL, mH);
		case massfunctions::salpeter:
			return salpeter_mass(p, msolar, mL, mH);
		case massfunctions::kroupa:
			return kroupa_mass(p, msolar, mL, mH);
		default:
			return equal_mass(p);
		}
	}

	/******************************************************************************
	calculate <mass> given a probability

	\param p -- number drawn uniformly in [0,1]
	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass(T msolar = 1, T mL = 0.01, T mH = 50)
	{
		switch (mass_function)
		{
		case massfunctions::equal:
			return mean_equal_mass();
		case massfunctions::uniform:
			return mean_uniform_mass(mL, mH);
		case massfunctions::salpeter:
			return mean_salpeter_mass(msolar, mL, mH);
		case massfunctions::kroupa:
			return mean_kroupa_mass(msolar, mL, mH);
		default:
			return mean_equal_mass();
		}
	}

	/******************************************************************************
	calculate <mass^2> given a probability

	\param msolar -- solar mass in arbitrary units
	\param mL -- lower mass cutoff for the distribution in arbitrary units
	\param mH -- upper mass cutoff for the distribution in arbitrary units
	******************************************************************************/
	__host__ __device__ T mean_mass2(T msolar = 1, T mL = 0.01, T mH = 50)
	{
		switch (mass_function)
		{
		case massfunctions::equal:
			return mean_equal_mass2();
		case massfunctions::uniform:
			return mean_uniform_mass2(mL, mH);
		case massfunctions::salpeter:
			return mean_salpeter_mass2(msolar, mL, mH);
		case massfunctions::kroupa:
			return mean_kroupa_mass2(msolar, mL, mH);
		default:
			return mean_equal_mass2();
		}
	}

};

