#pragma once

#include "power_law.cuh"


namespace massfunctions {

	/******************************************************************************
	template class for handling point mass lenses following a kroupa distribution
	******************************************************************************/
	template <typename T>
	class Kroupa : public PowerLaw<T>
	{

	public:

		/******************************************************************************
		a kroupa distribution is composed of 3 broken power law distributions
		******************************************************************************/
		T slopes[3];
		T breaks[2]; //in units of solar mass

		__host__ __device__ Kroupa()
		{
			slopes[0] = static_cast<T>(-0.3);
			slopes[1] = static_cast<T>(-1.3);
			slopes[2] = static_cast<T>(-2.3);

			breaks[0] = static_cast<T>(0.08);
			breaks[1] = static_cast<T>(0.5);
		}

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

			/******************************************************************************
			breakpoints converted from solar mass units to arbitrary units
			******************************************************************************/
			T m0 = breaks[0] * m_solar;
			T m1 = breaks[1] * m_solar;

			/******************************************************************************
			variable for probabilities, integrating over entire range of masses
			variable for normalization constants for the distributions
			******************************************************************************/
			T probs[3] = {0, 0, 0};
			T b[3] = {0, 0, 0};

			if (m_upper < m0)
			{
				probs[0] = power_integral(m_lower, m_upper, slopes[0]);
				b[0] = 1 / probs[0];
				probs[0] *= b[0];
			}
			else if (m_upper < m1)
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m_upper, slopes[1]) * pow(m0, slopes[0] - slopes[1]);

					b[0] = 1 / (probs[0] + probs[1]);
					probs[0] *= b[0];
					probs[1] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);
				}
				else
				{
					probs[1] = power_integral(m_lower, m_upper, slopes[1]);
					b[1] = 1 / probs[1];
					probs[1] *= b[1];
				}
			}
			else
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m1, slopes[1]) * pow(m0, slopes[0] - slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m0, slopes[0] - slopes[1]) * pow(m1, slopes[1] - slopes[2]);

					b[0] = 1 / (probs[0] + probs[1] + probs[2]);
					probs[0] *= b[0];
					probs[1] *= b[0];
					probs[2] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);
					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);
				}
				else if (m_lower < m1)
				{
					probs[1] = power_integral(m_lower, m1, slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m1, slopes[1] - slopes[2]);

					b[1] = 1 / (probs[1] + probs[2]);
					probs[1] *= b[1];
					probs[2] *= b[1];

					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);
				}
				else
				{
					probs[2] = power_integral(m_lower, m_upper, slopes[2]);
					b[2] = 1 / probs[2];
					probs[2] *= b[2];
				}

			}

			/******************************************************************************
			determine the mass
			******************************************************************************/
			T m;

			if (p < probs[0])
			{
				m = invert_power_integral(p, m_lower, slopes[0], b[0]);
			}
			else if (p < (probs[0] + probs[1]))
			{
				if (m_lower < m0)
				{
					m = invert_power_integral(p - probs[0], m0, slopes[1], b[1]);
				}
				else
				{
					m = invert_power_integral(p, m_lower, slopes[1], b[1]);
				}
			}
			else
			{
				if (m_lower < m0)
				{
					m = invert_power_integral(p - probs[0] - probs[1], m1, slopes[2], b[2]);
				}
				else if (m_lower < m1)
				{
					m = invert_power_integral(p - probs[1], m1, slopes[2], b[2]);
				}
				else
				{
					m = invert_power_integral(p, m_lower, slopes[2], b[2]);
				}
			}

			return m;
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

			/******************************************************************************
			breakpoints converted from solar mass units to arbitrary units
			******************************************************************************/
			T m0 = breaks[0] * m_solar;
			T m1 = breaks[1] * m_solar;

			/******************************************************************************
			variable for probabilities, integrating over entire range of masses
			variable for normalization constants for the distributions
			******************************************************************************/
			T probs[3] = { 0, 0, 0 };
			T b[3] = { 0, 0, 0 };

			/******************************************************************************
			variable for <m>
			******************************************************************************/
			T m;

			if (m_upper < m0)
			{
				probs[0] = power_integral(m_lower, m_upper, slopes[0]);
				b[0] = 1 / probs[0];
				probs[0] *= b[0];

				m = power_integral(m_lower, m_upper, slopes[0] + 1, b[0]);
			}
			else if (m_upper < m1)
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m_upper, slopes[1]) * pow(m0, slopes[0] - slopes[1]);

					b[0] = 1 / (probs[0] + probs[1]);
					probs[0] *= b[0];
					probs[1] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);

					m = power_integral(m_lower, m0, slopes[0] + 1, b[0]) 
						+ power_integral(m0, m_upper, slopes[1] + 1, b[1]);
				}
				else
				{
					probs[1] = power_integral(m_lower, m_upper, slopes[1]);
					b[1] = 1 / probs[1];
					probs[1] *= b[1];

					m = power_integral(m_lower, m_upper, slopes[1] + 1, b[1]);
				}
			}
			else
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m1, slopes[1]) * pow(m0, slopes[0] - slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m0, slopes[0] - slopes[1]) * pow(m1, slopes[1] - slopes[2]);

					b[0] = 1 / (probs[0] + probs[1] + probs[2]);
					probs[0] *= b[0];
					probs[1] *= b[0];
					probs[2] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);
					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_integral(m_lower, m0, slopes[0] + 1, b[0]) 
						+ power_integral(m0, m1, slopes[1] + 1, b[1])
						+ power_integral(m1, m_upper, slopes[2] + 1, b[2]);
				}
				else if (m_lower < m1)
				{
					probs[1] = power_integral(m_lower, m1, slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m1, slopes[1] - slopes[2]);

					b[1] = 1 / (probs[1] + probs[2]);
					probs[1] *= b[1];
					probs[2] *= b[1];

					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_integral(m_lower, m1, slopes[1] + 1, b[1])
						+ power_integral(m1, m_upper, slopes[2] + 1, b[2]);
				}
				else
				{
					probs[2] = power_integral(m_lower, m_upper, slopes[2]);
					b[2] = 1 / probs[2];
					probs[2] *= b[2];

					m = power_integral(m_lower, m_upper, slopes[2] + 1, b[2]);
				}

			}

			return m;
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
				return m_lower;
			}

			/******************************************************************************
			breakpoints converted from solar mass units to arbitrary units
			******************************************************************************/
			T m0 = breaks[0] * m_solar;
			T m1 = breaks[1] * m_solar;

			/******************************************************************************
			variable for probabilities, integrating over entire range of masses
			variable for normalization constants for the distributions
			******************************************************************************/
			T probs[3] = { 0, 0, 0 };
			T b[3] = { 0, 0, 0 };

			/******************************************************************************
			variable for <m^2>
			******************************************************************************/
			T m;

			if (m_upper < m0)
			{
				probs[0] = power_integral(m_lower, m_upper, slopes[0]);
				b[0] = 1 / probs[0];
				probs[0] *= b[0];

				m = power_integral(m_lower, m_upper, slopes[0] + 2, b[0]);
			}
			else if (m_upper < m1)
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m_upper, slopes[1]) * pow(m0, slopes[0] - slopes[1]);

					b[0] = 1 / (probs[0] + probs[1]);
					probs[0] *= b[0];
					probs[1] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);

					m = power_integral(m_lower, m0, slopes[0] + 2, b[0])
						+ power_integral(m0, m_upper, slopes[1] + 2, b[1]);
				}
				else
				{
					probs[1] = power_integral(m_lower, m_upper, slopes[1]);
					b[1] = 1 / probs[1];
					probs[1] *= b[1];

					m = power_integral(m_lower, m_upper, slopes[1] + 2, b[1]);
				}
			}
			else
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m1, slopes[1]) * pow(m0, slopes[0] - slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m0, slopes[0] - slopes[1]) * pow(m1, slopes[1] - slopes[2]);

					b[0] = 1 / (probs[0] + probs[1] + probs[2]);
					probs[0] *= b[0];
					probs[1] *= b[0];
					probs[2] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);
					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_integral(m_lower, m0, slopes[0] + 2, b[0])
						+ power_integral(m0, m1, slopes[1] + 2, b[1])
						+ power_integral(m1, m_upper, slopes[2] + 2, b[2]);
				}
				else if (m_lower < m1)
				{
					probs[1] = power_integral(m_lower, m1, slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m1, slopes[1] - slopes[2]);

					b[1] = 1 / (probs[1] + probs[2]);
					probs[1] *= b[1];
					probs[2] *= b[1];

					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_integral(m_lower, m1, slopes[1] + 2, b[1])
						+ power_integral(m1, m_upper, slopes[2] + 2, b[2]);
				}
				else
				{
					probs[2] = power_integral(m_lower, m_upper, slopes[2]);
					b[2] = 1 / probs[2];
					probs[2] *= b[2];

					m = power_integral(m_lower, m_upper, slopes[2] + 2, b[2]);
				}

			}

			return m;
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
				return m_lower;
			}

			/******************************************************************************
			breakpoints converted from solar mass units to arbitrary units
			******************************************************************************/
			T m0 = breaks[0] * m_solar;
			T m1 = breaks[1] * m_solar;

			/******************************************************************************
			variable for probabilities, integrating over entire range of masses
			variable for normalization constants for the distributions
			******************************************************************************/
			T probs[3] = { 0, 0, 0 };
			T b[3] = { 0, 0, 0 };

			/******************************************************************************
			variable for <m^2 * ln(m)>
			******************************************************************************/
			T m;

			if (m_upper < m0)
			{
				probs[0] = power_integral(m_lower, m_upper, slopes[0]);
				b[0] = 1 / probs[0];
				probs[0] *= b[0];

				m = power_log_integral(m_lower, m_upper, slopes[0] + 2, b[0]);
			}
			else if (m_upper < m1)
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m_upper, slopes[1]) * pow(m0, slopes[0] - slopes[1]);

					b[0] = 1 / (probs[0] + probs[1]);
					probs[0] *= b[0];
					probs[1] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);

					m = power_log_integral(m_lower, m0, slopes[0] + 2, b[0])
						+ power_log_integral(m0, m_upper, slopes[1] + 2, b[1]);
				}
				else
				{
					probs[1] = power_integral(m_lower, m_upper, slopes[1]);
					b[1] = 1 / probs[1];
					probs[1] *= b[1];

					m = power_log_integral(m_lower, m_upper, slopes[1] + 2, b[1]);
				}
			}
			else
			{
				if (m_lower < m0)
				{
					probs[0] = power_integral(m_lower, m0, slopes[0]);
					probs[1] = power_integral(m0, m1, slopes[1]) * pow(m0, slopes[0] - slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m0, slopes[0] - slopes[1]) * pow(m1, slopes[1] - slopes[2]);

					b[0] = 1 / (probs[0] + probs[1] + probs[2]);
					probs[0] *= b[0];
					probs[1] *= b[0];
					probs[2] *= b[0];

					b[1] = b[0] * pow(m0, slopes[0] - slopes[1]);
					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_log_integral(m_lower, m0, slopes[0] + 2, b[0])
						+ power_log_integral(m0, m1, slopes[1] + 2, b[1])
						+ power_log_integral(m1, m_upper, slopes[2] + 2, b[2]);
				}
				else if (m_lower < m1)
				{
					probs[1] = power_integral(m_lower, m1, slopes[1]);
					probs[2] = power_integral(m1, m_upper, slopes[2]) * pow(m1, slopes[1] - slopes[2]);

					b[1] = 1 / (probs[1] + probs[2]);
					probs[1] *= b[1];
					probs[2] *= b[1];

					b[2] = b[1] * pow(m1, slopes[1] - slopes[2]);

					m = power_log_integral(m_lower, m1, slopes[1] + 2, b[1])
						+ power_log_integral(m1, m_upper, slopes[2] + 2, b[2]);
				}
				else
				{
					probs[2] = power_integral(m_lower, m_upper, slopes[2]);
					b[2] = 1 / probs[2];
					probs[2] *= b[2];

					m = power_log_integral(m_lower, m_upper, slopes[2] + 2, b[2]);
				}

			}

			return m;

		}

	};

}

