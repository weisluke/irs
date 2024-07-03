#pragma once

#include "complex.cuh"


/******************************************************************************
Heaviside Step Function

\param x -- number to evaluate

\return 1 if x > 0, 0 if x <= 0
******************************************************************************/
template <typename T>
__device__ T heaviside(T x)
{
	if (x >= 0)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
2-Dimensional Boxcar Function

\param z -- complex number to evalulate
\param corner -- corner of the rectangular region

\return 1 if z lies within the rectangle defined by corner, 0 if it is on the
		border or outside
******************************************************************************/
template <typename T>
__device__ T boxcar(Complex<T> z, Complex<T> corner)
{
	if (-corner.re <= z.re && z.re <= corner.re && -corner.im <= z.im && z.im <= corner.im)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/******************************************************************************
calculate the deflection angle due to smooth matter

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return alpha_smooth
******************************************************************************/
template <typename T>
__device__ Complex<T> alpha_smooth(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> a_smooth;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> s1;
			Complex<T> s2;
			Complex<T> s3;
			Complex<T> s4;

			for (int i = taylor_smooth; i >= 1; i--)
			{
				s1 *= i;
				s1 += 1;
				s1 /= i;

				s2 *= i;
				s2 += 1;
				s2 /= i;

				s3 *= i;
				s3 += 1;
				s3 /= i;

				s4 *= i;
				s4 += 1;
				s4 /= i;

				s1 *= (z.conj() / corner);
				s2 *= (z.conj() / corner.conj());
				s3 *= (z.conj() / -corner);
				s4 *= (z.conj() / -corner.conj());
			}

			a_smooth = ((corner - z.conj()) * (corner.log() - s1) - (corner.conj() - z.conj()) * (corner.conj().log() - s2)
				+ (-corner - z.conj()) * ((-corner).log() - s3) - (-corner.conj() - z.conj()) * ((-corner).conj().log() - s4));
			a_smooth *= Complex<T>(0, -kappastar / PI);
			a_smooth -= kappastar * 2 * (corner.re + z.re);
		}
		else
		{
			Complex<T> c1 = corner - z.conj();
			Complex<T> c2 = corner.conj() - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			a_smooth = (c1 * c1.log() - c2 * c2.log() + c3 * c3.log() - c4 * c4.log());
			a_smooth *= Complex<T>(0, -kappastar / PI);
			a_smooth -= kappastar * 2 * (corner.re + z.re) * boxcar(z, corner);
			a_smooth -= kappastar * 4 * corner.re * heaviside(corner.im + z.im) * heaviside(corner.im - z.im) * heaviside(z.re - corner.re);
		}
	}
	else
	{
		a_smooth = -kappastar * z;
	}

	return a_smooth;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to z

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the rectangular field of
				 point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not

\return d_alpha_smooth_d_z
******************************************************************************/
template <typename T>
__device__ T d_alpha_smooth_d_z(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx)
{
	T d_a_smooth_d_z = -kappastar;

	if (rectangular && !approx)
	{
		d_a_smooth_d_z *= boxcar(z, corner);
	}

	return d_a_smooth_d_z;
}

/******************************************************************************
calculate the derivative of the deflection angle due to smooth matter with
respect to zbar

\param z -- complex image plane position
\param kappastar -- convergence in point mass lenses
\param rectangular -- whether the star field is rectangular or not
\param corner -- complex number denoting the corner of the
				 rectangular field of point mass lenses
\param approx -- whether the smooth matter deflection is approximate or not
\param taylor_smooth -- degree of the taylor series for alpha_smooth if
                        approximate

\return d_alpha_smooth_d_zbar
******************************************************************************/
template <typename T>
__device__ Complex<T> d_alpha_smooth_d_zbar(Complex<T> z, T kappastar, int rectangular, Complex<T> corner, int approx, int taylor_smooth)
{
	T PI = static_cast<T>(3.1415926535898);
	Complex<T> d_a_smooth_d_zbar;

	if (rectangular)
	{
		if (approx)
		{
			Complex<T> r = z.conj() / corner; //geometric series ratio
			Complex<T> phase = Complex<T>(0, 2 * corner.arg());

			if (taylor_smooth % 2 == 0)
			{
				d_a_smooth_d_zbar = (1 - (phase * taylor_smooth).exp());
			}

			for (int i = (taylor_smooth % 2 == 0 ? taylor_smooth : taylor_smooth - 1); i >= 2; i -= 2)
			{
				d_a_smooth_d_zbar += (1 - (phase * i).exp()) / i;
				d_a_smooth_d_zbar *= (r * r);
			}
			d_a_smooth_d_zbar *= 2;

			d_a_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_a_smooth_d_zbar += kappastar - 4 * kappastar * corner.arg() / PI;
		}
		else
		{
			Complex<T> c1 = corner.conj() - z.conj();
			Complex<T> c2 = corner - z.conj();
			Complex<T> c3 = -corner - z.conj();
			Complex<T> c4 = -corner.conj() - z.conj();

			d_a_smooth_d_zbar = (c1.log() - c2.log() - c3.log() + c4.log());
			d_a_smooth_d_zbar *= Complex<T>(0, -kappastar / PI);
			d_a_smooth_d_zbar -= kappastar * boxcar(z, corner);
		}
	}

	return d_a_smooth_d_zbar;
}

