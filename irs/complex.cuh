#pragma once

#include <cmath>


/****************************************************************************
template class for handling complex number arithmetic with operator overloads
****************************************************************************/
template <typename T>
class Complex
{
public:
	T re;
	T im;

	/*default constructor initializes complex number to zero*/
	__host__ __device__ Complex(T real = 0, T imag = 0)
	{
		re = real;
		im = imag;
	}

	__host__ __device__ Complex conj()
	{
		return Complex(re, -im);
	}

	/*sqrt(re*re + im*im)*/
	__host__ __device__ T abs()
	{
		/*use device or host square root function*/
		#ifdef CUDA_ARCH
			return sqrt(re * re + im * im);
		#else
			return std::sqrt(re * re + im * im);
		#endif
	}

	__host__ __device__ Complex operator+()
	{
		return Complex(re, im);
	}
	__host__ __device__ Complex operator-()
	{
		return Complex(-re, -im);
	}

	/*addition*/
	template <typename U> __host__ __device__ Complex& operator+=(Complex<U> c1)
	{
		re += c1.re;
		im += c1.im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator+=(U num)
	{
		re += num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator+(Complex c1, Complex<U> c2)
	{
		return Complex(c1.re + c2.re, c1.im + c2.im);
	}
	template <typename U> __host__ __device__ friend Complex operator+(Complex c1, U num)
	{
		return Complex(c1.re + num, c1.im);
	}
	template <typename U> __host__ __device__ friend Complex operator+(T num, Complex<U> c1)
	{
		return Complex(num + c1.re, c1.im);
	}

	/*subtraction*/
	template <typename U> __host__ __device__ Complex& operator-=(Complex<U> c1)
	{
		re -= c1.re;
		im -= c1.im;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator-=(U num)
	{
		re -= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator-(Complex c1, Complex<U> c2)
	{
		return Complex(c1.re - c2.re, c1.im - c2.im);
	}
	template <typename U> __host__ __device__ friend Complex operator-(Complex c1, U num)
	{
		return Complex(c1.re - num, c1.im);
	}
	template <typename U> __host__ __device__ friend Complex operator-(T num, Complex<U> c1)
	{
		return Complex(num - c1.re, -c1.im);
	}

	/*multiplication*/
	template <typename U> __host__ __device__ Complex& operator*=(Complex<U> c1)
	{
		re = re * c1.re - im * c1.im;
		im = re * c1.im + im * c1.re;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator*=(U num)
	{
		re *= num;
		im *= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator*(Complex c1, Complex<U> c2)
	{
		return Complex(c1.re * c2.re - c1.im * c2.im, c1.re * c2.im + c1.im * c2.re);
	}
	template <typename U> __host__ __device__ friend Complex operator*(Complex c1, U num)
	{
		return Complex(c1.re * num, c1.im * num);
	}
	template <typename U> __host__ __device__ friend Complex operator*(T num, Complex<U> c1)
	{
		return Complex(num * c1.re, num * c1.im);
	}

	/*division*/
	template <typename U> __host__ __device__ Complex& operator/=(Complex<U> c1)
	{
		T norm2 = c1.re * c1.re + c1.im * c1.im;
		re = (re * c1.re + im * c1.im) / norm2;
		im = (im * c1.re - re * c1.im) / norm2;
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator/=(U num)
	{
		re /= num;
		im /= num;
		return *this;
	}
	template <typename U> __host__ __device__ friend Complex operator/(Complex c1, Complex<U> c2)
	{
		T norm2 = c2.re * c2.re + c2.im * c2.im;
		return Complex((c1.re * c2.re + c1.im * c2.im) / norm2, (c1.im * c2.re - c1.re * c2.im) / norm2);
	}
	template <typename U> __host__ __device__ friend Complex operator/(Complex c1, U num)
	{
		return Complex(c1.re / num, c1.im / num);
	}
	template <typename U> __host__ __device__ friend Complex operator/(T num, Complex<U> c1)
	{
		T norm2 = c1.re * c1.re + c1.im * c1.im;
		return Complex(num * c1.re / norm2, -num * c1.im / norm2);
	}

};

