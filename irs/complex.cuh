#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept> //for std::logic_error


/******************************************************************************
template class for handling complex number arithmetic with operator overloads
******************************************************************************/
template <typename T>
class Complex
{
public:
	T re;
	T im;

	/******************************************************************************
	default constructor initializes the complex number to zero
	******************************************************************************/
	__host__ __device__ Complex()
	{
		re = 0;
		im = 0;
	}
	template <typename U> __host__ __device__ Complex(U real)
	{
		re = static_cast<T>(real);
		im = 0;
	}
	template <typename U, typename V> __host__ __device__ Complex(U real, V imag)
	{
		re = static_cast<T>(real);
		im = static_cast<T>(imag);
	}

	/******************************************************************************
	copy constructor
	******************************************************************************/
	template <typename U> __host__ __device__ Complex(const Complex<U>& c1)
	{
		re = static_cast<T>(c1.re);
		im = static_cast<T>(c1.im);
	}

	/******************************************************************************
	assignment operators
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator=(const Complex<U>& c1)
	{
		re = static_cast<T>(c1.re);
		im = static_cast<T>(c1.im);
		return *this;
	}
	template <typename U> __host__ __device__ Complex& operator=(const U& num)
	{
		re = static_cast<T>(num);
		im = static_cast<T>(0);
		return *this;
	}

	/******************************************************************************
	print a complex number to screen
	******************************************************************************/
	friend std::ostream& operator<< (std::ostream& out, const Complex& c1)
	{
		out << "(" << c1.re << ", " << c1.im << ")";
		return out;
	}

	/******************************************************************************
	complex conjugate of the complex number
	******************************************************************************/
	__host__ __device__ Complex conj()
	{
		return Complex(re, -im);
	}

	/******************************************************************************
	norm of the complex number = sqrt(re*re + im*im)
	******************************************************************************/
	__host__ __device__ T abs()
	{
		/*use device or host square root function*/
#ifdef CUDA_ARCH
		return sqrt(re * re + im * im);
#else
		return std::sqrt(re * re + im * im);
#endif
	}

	/******************************************************************************
	argument of the complex number in the range [-pi, pi]
	******************************************************************************/
	__host__ __device__ T arg()
	{
#ifdef CUDA_ARCH
		return atan2(im, re);
#else
		return std::atan2(im, re);
#endif
	}

	/******************************************************************************
	exponential of the complex number
	******************************************************************************/
	__host__ __device__ Complex exp()
	{
#ifdef CUDA_ARCH
		return Complex(exp(re) * cos(im), exp(re) * sin(im));
#else
		return Complex(std::exp(re) * std::cos(im), std::exp(re) * std::sin(im));
#endif
	}

	/******************************************************************************
	logarithm of the complex number
	******************************************************************************/
	__host__ __device__ Complex log()
	{
		T abs = this->abs();
		T arg = this->arg();
#ifdef CUDA_ARCH
		return Complex(log(abs), arg);
#else
		return Complex(std::log(abs), arg);
#endif
	}

	/******************************************************************************
	positive and negative operators
	******************************************************************************/
	__host__ __device__ Complex operator+()
	{
		return Complex(re, im);
	}
	__host__ __device__ Complex operator-()
	{
		return Complex(-re, -im);
	}

	/******************************************************************************
	addition
	******************************************************************************/
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

	/******************************************************************************
	subtraction
	******************************************************************************/
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

	/******************************************************************************
	multiplication
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator*=(Complex<U> c1)
	{
		T new_re = re * c1.re - im * c1.im;
		T new_im = re * c1.im + im * c1.re;
		re = new_re;
		im = new_im;
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

	/******************************************************************************
	division
	******************************************************************************/
	template <typename U> __host__ __device__ Complex& operator/=(Complex<U> c1)
	{
		T norm2 = c1.re * c1.re + c1.im * c1.im;
		T new_re = (re * c1.re + im * c1.im) / norm2;
		T new_im = (im * c1.re - re * c1.im) / norm2;
		re = new_re;
		im = new_im;
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

	/******************************************************************************
	exponentiation
	******************************************************************************/
	__host__ __device__ Complex pow(int num)
	{
		Complex res = Complex(1, 0);

		if (num > 0)
		{
			for (int i = 0; i < num; i++)
			{
				res *= *this;
			}
		}
		else if (num < 0)
		{
			for (int i = 0; i > num; i--)
			{
				res *= *this;
			}
			res = 1 / res;
		}

		return res;
	}
	template <typename U> __host__ __device__ Complex pow(Complex<U> num)
	{
		throw std::logic_error("Complex pow(Complex<U> num) not implemented.");
	}
	template <typename U> __host__ __device__ Complex pow(U num)
	{
		throw std::logic_error("Complex pow(U num) not implemented.");
	}

};

