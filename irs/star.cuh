#pragma once

#include "complex.cuh"


/*structure to hold position and mass of a star*/
template <typename T>
struct star
{
	Complex<T> position;
	T mass;
};

