#pragma once

#include "mass_functions/equal.cuh"
#include "mass_functions/kroupa.cuh"
#include "mass_functions/mass_function_base.cuh"
#include "mass_functions/optical_depth.cuh"
#include "mass_functions/salpeter.cuh"
#include "mass_functions/uniform.cuh"

#include <map>
#include <memory> //for std::shared_ptr and std::make_shared
#include <string>


namespace massfunctions
{
	template <typename T>
	const std::map<std::string, std::shared_ptr<MassFunction<T>>> MASS_FUNCTIONS
	{
		{"equal", std::make_shared<Equal<T>>()},
		{"uniform", std::make_shared<Uniform<T>>()},
		{"salpeter", std::make_shared<Salpeter<T>>()},
		{"kroupa", std::make_shared<Kroupa<T>>()},
		{"optical_depth", std::make_shared<OpticalDepth<T>>()}
	};

}

