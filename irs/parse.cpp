#include "parse.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>


bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

bool cmd_option_valid(const std::string* begin, const std::string* end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

char* cmd_option_value(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);
	/*if found option doesn't equal end and there is something in the following position*/
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return nullptr;
}

bool valid_double(char* val)
{
	char* error = nullptr;
	double result = std::strtod(val, &error);
	/*if result lies between the minimum and maximum double values, and there is no error in the conversion*/
	if (std::isfinite(result) && !(*error))
	{
		return true;
	}
	return false;
}

bool valid_float(char* val)
{
	char* error = nullptr;
	float result = std::strtof(val, &error);
	/*if result lies between the minimum and maximum float values, and there is no error in the conversion*/
	if (std::isfinite(result) && !(*error))
	{
		return true;
	}
	return false;
}

