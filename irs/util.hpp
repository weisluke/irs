#pragma once

#include <algorithm>
#include <iostream>
#include <string>


/*******************************************************
determine if a string exists within some array of char*s

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find within the array

\returns bool -- true if string is in array, false if not
********************************************************/
bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

/***************************************************************
determine if a string is a valid option from some options array

\param begin -- starting pointer of array of string options
\param end -- ending pointer of array of string options
\param option -- string to find within the array

\returns bool -- true if string was found in array, false if not
***************************************************************/
bool cmd_option_valid(const std::string* begin, const std::string* end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

/**************************************************************
determine the parameter value of an option in array of char*s
assumed to be placed immediately after the option in some range

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find the value of

\returns char* -- array of chars of the value after the option
**************************************************************/
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

/****************************************************
function to print out progress bar of loops
examples: [====    ] 50%       [=====  ] 62%

\param icurr -- current position in the loop
\param imax -- maximum position in the loop
\param num_bars -- number of = symbols inside the bar
				   default value: 50
****************************************************/
void print_progress(int icurr, int imax, int num_bars = 50)
{
	std::cout << "\r[";
	for (int i = 0; i < num_bars; i++)
	{
		if (i <= icurr * num_bars / imax)
		{
			std::cout << "=";
		}
		else
		{
			std::cout << " ";
		}
	}
	std::cout << "] " << icurr * 100 / imax << " %";
}

