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
examples: [====    ] 50%       [=====  ] 73%

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

/*********************************************************************
CUDA error checking

\param name -- to print in error msg
\param sync -- boolean of whether device needs synchronized or not
\param name -- the file being run
\param line -- line number of the source code where the error is given

\return bool -- true for error, false for no error
*********************************************************************/
bool cuda_error(const char* name, bool sync, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();
	/*if the last error message is not a success, print the error code and msg
	and return true (i.e., an error occurred)*/
	if (err != cudaSuccess)
	{
		const char* errMsg = cudaGetErrorString(err);
		std::cerr << "CUDA error check for " << name << " failed at " << file << ":" << line << "\n";
		std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
		return true;
	}
	/*if a device synchronization is also to be done*/
	if (sync)
	{
		/*perform the same error checking as initially*/
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			const char* errMsg = cudaGetErrorString(err);
			std::cerr << "CUDA error check for cudaDeviceSynchronize failed at " << file << ":" << line << "\n";
			std::cerr << "Error code: " << err << " (" << errMsg << ")\n";
			return true;
		}
	}
	return false;
}

/*****************************************************
set the number of threads per block for the kernel

\param threads -- reference to threads
\param x -- number of threads per block in x dimension
\param y -- number of threads per block in y dimension
\param z -- number of threads per block in z dimension
*****************************************************/
void set_threads(dim3& threads, int x = 1, int y = 1, int z = 1)
{
	threads.x = x;
	threads.y = y;
	threads.z = z;
}

/*******************************************
set the number of blocks for the kernel

\param threads -- reference to threads
\param blocks -- reference to blocks
\param x -- number of threads in x dimension
\param y -- number of threads in y dimension
\param z -- number of threads in z dimension
*******************************************/
void set_blocks(dim3& threads, dim3& blocks, int x = 1, int y = 1, int z = 1)
{
	blocks.x = static_cast<int>((x - 1) / threads.x) + 1;
	blocks.y = static_cast<int>((y - 1) / threads.y) + 1;
	blocks.z = static_cast<int>((z - 1) / threads.z) + 1;
}

