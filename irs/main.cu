/*****************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

*****************************************************************/


#include "complex.cuh"
#include "irs_microlensing.cuh"
#include "star.cuh"
#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>



/*constants to be used*/
const float PI = 3.1415926535898f;
constexpr int OPTS_SIZE = 2 * 16;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-k", "--kappa_tot",
	"-s", "--shear",
	"-t", "--theta_e",
	"-ks", "--kappa_star",
	"-r", "--rectangular",
	"-ss", "--safety_scale",
	"-sf", "--starfile",
	"-hl", "--half_length",
	"-px", "--pixels",
	"-nr", "--num_rays",
	"-rs", "--random_seed",
	"-wm", "--write_maps",
	"-wp", "--write_parities",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};


/*default input option values*/
float kappa_tot = 0.3f;
float shear = 0.3f;
float theta_e = 1.0f;
float kappa_star = 0.27f;
int rectangular = 1;
float safety_scale = 1.37f;
std::string starfile = "";
float half_length = 5.0f;
int num_pixels = 1000;
int num_rays = 100;
int random_seed = 0;
int write_maps = 1;
int write_parities = 1;
std::string outfile_type = ".bin";
std::string outfile_prefix = "./";

/*default derived parameter values
number of stars, upper and lower mass cutoffs,
<m>, and <m^2>*/
int num_stars = 0;
float m_lower = 1.0f;
float m_upper = 1.0f;
float mean_mass = 1.0f;
float mean_squared_mass = 1.0f;



/************************************
Print the program usage help message

\param name -- name of the executable
************************************/
void display_usage(char* name)
{
	if (name)
	{
		std::cout << "Usage: " << name << " opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	else
	{
		std::cout << "Usage: programname opt1 val1 opt2 val2 opt3 val3 ...\n";
	}
	std::cout 
		<< "Options:\n"
		<< "   -h,--help              Show this help message.\n"
		<< "   -k,--kappa_tot         Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "   -s,--shear             Specify the external shear. Default value: " << shear << "\n"
		<< "   -t,--theta_e           Specify the size of the Einstein radius of a unit\n"
		<< "                          mass point lens in arbitrary units. Default value: " << theta_e << "\n"
		<< "   -ks,--kappa_star       Specify the convergence in point mass lenses.\n"
		<< "                          Default value: " << kappa_star << "\n"
		<< "   -r,--rectangular       Specify whether the star field should be\n"
		<< "                          rectangular (1) or circular (0). Default value: " << rectangular << "\n"
		<< "   -ss,--safety_scale     Specify the multiplicative safety factor over the\n"
		<< "                          shooting region to be used when generating the star\n"
		<< "                          field. Default value: " << safety_scale << "\n"
		<< "   -sf,--starfile         Specify the location of a star positions and masses\n"
		<< "                          file. The file may be either a whitespace delimited\n"
		<< "                          text file containing valid double precision values\n"
		<< "                          for a star's x coordinate, y coordinate, and mass, in\n"
		<< "                          that order, on each line, or a binary file of star\n"
		<< "                          structures (as defined in this source code). If\n"
		<< "                          specified, the number of stars is determined through\n"
		<< "                          this file.\n"
		<< "   -hl,--half_length      Specify the half-length of the square source plane\n"
		<< "                          region to find the magnification in.\n"
		<< "                          Default value: " << half_length << "\n"
		<< "   -px,--pixels           Specify the number of pixels per side for the\n"
		<< "                          magnification map. Default value: " << num_pixels << "\n"
		<< "   -nr,--num_rays         Specify the average number of rays per pixel.\n"
		<< "                          Default value: " << num_rays << "\n"
		<< "   -rs,--random_seed      Specify the random seed for star field generation.\n"
		<< "                          A value of 0 is reserved for star input files.\n"
		<< "   -wm,--write_maps       Specify whether to write magnification maps.\n"
		<< "                          Default value: " << write_maps << "\n"
		<< "   -wp,--write_parities   Specify whether to write parity specific\n"
		<< "                          magnification maps. Default value: " << write_parities << "\n"
		<< "   -ot,--outfile_type     Specify the type of file to be output. Valid options\n"
		<< "                          are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "   -o,--outfile_prefix    Specify the prefix to be used in output file names.\n"
		<< "                          Default value: " << outfile_prefix << "\n"
		<< "                          Lines of .txt output files are whitespace delimited.\n"
		<< "                          Filenames are:\n"
		<< "                             irs_parameter_info      various parameter values\n"
		<< "                                                        used in calculations\n"
		<< "                             irs_stars               the first item is\n"
		<< "                                                        num_stars followed by\n"
		<< "                                                        binary representations\n"
		<< "                                                        of the star structures\n"
		<< "                             irs_numrays_numpixels   each line contains a\n"
		<< "                                                        number of rays and the\n"
		<< "                                                        number of pixels with\n"
		<< "                                                        that many rays\n"
		<< "                             irs_magnifications      the first item is\n"
		<< "                                                        num_pixels and the\n"
		<< "                                                        second item is\n"
		<< "                                                        num_pixels followed by\n"
		<< "                                                        the number of rays in\n"
		<< "                                                        each pixel\n";
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



int main(int argc, char* argv[])
{
	/*set precision for printing numbers to screen*/
	std::cout.precision(7);

	/*if help option has been input, display usage message*/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/*if there are input options, but not an even number (since all options
	take a parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name*/
	if ((argc - 1) % 2 != 0)
	{
		std::cerr << "Error. Invalid input syntax.\n";
		display_usage(argv[0]);
		return -1;
	}

	/*check that all options given are valid. use step of 2 since all input
	options take parameters (assumed to be given immediately after the option)
	start at 1, since first array element, argv[0], is program name*/
	for (int i = 1; i < argc; i += 2)
	{
		if (!cmd_option_valid(OPTS, OPTS + OPTS_SIZE, argv[i]))
		{
			std::cerr << "Error. Invalid input syntax.\n";
			display_usage(argv[0]);
			return -1;
		}
	}


	/******************************************************************************
	BEGIN read in options and values, checking correctness and exiting if necessary
	******************************************************************************/

	char* cmdinput = nullptr;

	for (int i = 1; i < argc; i += 2)
	{
		cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				kappa_tot = std::stof(cmdinput);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--shear"))
		{
			try
			{
				shear = std::stof(cmdinput);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_e"))
		{
			try
			{
				theta_e = std::stof(cmdinput);
				if (theta_e < std::numeric_limits<float>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be > " << std::numeric_limits<float>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid theta_e input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_star"))
		{
			try
			{
				kappa_star = std::stof(cmdinput);
				if (kappa_star < std::numeric_limits<float>::min())
				{
					std::cerr << "Error. Invalid kappa_star input. kappa_star must be > " << std::numeric_limits<float>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_star input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-r") || argv[i] == std::string("--rectangular"))
		{
			try
			{
				rectangular = std::stoi(cmdinput);
				if (rectangular != 0 && rectangular != 1)
				{
					std::cerr << "Error. Invalid rectangular input. rectangular must be 1 (rectangular) or 0 (circular).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid rectangular input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ss") || argv[i] == std::string("--safety_scale"))
		{
			try
			{
				safety_scale = std::stof(cmdinput);
				if (safety_scale < 1)
				{
					std::cerr << "Error. Invalid safety_scale input. safety_scale must be > 1\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid safety_scale input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-sf") || argv[i] == std::string("--starfile"))
		{
			starfile = cmdinput;
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			try
			{
				half_length = std::stof(cmdinput);
				if (half_length < std::numeric_limits<float>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be > " << std::numeric_limits<float>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid half_length input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-px") || argv[i] == std::string("--pixels"))
		{
			try
			{
				num_pixels = std::stoi(cmdinput);
				if (num_pixels < 1)
				{
					std::cerr << "Error. Invalid num_pixels input. num_pixels must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_pixels input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-nr") || argv[i] == std::string("--num_rays"))
		{
			try
			{
				num_rays = std::stoi(cmdinput);
				if (num_rays < 1)
				{
					std::cerr << "Error. Invalid num_rays input. num_rays must be an integer > 0\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid num_rays input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-rs") || argv[i] == std::string("--random_seed"))
		{
			try
			{
				random_seed = std::stoi(cmdinput);
				if (random_seed == 0)
				{
					std::cerr << "Error. Invalid random_seed input. Seed of 0 is reserved for star input files.\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid random_seed input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wm") || argv[i] == std::string("--write_maps"))
		{
			try
			{
				write_maps = std::stoi(cmdinput);
				if (write_maps != 0 && write_maps != 1)
				{
					std::cerr << "Error. Invalid write_maps input. write_maps must be 0 (false) or 1 (true).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_maps input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wp") || argv[i] == std::string("--write_parities"))
		{
			try
			{
				write_parities = std::stoi(cmdinput);
				if (write_parities != 0 && write_parities != 1)
				{
					std::cerr << "Error. Invalid write_parities input. write_parities must be 0 (false) or 1 (true).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_parities input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ot") || argv[i] == std::string("--outfile_type"))
		{
			outfile_type = cmdinput;
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			outfile_prefix = cmdinput;
		}
	}

	/****************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	****************************************************************************/


	/*check that a CUDA capable device is present*/
	cudaSetDevice(0);
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/*if star file is specified, check validity of values and set num_stars, m_lower, m_upper,
	mean_mass, mean_squared_mass, and kappa_star based on star information*/
	if (starfile != "")
	{
		std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

		if (!read_star_params<float>(num_stars, m_lower, m_upper, mean_mass, mean_squared_mass, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n";
	}

	/*average magnification of the system*/
	float mu_ave = 1.0f / ((1.0f - kappa_tot) * (1.0f - kappa_tot) - shear * shear);

	/*number density of rays in the lens plane
	uses the fact that for a given user specified number density of rays
	in the source plane, further subdivisions are made that multiply the
	effective number of rays in the image plane by 27^2*/
	float num_rays_lens = 1.0f / (27.0f * 27.0f) * num_rays / std::abs(mu_ave) * (num_pixels * num_pixels) / (2.0f * half_length * 2.0f * half_length);

	/*average separation between rays in one dimension is 1/sqrt(number density)*/
	float ray_sep = 1.0f / std::sqrt(num_rays_lens);

	/*shooting region is greater than outer boundary for macro-mapping by the
	size of the region of images visible for a macro-image which contain 99%
	of the flux*/
	float lens_hl_x = (half_length + 10.0f * theta_e * std::sqrt(kappa_star * mean_squared_mass / mean_mass)) / std::abs(1.0f - kappa_tot + shear);
	float lens_hl_y = (half_length + 10.0f * theta_e * std::sqrt(kappa_star * mean_squared_mass / mean_mass)) / std::abs(1.0f - kappa_tot - shear);

	/*make shooting region a multiple of the ray separation*/
	lens_hl_x = ray_sep * (static_cast<int>(lens_hl_x / ray_sep) + 1.0f);
	lens_hl_y = ray_sep * (static_cast<int>(lens_hl_y / ray_sep) + 1.0f);
	
	/*if stars are not drawn from external file, calculate final number of stars to use*/
	if (starfile == "")
	{
		if (rectangular)
		{
			num_stars = static_cast<int>((safety_scale * 2.0f * lens_hl_x) * (safety_scale * 2.0f * lens_hl_y) * kappa_star / (PI * theta_e * theta_e * mean_mass)) + 1;
		}
		else
		{
			num_stars = static_cast<int>(safety_scale * safety_scale * (lens_hl_x * lens_hl_x + lens_hl_y * lens_hl_y) * kappa_star / (theta_e * theta_e * mean_mass)) + 1;
		}
	}
	
	std::cout << "Number of stars used: " << num_stars << "\n";

	Complex<float> c = std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4.0f * kappa_star)) 
		* Complex<float>(
			std::sqrt(std::abs((1.0f - kappa_tot - shear) / (1.0f - kappa_tot + shear))), 
			std::sqrt(std::abs((1.0f - kappa_tot + shear) / (1.0f - kappa_tot - shear)))
			);
	float rad = std::sqrt(theta_e * theta_e * num_stars * mean_mass / kappa_star);


	/**********************
	BEGIN memory allocation
	**********************/

	star<float>* stars = nullptr;
	int* pixels_minima = nullptr;
	int* pixels_saddles = nullptr;
	int* pixels = nullptr;

	/*allocate memory for stars*/
	cudaMallocManaged(&stars, num_stars * sizeof(star<float>));
	if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for pixels*/
	cudaMallocManaged(&pixels_minima, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*pixels_minima)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&pixels_saddles, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*pixels_saddles)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&pixels, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*pixels)", false, __FILE__, __LINE__)) return -1;

	/********************
	END memory allocation
	********************/


	/**************************
	BEGIN populating star array
	**************************/

	std::cout << "\n";

	if (starfile == "")
	{
		std::cout << "Generating star field...\n";

		/*generate random star field if no star file has been given
		if random seed is provided, use it,
		uses default star mass of 1.0*/
		if (random_seed != 0)
		{
			if (rectangular)
			{
				generate_rectangular_star_field<float>(stars, num_stars, c, 1.0f, random_seed);
			}
			else
			{
				generate_circular_star_field<float>(stars, num_stars, rad, 1.0f, random_seed);
			}
		}
		else
		{
			if (rectangular)
			{
				generate_rectangular_star_field<float>(stars, num_stars, c, 1.0f);
			}
			else
			{
				generate_circular_star_field<float>(stars, num_stars, rad, 1.0f);
			}
		}

		std::cout << "Done generating star field.\n";
	}
	else
	{
		/*ensure random seed is 0 to denote that stars come from external file*/
		random_seed = 0;

		std::cout << "Reading star field from file " << starfile << "\n";

		/*reading star field from external file*/
		if (!read_star_file<float>(stars, num_stars, starfile))
		{
			std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done reading star field from file " << starfile << "\n";
	}

	/************************
	END populating star array
	************************/


	/*initialize pixel values*/
	for (int i = 0; i < num_pixels * num_pixels; i++)
	{
		pixels_minima[i] = 0;
		pixels_saddles[i] = 0;
		pixels[i] = 0;
	}

	/*number of threads per block, and number of blocks per grid
	uses 16 for number of threads in x and y dimensions, as
	32*32=1024 is the maximum allowable number of threads per block
	but is too large for some memory allocation, and 16 is
	next power of 2 smaller*/

	int num_threads_y = 16;
	int num_threads_x = 16;

	int num_blocks_y = static_cast<int>((2.0f * lens_hl_y / ray_sep - 1) / num_threads_y) + 1;
	if (num_blocks_y > 32768 || num_blocks_y < 1)
	{
		num_blocks_y = 32768;
	}
	int num_blocks_x = static_cast<int>((2.0f * lens_hl_x / ray_sep - 1) / num_threads_x) + 1;
	if (num_blocks_x > 32768 || num_blocks_x < 1)
	{
		num_blocks_x = 32768;
	}
	dim3 blocks(num_blocks_x, num_blocks_y);
	dim3 threads(num_threads_x, num_threads_y);


	/*start and end time for timing purposes*/
	std::chrono::high_resolution_clock::time_point starttime;
	std::chrono::high_resolution_clock::time_point endtime;

	std::cout << "\nShooting rays...\n";
	/*get current time at start*/
	starttime = std::chrono::high_resolution_clock::now();

	if (rectangular)
	{
		shoot_rays_kernel << <blocks, threads >> > (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, c, lens_hl_x, lens_hl_y, ray_sep, half_length, pixels_minima, pixels_saddles, pixels, num_pixels);
	}
	else
	{
		shoot_rays_kernel << <blocks, threads >> > (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, lens_hl_x, lens_hl_y, ray_sep, half_length, pixels_minima, pixels_saddles, pixels, num_pixels);
	}
	if (cuda_error("shoot_rays_kernel", true, __FILE__, __LINE__)) return -1;
	/*get current time at end of loop, and calculate duration in milliseconds*/
	endtime = std::chrono::high_resolution_clock::now();
	double t_ray_shoot = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "Done shooting rays. Elapsed time: " << t_ray_shoot << " seconds.\n";

	
	/*create histogram of pixel values*/

	const int min_rays = *std::min_element(pixels, pixels + num_pixels * num_pixels);
	const int max_rays = *std::max_element(pixels, pixels + num_pixels * num_pixels);

	int* histogram = new (std::nothrow) int[max_rays - min_rays + 1];
	if (!histogram)
	{
		std::cerr << "Error. Memory allocation for *histogram failed.\n";
		return -1;
	}
	for (int i = 0; i <= max_rays - min_rays; i++)
	{
		histogram[i] = 0;
	}
	for (int i = 0; i < num_pixels * num_pixels; i++)
	{
		histogram[pixels[i] - min_rays]++;
	}

	

	/*stream for writing output files
	set precision to 9 digits*/
	std::ofstream outfile;
	outfile.precision(9);

	std::cout << "\nWriting parameter info...\n";
	outfile.open(outfile_prefix + "irs_parameter_info.txt");
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << (outfile_prefix + "irs_parameter_info.txt") << "\n";
		return -1;
	}
	outfile << "kappa_tot " << kappa_tot << "\n";
	outfile << "shear " << shear << "\n";
	outfile << "mu_ave " << mu_ave << "\n";
	outfile << "theta_e " << theta_e << "\n";
	outfile << "kappa_star " << kappa_star << "\n";
	outfile << "lower_mass_cutoff " << m_lower << "\n";
	outfile << "upper_mass_cutoff " << m_upper << "\n";
	outfile << "mean_mass " << mean_mass << "\n";
	outfile << "mean_squared_mass " << mean_squared_mass << "\n";
	outfile << "num_stars " << num_stars << "\n";
	if (rectangular)
	{
		outfile << "corner_x1 " << c.re << "\n";
		outfile << "corner_x2 " << c.im << "\n";
	}
	else
	{
		outfile << "rad " << rad << "\n";
	}
	outfile << "half_length " << half_length << "\n";
	outfile << "num_pixels " << num_pixels << "\n";
	outfile << "mean_rays_per_pixel " << num_rays << "\n";
	outfile << "random_seed " << random_seed << "\n";
	outfile << "lens_hl_x " << lens_hl_x << "\n";
	outfile << "lens_hl_y " << lens_hl_y << "\n";
	outfile << "ray_sep " << ray_sep << "\n";
	outfile << "t_ray_shoot " << t_ray_shoot << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << outfile_prefix << "irs_parameter_info.txt.\n";

	std::cout << "\nWriting star info...\n";
	if (!write_star_file<float>(stars, num_stars, outfile_prefix + "irs_stars" + outfile_type))
	{
		std::cerr << "Error. Unable to write star info to file " << outfile_prefix << "irs_stars" + outfile_type << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << outfile_prefix << "irs_stars" + outfile_type << "\n";


	/*histogram of magnification map*/
	std::cout << "\nWriting magnification histogram...\n";
	outfile.open(outfile_prefix + "irs_numrays_numpixels.txt");
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << outfile_prefix << "irs_numrays_numpixels.txt\n";
		return -1;
	}
	for (int i = 0; i <= max_rays - min_rays; i++)
	{
		if (histogram[i] != 0)
		{
			outfile << i + min_rays << " " << histogram[i] << "\n";
		}
	}
	outfile.close();
	std::cout << "Done writing magnification histogram to file " << outfile_prefix << "irs_numrays_numpixels.txt\n";


	/*write magnifications for minima, saddle, and combined maps*/
	if (write_maps)
	{
		std::cout << "\nWriting magnifications...\n";
		if (!write_array<int>(pixels, num_pixels, num_pixels, outfile_prefix + "irs_magnifications" + outfile_type))
		{
			std::cerr << "Error. Unable to write magnifications to file " << outfile_prefix << "irs_magnifications" + outfile_type << "\n";
			return -1;
		}
		std::cout << "Done writing magnifications to file " << outfile_prefix << "irs_magnifications" + outfile_type << "\n";
		if (write_parities)
		{
			if (!write_array<int>(pixels_minima, num_pixels, num_pixels, outfile_prefix + "irs_magnifications_minima" + outfile_type))
			{
				std::cerr << "Error. Unable to write magnifications to file " << outfile_prefix << "irs_magnifications_minima" + outfile_type << "\n";
				return -1;
			}
			std::cout << "Done writing magnifications to file " << outfile_prefix << "irs_magnifications_minima" + outfile_type << "\n";
			if (!write_array<int>(pixels_saddles, num_pixels, num_pixels, outfile_prefix + "irs_magnifications_saddles" + outfile_type))
			{
				std::cerr << "Error. Unable to write magnifications to file " << outfile_prefix << "irs_magnifications_saddles" + outfile_type << "\n";
				return -1;
			}
			std::cout << "Done writing magnifications to file " << outfile_prefix << "irs_magnifications_saddles" + outfile_type << "\n";
		}
	}

	std::cout << "\nDone.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

