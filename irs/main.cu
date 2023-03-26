/*****************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

*****************************************************************/


#include "complex.cuh"
#include "irs_microlensing.cuh"
#include "star.cuh"
#include "util.hpp"

#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>


using dtype = float;

/*constants to be used*/
const dtype PI = static_cast<dtype>(3.1415926535898);
constexpr int OPTS_SIZE = 2 * 18;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-k", "--kappa_tot",
	"-s", "--shear",
	"-t", "--theta_e",
	"-ks", "--kappa_star",
	"-r", "--rectangular",
	"-a", "--approx",
	"-ss", "--safety_scale",
	"-sf", "--starfile",
	"-hl", "--half_length",
	"-px", "--pixels",
	"-nr", "--num_rays",
	"-rs", "--random_seed",
	"-wm", "--write_maps",
	"-wp", "--write_parities",
	"-wh", "--write_histograms",
	"-ot", "--outfile_type",
	"-o", "--outfile_prefix"
};


/*default input option values*/
dtype kappa_tot = static_cast<dtype>(0.3);
dtype shear = static_cast<dtype>(0.3);
dtype theta_e = static_cast<dtype>(1);
dtype kappa_star = static_cast<dtype>(0.27);
int rectangular = 1;
int approx = 1;
dtype safety_scale = static_cast<dtype>(1.37);
std::string starfile = "";
dtype half_length = static_cast<dtype>(5);
int num_pixels = 1000;
int num_rays = 100;
int random_seed = 0;
int write_maps = 1;
int write_parities = 0;
int write_histograms = 1;
std::string outfile_type = ".bin";
std::string outfile_prefix = "./";

/*default derived parameter values
number of stars, upper and lower mass cutoffs,
<m>, and <m^2>*/
int num_stars = 0;
dtype m_lower = static_cast<dtype>(1);
dtype m_upper = static_cast<dtype>(1);
dtype mean_mass = static_cast<dtype>(1);
dtype mean_squared_mass = static_cast<dtype>(1);



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
		<< "                                                                               \n"
		<< "Options:\n"
		<< "  -h,--help               Show this help message.\n"
		<< "  -k,--kappa_tot          Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "  -s,--shear              Specify the external shear. Default value: " << shear << "\n"
		<< "  -t,--theta_e            Specify the size of the Einstein radius of a unit\n"
		<< "                          mass point lens in arbitrary units. Default value: " << theta_e << "\n"
		<< "  -ks,--kappa_star        Specify the convergence in point mass lenses.\n"
		<< "                          Default value: " << kappa_star << "\n"
		<< "  -r,--rectangular        Specify whether the star field should be\n"
		<< "                          rectangular (1) or circular (0). Default value: " << rectangular << "\n"
		<< "  -a,--approx             Specify whether terms for alpha_smooth should be\n"
		<< "                          approximated (1) or exact (0). Default value: " << approx << "\n"
		<< "  -ss,--safety_scale      Specify the multiplicative safety factor over the\n"
		<< "                          shooting region to be used when generating the star\n"
		<< "                          field. Default value: " << safety_scale << "\n"
		<< "  -sf,--starfile          Specify the location of a star positions and masses\n"
		<< "                          file. The file may be either a whitespace delimited\n"
		<< "                          text file containing valid double precision values\n"
		<< "                          for a star's x coordinate, y coordinate, and mass, in\n"
		<< "                          that order, on each line, or a binary file of star\n"
		<< "                          structures (as defined in this source code). If\n"
		<< "                          specified, the number of stars is determined through\n"
		<< "                          this file.\n"
		<< "  -hl,--half_length       Specify the half-length of the square source plane\n"
		<< "                          region to find the magnification in.\n"
		<< "                          Default value: " << half_length << "\n"
		<< "  -px,--pixels            Specify the number of pixels per side for the\n"
		<< "                          magnification map. Default value: " << num_pixels << "\n"
		<< "  -nr,--num_rays          Specify the average number of rays per pixel.\n"
		<< "                          Default value: " << num_rays << "\n"
		<< "  -rs,--random_seed       Specify the random seed for star field generation.\n"
		<< "                          A value of 0 is reserved for star input files.\n"
		<< "  -wm,--write_maps        Specify whether to write magnification maps (1) or\n"
		<< "                          not (0). Default value: " << write_maps << "\n"
		<< "  -wp,--write_parities    Specify whether to write parity specific\n"
		<< "                          magnification maps (1) or not (0). Default value: " << write_parities << "\n"
		<< "  -wh,--write_histograms  Specify whether to write histograms (1) or not (0).\n"
		<< "                          Default value: " << write_histograms << "\n"
		<< "  -ot,--outfile_type      Specify the type of file to be output. Valid options\n"
		<< "                          are binary (.bin) or text (.txt). Default value: " << outfile_type << "\n"
		<< "  -o,--outfile_prefix     Specify the prefix to be used in output file names.\n"
		<< "                          Default value: " << outfile_prefix << "\n"
		<< "                          Lines of .txt output files are whitespace delimited.\n"
		<< "                          Filenames are:\n"
		<< "                            irs_parameter_info     various parameter values\n"
		<< "                                                     used in calculations\n"
		<< "                            irs_stars              the first item is num_stars\n"
		<< "                                                     followed by binary\n"
		<< "                                                     representations of the\n"
		<< "                                                     star structures\n"
		<< "                            irs_numrays_numpixels  each line contains a number\n"
		<< "                                                     of rays and the number of\n"
		<< "                                                     pixels with that many rays\n"
		<< "                            irs_magnifications     the first item is num_pixels\n"
		<< "                                                     and the second item is\n"
		<< "                                                     num_pixels followed by the\n"
		<< "                                                     number of rays in each\n"
		<< "                                                     pixel\n";
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
			std::cerr << "Error. Invalid input syntax. Unknown option " << argv[i] << "\n";
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
				kappa_tot = static_cast<dtype>(std::stod(cmdinput));
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
				shear = static_cast<dtype>(std::stod(cmdinput));;
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
				theta_e = static_cast<dtype>(std::stod(cmdinput));;
				if (theta_e < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be > " << std::numeric_limits<dtype>::min() << "\n";
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
				kappa_star = static_cast<dtype>(std::stod(cmdinput));;
				if (kappa_star < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid kappa_star input. kappa_star must be > " << std::numeric_limits<dtype>::min() << "\n";
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
		else if (argv[i] == std::string("-a") || argv[i] == std::string("--approx"))
		{
			try
			{
				approx = std::stoi(cmdinput);
				if (approx != 0 && approx != 1)
				{
					std::cerr << "Error. Invalid approx input. approx must be 1 (approximate) or 0 (exact).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid approx input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ss") || argv[i] == std::string("--safety_scale"))
		{
			try
			{
				safety_scale = static_cast<dtype>(std::stod(cmdinput));;
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
				half_length = static_cast<dtype>(std::stod(cmdinput));;
				if (half_length < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be > " << std::numeric_limits<dtype>::min() << "\n";
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
				if (random_seed == 0 && !(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--star_file")))
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
					std::cerr << "Error. Invalid write_maps input. write_maps must be 1 (true) or 0 (false).\n";
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
					std::cerr << "Error. Invalid write_parities input. write_parities must be 1 (true) or 0 (false).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_parities input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-wh") || argv[i] == std::string("--write_histograms"))
		{
			try
			{
				write_histograms = std::stoi(cmdinput);
				if (write_histograms != 0 && write_histograms != 1)
				{
					std::cerr << "Error. Invalid write_histograms input. write_histograms must be 1 (true) or 0 (false).\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid write_histograms input.\n";
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

		if (!read_star_params<dtype>(num_stars, m_lower, m_upper, mean_mass, mean_squared_mass, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n\n";
	}

	/*average magnification of the system*/
	dtype mu_ave = 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear);

	/*number density of rays in the lens plane
	uses the fact that for a given user specified number density of rays
	in the source plane, further subdivisions are made that multiply the
	effective number of rays in the image plane by 27^2*/
	dtype num_rays_lens = num_rays / std::abs(mu_ave) * num_pixels * num_pixels / (2 * half_length * 2 * half_length) * 1 / (27 * 27);

	/*average separation between rays in one dimension is 1/sqrt(number density)*/
	dtype ray_sep = 1 / std::sqrt(num_rays_lens);

	/*shooting region is greater than outer boundary for macro-mapping by the
	size of the region of images visible for a macro-image which contain 99%
	of the flux*/
	dtype lens_hl_x1 = (half_length + 10 * theta_e * std::sqrt(kappa_star * mean_squared_mass / mean_mass)) / std::abs(1 - kappa_tot + shear);
	dtype lens_hl_x2 = (half_length + 10 * theta_e * std::sqrt(kappa_star * mean_squared_mass / mean_mass)) / std::abs(1 - kappa_tot - shear);

	/*make shooting region a multiple of the ray separation*/
	lens_hl_x1 = ray_sep * (static_cast<int>(lens_hl_x1 / ray_sep) + 1);
	lens_hl_x2 = ray_sep * (static_cast<int>(lens_hl_x2 / ray_sep) + 1);

	/*if stars are not drawn from external file, calculate final number of stars to use*/
	if (starfile == "")
	{
		if (rectangular)
		{
			num_stars = static_cast<int>((safety_scale * 2 * lens_hl_x1) * (safety_scale * 2 * lens_hl_x2) 
				* kappa_star / (PI * theta_e * theta_e * mean_mass)) + 1;
		}
		else
		{
			num_stars = static_cast<int>(safety_scale * safety_scale * (lens_hl_x1 * lens_hl_x1 + lens_hl_x2 * lens_hl_x2) 
				* kappa_star / (theta_e * theta_e * mean_mass)) + 1;
		}
	}

	std::cout << "Number of stars used: " << num_stars << "\n\n";

	Complex<dtype> c = std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
		* Complex<dtype>(
			std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
			std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
			);
	dtype rad = std::sqrt(theta_e * theta_e * num_stars * mean_mass / kappa_star);

	int taylor = static_cast<int>(std::log(2 * kappa_star * c.abs() / (2 * half_length / num_pixels * PI)) / std::log(safety_scale));
	if (taylor < 1)
	{
		taylor = 1;
	}

	/**********************
	BEGIN memory allocation
	**********************/

	std::cout << "Beginning memory allocation...\n";

	curandState* states = nullptr;
	star<dtype>* stars = nullptr;
	int* pixels = nullptr;
	int* pixels_minima = nullptr;
	int* pixels_saddles = nullptr;

	/*allocate memory for stars*/
	cudaMallocManaged(&states, num_stars * sizeof(curandState));
	if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&stars, num_stars * sizeof(star<dtype>));
	if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;

	/*allocate memory for pixels*/
	cudaMallocManaged(&pixels, num_pixels * num_pixels * sizeof(int));
	if (cuda_error("cudaMallocManaged(*pixels)", false, __FILE__, __LINE__)) return -1;
	if (write_parities)
	{
		cudaMallocManaged(&pixels_minima, num_pixels * num_pixels * sizeof(int));
		if (cuda_error("cudaMallocManaged(*pixels_minima)", false, __FILE__, __LINE__)) return -1;
		cudaMallocManaged(&pixels_saddles, num_pixels * num_pixels * sizeof(int));
		if (cuda_error("cudaMallocManaged(*pixels_saddles)", false, __FILE__, __LINE__)) return -1;
	}

	std::cout << "Done allocating memory.\n\n";

	/********************
	END memory allocation
	********************/


	/*number of threads per block, and number of blocks per grid
	uses 512 for number of threads in x dimension, as 1024 is the
	maximum allowable number of threads per block but is too large
	for some memory allocation, and 512 is next power of 2 smaller*/

	int num_threads_z = 1;
	int num_threads_y = 1;
	int num_threads_x = 512;

	int num_blocks_z = 1;
	int num_blocks_y = 1;
	int num_blocks_x = static_cast<int>((num_stars - 1) / num_threads_x) + 1;

	dim3 blocks(num_blocks_x, num_blocks_y, num_blocks_z);
	dim3 threads(num_threads_x, num_threads_y, num_threads_z);


	/**************************
	BEGIN populating star array
	**************************/


	if (starfile == "")
	{
		std::cout << "Generating star field...\n";

		/*if random seed was not provided, get one based on the time*/
		if (random_seed == 0)
		{
			random_seed = static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count());
		}

		/*generate random star field if no star file has been given
		uses default star mass of 1.0*/
		initialize_curand_states_kernel<dtype> <<<blocks, threads>>> (states, num_stars, random_seed);
		if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return -1;
		if (rectangular)
		{
			generate_rectangular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, c, static_cast<dtype>(1));
		}
		else
		{
			generate_circular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, rad, static_cast<dtype>(1));
		}
		if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return -1;

		std::cout << "Done generating star field.\n\n";
	}
	else
	{
		/*ensure random seed is 0 to denote that stars come from external file*/
		random_seed = 0;

		std::cout << "Reading star field from file " << starfile << "\n";

		/*reading star field from external file*/
		if (!read_star_file<dtype>(stars, num_stars, starfile))
		{
			std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done reading star field from file " << starfile << "\n\n";
	}

	/************************
	END populating star array
	************************/


	/*redefine thread and block size to maximize parallelization*/
	num_threads_y = 16;
	num_threads_x = 16;

	num_blocks_y = static_cast<int>((2 * lens_hl_x2 / ray_sep - 1) / num_threads_y) + 1;
	num_blocks_x = static_cast<int>((2 * lens_hl_x1 / ray_sep - 1) / num_threads_x) + 1;

	blocks.x = num_blocks_x;
	blocks.y = num_blocks_y;
	threads.x = num_threads_x;
	threads.y = num_threads_y;


	/*initialize pixel values*/
	initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels, num_pixels);
	if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
	if (write_parities)
	{
		initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels_minima, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
		initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels_saddles, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
	}


	/*start and end time for timing purposes*/
	std::chrono::high_resolution_clock::time_point starttime;
	std::chrono::high_resolution_clock::time_point endtime;

	std::cout << "Shooting rays...\n";
	/*get current time at start*/
	starttime = std::chrono::high_resolution_clock::now();
	shoot_rays_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star, rectangular, c, approx, taylor, lens_hl_x1, lens_hl_x2, ray_sep, half_length, pixels_minima, pixels_saddles, pixels, num_pixels);
	if (cuda_error("shoot_rays_kernel", true, __FILE__, __LINE__)) return -1;
	/*get current time at end of loop, and calculate duration in milliseconds*/
	endtime = std::chrono::high_resolution_clock::now();
	double t_ray_shoot = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime).count() / 1000.0;
	std::cout << "Done shooting rays. Elapsed time: " << t_ray_shoot << " seconds.\n\n";


	/********************************
	create histograms of pixel values
	********************************/

	int* min_rays = nullptr;
	int* max_rays = nullptr;

	int* histogram = nullptr;
	int* histogram_minima = nullptr;
	int* histogram_saddles = nullptr;

	int histogram_length = 0;

	if (write_histograms)
	{
		std::cout << "Creating histograms...\n";

		cudaMallocManaged(&min_rays, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_rays)", false, __FILE__, __LINE__)) return -1;
		cudaMallocManaged(&max_rays, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_rays)", false, __FILE__, __LINE__)) return -1;

		*min_rays = std::numeric_limits<int>::max();
		*max_rays = 0;

		/*redefine thread and block size to maximize parallelization*/
		num_threads_y = 16;
		num_threads_x = 16;

		num_blocks_y = static_cast<int>((num_pixels - 1) / num_threads_y) + 1;
		num_blocks_x = static_cast<int>((num_pixels - 1) / num_threads_x) + 1;

		blocks.x = num_blocks_x;
		blocks.y = num_blocks_y;
		threads.x = num_threads_x;
		threads.y = num_threads_y;

		histogram_min_max_kernel<dtype> <<<blocks, threads>>> (pixels, num_pixels, min_rays, max_rays);
		if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return -1;
		if (write_parities)
		{
			histogram_min_max_kernel<dtype> <<<blocks, threads>>> (pixels_minima, num_pixels, min_rays, max_rays);
			if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return -1;
			histogram_min_max_kernel<dtype> <<<blocks, threads>>> (pixels_saddles, num_pixels, min_rays, max_rays);
			if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return -1;
		}

		histogram_length = *max_rays - *min_rays + 1;

		cudaMallocManaged(&histogram, histogram_length * sizeof(int));
		if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return -1;
		if (write_parities)
		{
			cudaMallocManaged(&histogram_minima, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram_minima)", false, __FILE__, __LINE__)) return -1;
			cudaMallocManaged(&histogram_saddles, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram_saddles)", false, __FILE__, __LINE__)) return -1;
		}

		/*redefine thread and block size to maximize parallelization*/
		num_threads_y = 1;
		num_threads_x = 512;

		num_blocks_y = 1;
		num_blocks_x = static_cast<int>((histogram_length - 1) / num_threads_x) + 1;

		blocks.x = num_blocks_x;
		blocks.y = num_blocks_y;
		threads.x = num_threads_x;
		threads.y = num_threads_y;

		initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram, histogram_length);
		if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
		if (write_parities)
		{
			initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram_minima, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
			initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram_saddles, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
		}

		/*redefine thread and block size to maximize parallelization*/
		num_threads_y = 16;
		num_threads_x = 16;

		num_blocks_y = static_cast<int>((num_pixels - 1) / num_threads_y) + 1;
		num_blocks_x = static_cast<int>((num_pixels - 1) / num_threads_x) + 1;

		blocks.x = num_blocks_x;
		blocks.y = num_blocks_y;
		threads.x = num_threads_x;
		threads.y = num_threads_y;

		histogram_kernel<dtype> <<<blocks, threads>>> (pixels, num_pixels, *min_rays, histogram);
		if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return -1;
		if (write_parities)
		{
			histogram_kernel<dtype> <<<blocks, threads>>> (pixels_minima, num_pixels, *min_rays, histogram_minima);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return -1;
			histogram_kernel<dtype> <<<blocks, threads>>> (pixels_saddles, num_pixels, *min_rays, histogram_saddles);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return -1;
		}

		std::cout << "Done creating histograms.\n\n";
	}
	/***************************************
	done creating histograms of pixel values
	***************************************/


	/*stream for writing output files
	set precision to 9 digits*/
	std::ofstream outfile;
	outfile.precision(9);
	std::string fname;


	fname = outfile_prefix + "irs_parameter_info.txt";
	std::cout << "Writing parameter info...\n";
	outfile.open(fname);
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
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
		if (approx)
		{
			outfile << "taylor " << taylor << "\n";
		}
	}
	else
	{
		outfile << "rad " << rad << "\n";
	}
	outfile << "safety_scale " << safety_scale << "\n";
	outfile << "half_length " << half_length << "\n";
	outfile << "num_pixels " << num_pixels << "\n";
	outfile << "mean_rays_per_pixel " << num_rays << "\n";
	outfile << "random_seed " << random_seed << "\n";
	outfile << "lens_hl_x1 " << lens_hl_x1 << "\n";
	outfile << "lens_hl_x2 " << lens_hl_x2 << "\n";
	outfile << "ray_sep " << ray_sep << "\n";
	outfile << "t_ray_shoot " << t_ray_shoot << "\n";
	outfile.close();
	std::cout << "Done writing parameter info to file " << fname << "\n\n";


	fname = outfile_prefix + "irs_stars" + outfile_type;
	std::cout << "Writing star info...\n";
	if (!write_star_file<dtype>(stars, num_stars, fname))
	{
		std::cerr << "Error. Unable to write star info to file " << fname << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << fname << "\n\n";


	/*histograms of magnification maps*/
	if (write_histograms)
	{
		std::cout << "Writing magnification histograms...\n";

		fname = outfile_prefix + "irs_numrays_numpixels.txt";
		if (!write_histogram<dtype>(histogram, histogram_length, *min_rays, fname))
		{
			std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing magnification histogram to file " << fname << "\n";
		if (write_parities)
		{
			fname = outfile_prefix + "irs_numrays_numpixels_minima.txt";
			if (!write_histogram<dtype>(histogram_minima, histogram_length, *min_rays, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return -1;
			}
			std::cout << "Done writing magnification histogram to file " << fname << "\n";

			fname = outfile_prefix + "irs_numrays_numpixels_saddles.txt";
			if (!write_histogram<dtype>(histogram_saddles, histogram_length, *min_rays, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return -1;
			}
			std::cout << "Done writing magnification histogram to file " << fname << "\n";
		}
		std::cout << "\n";
	}


	/*write magnifications for minima, saddle, and combined maps*/
	if (write_maps)
	{
		std::cout << "Writing magnifications...\n";

		fname = outfile_prefix + "irs_magnifications" + outfile_type;
		if (!write_array<int>(pixels, num_pixels, num_pixels, fname))
		{
			std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
			return -1;
		}
		std::cout << "Done writing magnifications to file " << fname << "\n";
		if (write_parities)
		{
			fname = outfile_prefix + "irs_magnifications_minima" + outfile_type;
			if (!write_array<int>(pixels_minima, num_pixels, num_pixels, fname))
			{
				std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
				return -1;
			}
			std::cout << "Done writing magnifications to file " << fname << "\n";

			fname = outfile_prefix + "irs_magnifications_saddles" + outfile_type;
			if (!write_array<int>(pixels_saddles, num_pixels, num_pixels, fname))
			{
				std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
				return -1;
			}
			std::cout << "Done writing magnifications to file " << fname << "\n";
		}
		std::cout << "\n";
	}

	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

