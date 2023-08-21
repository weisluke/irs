/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "irs_microlensing.cuh"
#include "mass_function.cuh"
#include "tree_node.cuh"
#include "star.cuh"
#include "util.hpp"

#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>


using dtype = float;

/******************************************************************************
constants to be used
******************************************************************************/
const dtype PI = 3.1415926535898;
constexpr int OPTS_SIZE = 2 * 25;
const std::string OPTS[OPTS_SIZE] =
{
	"-h", "--help",
	"-v", "--verbose",
	"-k", "--kappa_tot",
	"-y", "--shear",
	"-s", "--smooth_fraction",
	"-ks", "--kappa_star",
	"-t", "--theta_e",
	"-mf", "--mass_function",
	"-ms", "--m_solar",
	"-ml", "--m_lower",
	"-mh", "--m_upper",
	"-ll", "--light_loss",
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
const std::map<std::string, enumMassFunction> MASS_FUNCTIONS{
	{"equal", equal},
	{"uniform", uniform},
	{"salpeter", salpeter},
	{"kroupa", kroupa}
};


/******************************************************************************
default input option values
******************************************************************************/
bool verbose = false;
dtype kappa_tot = 0.3;
dtype shear = 0.3;
dtype smooth_fraction = 0.1;
dtype kappa_star = 0.27;
dtype theta_e = 1;
std::string mass_function_str = "equal";
dtype m_solar = 1;
dtype m_lower = 0.01;
dtype m_upper = 50;
dtype light_loss = 0.01;
int rectangular = 1;
int approx = 1;
dtype safety_scale = 1.37;
std::string starfile = "";
dtype half_length = 5;
int num_pixels = 1000;
int num_rays = 100;
int random_seed = 0;
int write_maps = 1;
int write_parities = 0;
int write_histograms = 1;
std::string outfile_type = ".bin";
std::string outfile_prefix = "./";



/******************************************************************************
Print the program usage help message

\param name -- name of the executable
******************************************************************************/
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
		<< "  -v,--verbose            Toggle verbose output. Takes no option value.\n"
		<< "  -k,--kappa_tot          Specify the total convergence. Default value: " << kappa_tot << "\n"
		<< "  -y,--shear              Specify the external shear. Default value: " << shear << "\n"
		<< "  -s,--smooth_fraction    Specify the fraction of convergence due to smoothly\n"
		<< "                          distributed mass. Default value: " << smooth_fraction << "\n"
		<< "  -ks,--kappa_star        Specify the convergence in point mass lenses. If\n"
		<< "                          provided, this overrides any supplied value for the\n"
		<< "                          smooth fraction. Default value: " << kappa_star << "\n"
		<< "  -t,--theta_e            Specify the size of the Einstein radius of a unit\n"
		<< "                          mass point lens in arbitrary units. Default value: " << theta_e << "\n"
		<< "  -mf,--mass_function     Specify the mass function to use for the point mass\n"
		<< "                          lenses. Options are: equal, uniform, Salpeter, and\n"
		<< "                          Kroupa. Default value: " << mass_function_str << "\n"
		<< "  -ms,--m_solar           Specify the solar mass in arbitrary units.\n"
		<< "                          Default value: " << m_solar << "\n"
		<< "  -ml,--m_lower           Specify the lower mass cutoff in arbitrary units.\n"
		<< "                          Default value: " << m_lower << "\n"
		<< "  -mh,--m_upper           Specify the upper mass cutoff in arbitrary units.\n"
		<< "                          Default value: " << m_upper << "\n"
		<< "  -ll,--light_loss        Allowed average fraction of light lost due to scatter\n"
		<< "                          by the microlenses in the large deflection limit.\n"
		<< "                          Default value: " << light_loss << "\n"
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
	/******************************************************************************
	set precision for printing numbers to screen
	******************************************************************************/
	std::cout.precision(7);

	/******************************************************************************
	if help option has been input, display usage message
	******************************************************************************/
	if (cmd_option_exists(argv, argv + argc, "-h") || cmd_option_exists(argv, argv + argc, "--help"))
	{
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	if there are input options, but not an even number (since all options take a
	parameter), display usage message and exit
	subtract 1 to take into account that first argument array value is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	if ((argc - 1) % 2 != 0 &&
		!(cmd_option_exists(argv, argv + argc, "-v") || cmd_option_exists(argv, argv + argc, "--verbose")))
	{
		std::cerr << "Error. Invalid input syntax.\n";
		display_usage(argv[0]);
		return -1;
	}

	/******************************************************************************
	check that all options given are valid. use step of 2 since all input options
	take parameters (assumed to be given immediately after the option). start at 1,
	since first array element, argv[0], is program name
	account for possible verbose option, which is a toggle and takes no input
	******************************************************************************/
	for (int i = 1; i < argc; i += 2)
	{
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			verbose = true;
			i--;
			continue;
		}
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
		/******************************************************************************
		account for possible verbose option, which is a toggle and takes no input
		******************************************************************************/
		if (argv[i] == std::string("-v") || argv[i] == std::string("--verbose"))
		{
			i--;
			continue;
		}

		cmdinput = cmd_option_value(argv, argv + argc, std::string(argv[i]));

		if (argv[i] == std::string("-k") || argv[i] == std::string("--kappa_tot"))
		{
			try
			{
				set_param("kappa_tot", kappa_tot, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid kappa_tot input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-y") || argv[i] == std::string("--shear"))
		{
			try
			{
				set_param("shear", shear, std::stod(cmdinput), verbose);
			}
			catch (...)
			{
				std::cerr << "Error. Invalid shear input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-s") || argv[i] == std::string("--smooth_fraction"))
		{
			if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
			{
				continue;
			}
			try
			{
				set_param("smooth_fraction", smooth_fraction, std::stod(cmdinput), verbose);
				if (smooth_fraction < 0)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be >= 0\n";
					return -1;
				}
				else if (smooth_fraction >= 1)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be < 1\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid smooth_fraction input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ks") || argv[i] == std::string("--kappa_star"))
		{
			try
			{
				set_param("kappa_star", kappa_star, std::stod(cmdinput), verbose);
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
		else if (argv[i] == std::string("-t") || argv[i] == std::string("--theta_e"))
		{
			try
			{
				set_param("theta_e", theta_e, std::stod(cmdinput), verbose);
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
		else if (argv[i] == std::string("-mf") || argv[i] == std::string("--mass_function"))
		{
			set_param("mass_function", mass_function_str, make_lowercase(cmdinput), verbose);
			if (!MASS_FUNCTIONS.count(mass_function_str))
			{
				std::cerr << "Error. Invalid mass_function input. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			try
			{
				set_param("m_solar", m_solar, std::stod(cmdinput), verbose);
				if (m_solar < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_solar input. m_solar must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_solar input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ml") || argv[i] == std::string("--m_lower"))
		{
			try
			{
				set_param("m_lower", m_lower, std::stod(cmdinput), verbose);
				if (m_lower < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_lower input. m_lower must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_lower input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-mh") || argv[i] == std::string("--m_upper"))
		{
			try
			{
				set_param("m_upper", m_upper, std::stod(cmdinput), verbose);
				if (m_upper < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (m_upper > std::numeric_limits<dtype>::max())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be < " << std::numeric_limits<dtype>::max() << "\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid m_upper input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ll") || argv[i] == std::string("--light_loss"))
		{
			try
			{
				set_param("light_loss", light_loss, std::stod(cmdinput), verbose);
				if (light_loss < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid light_loss input. light_loss must be > " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (light_loss > 0.01)
				{
					std::cerr << "Error. Invalid light_loss input. light_loss must be < 0.01\n";
					return -1;
				}
			}
			catch (...)
			{
				std::cerr << "Error. Invalid light_loss input.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-r") || argv[i] == std::string("--rectangular"))
		{
			try
			{
				set_param("rectangular", rectangular, std::stoi(cmdinput), verbose);
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
				set_param("approx", approx, std::stoi(cmdinput), verbose);
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
				set_param("safety_scale", safety_scale, std::stod(cmdinput), verbose);
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
			set_param("starfile", starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			try
			{
				set_param("half_length", half_length, std::stod(cmdinput), verbose);
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
				set_param("num_pixels", num_pixels, std::stoi(cmdinput), verbose);
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
				set_param("num_rays", num_rays, std::stoi(cmdinput), verbose);
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
				set_param("random_seed", random_seed, std::stoi(cmdinput), verbose);
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
				set_param("write_maps", write_maps, std::stoi(cmdinput), verbose);
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
				set_param("write_parities", write_parities, std::stoi(cmdinput), verbose);
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
				set_param("write_histograms", write_histograms, std::stoi(cmdinput), verbose);
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
			set_param("outfile_type", outfile_type, make_lowercase(cmdinput), verbose);
			if (outfile_type != ".bin" && outfile_type != ".txt")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin or .txt\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", outfile_prefix, cmdinput, verbose);
		}
	}

	std::cout << "\n";

	if (m_lower >= m_upper)
	{
		std::cerr << "Error. m_lower must be less than m_upper.\n";
		return -1;
	}

	if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
	{
		set_param("smooth_fraction", smooth_fraction, 1 - kappa_star / kappa_tot, verbose, true);
	}
	else
	{
		set_param("kappa_star", kappa_star, (1 - smooth_fraction) * kappa_tot, verbose, true);
	}

	/******************************************************************************
	END read in options and values, checking correctness and exiting if necessary
	******************************************************************************/


	/******************************************************************************
	check that a CUDA capable device is present
	******************************************************************************/
	int n_devices = 0;

	cudaGetDeviceCount(&n_devices);
	if (cuda_error("cudaGetDeviceCount", false, __FILE__, __LINE__)) return -1;

	if (verbose)
	{
		std::cout << "Available CUDA capable devices:\n\n";

		for (int i = 0; i < n_devices; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			if (cuda_error("cudaGetDeviceProperties", false, __FILE__, __LINE__)) return -1;

			show_device_info(i, prop);
			std::cout << "\n";
		}
	}

	if (n_devices > 1)
	{
		std::cout << "More than one CUDA capable device detected. Defaulting to first device.\n\n";
	}
	cudaSetDevice(0);
	if (cuda_error("cudaSetDevice", false, __FILE__, __LINE__)) return -1;


	/******************************************************************************
	determine mass function, <m>, and <m^2>
	******************************************************************************/
	enumMassFunction mass_function = MASS_FUNCTIONS.at(mass_function_str);
	dtype mean_mass;
	set_param("mean_mass", mean_mass, MassFunction<dtype>(mass_function).mean_mass(m_solar, m_lower, m_upper), verbose);
	dtype mean_mass2;
	set_param("mean_mass2", mean_mass2, MassFunction<dtype>(mass_function).mean_mass2(m_solar, m_lower, m_upper), verbose, true);

	/******************************************************************************
	calculated values for the number of stars, kappa_star, upper and lower mass
	cutoffs, <m>, and <m^2>
	******************************************************************************/
	int num_stars = 0;
	dtype kappa_star_actual = static_cast<dtype>(1);
	dtype m_lower_actual = static_cast<dtype>(1);
	dtype m_upper_actual = static_cast<dtype>(1);
	dtype mean_mass_actual = static_cast<dtype>(1);
	dtype mean_mass2_actual = static_cast<dtype>(1);


	/******************************************************************************
	if star file is specified, check validity of values and set num_stars,
	m_lower_actual, m_upper_actual, mean_mass_actual, and mean_mass2_actual based
	on star information
	******************************************************************************/
	if (starfile != "")
	{
		std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

		if (!read_star_params<dtype>(num_stars, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual, starfile))
		{
			std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n\n";

		m_lower = m_lower_actual;
		m_upper = m_upper_actual;
		mean_mass = mean_mass_actual;
		mean_mass2 = mean_mass2_actual;
	}

	/******************************************************************************
	average magnification of the system
	******************************************************************************/
	dtype mu_ave;
	set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

	/******************************************************************************
	number density of rays in the lens plane
	uses the fact that for a given user specified number density of rays in the
	source plane, further subdivisions are made that multiply the effective number
	of rays in the image plane by 27^2
	******************************************************************************/
	dtype num_rays_lens = num_rays / std::abs(mu_ave) * num_pixels * num_pixels / (2 * half_length * 2 * half_length) * 1 / (27 * 27);

	/******************************************************************************
	average separation between rays in one dimension is 1/sqrt(number density)
	******************************************************************************/
	dtype ray_sep;
	set_param("ray_sep", ray_sep, 1 / std::sqrt(num_rays_lens), verbose);

	/******************************************************************************
	shooting region is greater than outer boundary for macro-mapping by the size of
	the region of images visible for a macro-image which on average loses no more
	than the desired amount of flux
	******************************************************************************/
	dtype lens_hl_x1 = (half_length + theta_e * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss))) / std::abs(1 - kappa_tot + shear);
	dtype lens_hl_x2 = (half_length + theta_e * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss))) / std::abs(1 - kappa_tot - shear);

	/******************************************************************************
	make shooting region a multiple of the ray separation
	******************************************************************************/
	set_param("lens_hl_x1", lens_hl_x1, ray_sep * (static_cast<int>(lens_hl_x1 / ray_sep) + 1), verbose);
	set_param("lens_hl_x2", lens_hl_x2, ray_sep * (static_cast<int>(lens_hl_x2 / ray_sep) + 1), verbose, true);

	/******************************************************************************
	if stars are not drawn from external file, calculate final number of stars to
	use
	******************************************************************************/
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

	Complex<dtype> c;
	set_param("corner", c, 
		std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
		* Complex<dtype>(
			std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
			std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
			), 
		verbose && rectangular, true && !approx);

	int taylor;
	set_param("taylor", taylor, 
		std::max(
			static_cast<int>(std::log(2 * kappa_star * c.abs() / (2 * half_length / num_pixels * PI)) / std::log(safety_scale)),
			1),
		verbose && rectangular && approx, true);

	dtype rad;
	set_param("rad", rad, std::sqrt(theta_e * theta_e * num_stars * mean_mass / kappa_star), verbose && !rectangular);

	int tree_levels;
	set_param("tree_levels", tree_levels, static_cast<int>(std::log2(num_stars / (c.re > c.im ? c.im : c.re ) * (c.re > c.im ? c.re : c.im)) / 2 + 1), verbose);

	int tree_size = 2;
	tree_size = tree_size << (2 * tree_levels + 1);
	tree_size -= 1;
	set_param("tree_size", tree_size, tree_size / 3, verbose);

	int multipole_order;
	set_param("multipole_order", multipole_order, std::log2(7 * m_upper * num_pixels / (2 * half_length)), verbose, true);


	/******************************************************************************
	BEGIN memory allocation
	******************************************************************************/

	std::cout << "Beginning memory allocation...\n";

	curandState* states = nullptr;
	star<dtype>* stars = nullptr;
	star<dtype>* temp_stars = nullptr;
	int* binomial_coeffs = nullptr;
	TreeNode<dtype>* tree = nullptr;
	int* pixels = nullptr;
	int* pixels_minima = nullptr;
	int* pixels_saddles = nullptr;

	/******************************************************************************
	allocate memory for stars
	******************************************************************************/
	cudaMallocManaged(&states, num_stars * sizeof(curandState));
	if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&stars, num_stars * sizeof(star<dtype>));
	if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&temp_stars, num_stars * sizeof(star<dtype>));
	if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for quadtree
	******************************************************************************/
	cudaMallocManaged(&binomial_coeffs, multipole_order * (multipole_order + 3) / 2 * sizeof(int));
	if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&tree, tree_size * sizeof(TreeNode<dtype>));
	if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return -1;

	/******************************************************************************
	allocate memory for pixels
	******************************************************************************/
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

	/******************************************************************************
	END memory allocation
	******************************************************************************/


	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	number of threads per block, and number of blocks per grid
	uses 512 for number of threads in x dimension, as 1024 is the maximum allowable
	number of threads per block but is too large for some memory allocation, and
	512 is next power of 2 smaller
	******************************************************************************/
	set_threads(threads, 512);
	set_blocks(threads, blocks, num_stars);


	/******************************************************************************
	BEGIN populating star array
	******************************************************************************/

	if (starfile == "")
	{
		std::cout << "Generating star field...\n";

		/******************************************************************************
		if random seed was not provided, get one based on the time
		******************************************************************************/
		if (random_seed == 0)
		{
			set_param("random_seed", random_seed, static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()), verbose);
		}

		/******************************************************************************
		generate random star field if no star file has been given
		******************************************************************************/
		initialize_curand_states_kernel<dtype> <<<blocks, threads>>> (states, num_stars, random_seed);
		if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return -1;
		if (rectangular)
		{
			generate_rectangular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, c, mass_function, m_solar, m_lower, m_upper);
		}
		else
		{
			generate_circular_star_field_kernel<dtype> <<<blocks, threads>>> (states, stars, num_stars, rad, mass_function, m_solar, m_lower, m_upper);
		}
		if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return -1;

		std::cout << "Done generating star field.\n\n";

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		and mean_mass2_actual based on star information
		******************************************************************************/
		calculate_star_params<dtype>(stars, num_stars, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual);
		if (rectangular)
		{
			kappa_star_actual = PI * theta_e * theta_e * num_stars * mean_mass_actual / (4 * c.re * c.im);
		}
		else
		{
			kappa_star_actual = theta_e * theta_e * num_stars * mean_mass_actual / (rad * rad);
		}
	}
	else
	{
		/******************************************************************************
		ensure random seed is 0 to denote that stars come from external file
		******************************************************************************/
		set_param("random_seed", random_seed, 0, verbose);

		std::cout << "Reading star field from file " << starfile << "\n";

		/******************************************************************************
		reading star field from external file
		******************************************************************************/
		if (!read_star_file<dtype>(stars, num_stars, starfile))
		{
			std::cerr << "Error. Unable to read star field from file " << starfile << "\n";
			return -1;
		}

		std::cout << "Done reading star field from file " << starfile << "\n\n";
	}

	/******************************************************************************
	END populating star array
	******************************************************************************/


	tree[0] = TreeNode<dtype>(Complex<dtype>(0, 0), c.re > c.im ? c.re : c.im, 0, 0);
	tree[0].numstars = num_stars;
	
	int* max_num_stars_in_level;
	int* min_num_stars_in_level;
	cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
	if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return -1;
	cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
	if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return -1;

	for (int i = 0; i < tree_levels; i++)
	{
		if (verbose)
		{
			std::cout << "Loop " << (i + 1) <<  " /  " << tree_levels << "\n";
		}

		set_threads(threads, 512);
		set_blocks(threads, blocks, get_num_nodes(i));
		create_children_kernel<dtype> <<<blocks, threads>>> (tree, i);
		if (cuda_error("create_tree_kernel", true, __FILE__, __LINE__)) return -1;

		set_threads(threads, 512);
		set_blocks(threads, blocks, 512 * get_num_nodes(i));
		sort_stars_kernel<dtype> <<<blocks, threads>>> (tree, i, stars, temp_stars);
		if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return -1;


		*max_num_stars_in_level = 0;
		*min_num_stars_in_level = num_stars;

		set_threads(threads, 512);
		set_blocks(threads, blocks, get_num_nodes(i));
	
		get_min_max_stars_kernel<dtype> <<<blocks, threads>>> (tree, i + 1, min_num_stars_in_level, max_num_stars_in_level);
		if (cuda_error("get_min_max_stars_kernel", true, __FILE__, __LINE__)) return -1;
		if (*max_num_stars_in_level <= 7)
		{
			std::cout << "Necessary recursion limit reached. Maximum number of stars in a node is " << *max_num_stars_in_level << "\n";
			std::cout << "Minimum number of stars in a node is " << *min_num_stars_in_level << "\n";
			set_param("tree_levels", tree_levels, i + 1, true);
			break;
		}
		else
		{
			std::cout << "Maximum number of stars in a node is " << *max_num_stars_in_level << "\n";
			std::cout << "Minimum number of stars in a node is " << *min_num_stars_in_level << "\n";
		}
	}

	set_threads(threads, multipole_order + 1);
	set_blocks(threads, blocks, (multipole_order + 1) * get_num_nodes(tree_levels));

	if (verbose)
	{
		std::cout << "Calculating multipole coefficients...\n";
	}
	calculate_multipole_coeffs_kernel<dtype> <<<blocks, threads, (multipole_order + 1) * sizeof(Complex<dtype>)>>> (tree, tree_levels, stars, multipole_order);
	if (cuda_error("calculate_multipole_coeffs_kernel", true, __FILE__, __LINE__)) return -1;
	if (verbose)
	{
		std::cout << "Done calculating multipole coefficients.\n\n";
	}

	if (verbose)
	{
		std::cout << "Calculating binomial coefficients...\n";
	}
	calculate_binomial_coeffs(binomial_coeffs, multipole_order);
	if (verbose)
	{
		std::cout << "Done calculating binomial coefficients.\n\n";
	}

	/******************************************************************************
	redefine thread and block size to maximize parallelization
	******************************************************************************/
	set_threads(threads, 16, 16);
	set_blocks(threads, blocks, 2 * lens_hl_x1 / ray_sep, 2 * lens_hl_x2 / ray_sep);

	/******************************************************************************
	initialize pixel values
	******************************************************************************/
	if (verbose)
	{
		std::cout << "Initializing pixel values...\n";
	}
	initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels, num_pixels);
	if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
	if (write_parities)
	{
		initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels_minima, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
		initialize_pixels_kernel<dtype> <<<blocks, threads>>> (pixels_saddles, num_pixels);
		if (cuda_error("initialize_pixels_kernel", true, __FILE__, __LINE__)) return -1;
	}
	if (verbose)
	{
		std::cout << "Done initializing pixel values.\n\n";
	}


	/******************************************************************************
	start and end time for timing purposes
	******************************************************************************/
	std::chrono::high_resolution_clock::time_point t_start;
	std::chrono::high_resolution_clock::time_point t_end;


	/******************************************************************************
	shoot rays and calculate time taken in seconds
	******************************************************************************/
	std::cout << "Shooting rays...\n";
	t_start = std::chrono::high_resolution_clock::now();
	shoot_rays_kernel<dtype> <<<blocks, threads>>> (kappa_tot, shear, theta_e, stars, num_stars, kappa_star,
		rectangular, c, approx, taylor, lens_hl_x1, lens_hl_x2, ray_sep, half_length, pixels_minima, pixels_saddles, pixels, num_pixels);
	if (cuda_error("shoot_rays_kernel", true, __FILE__, __LINE__)) return -1;
	t_end = std::chrono::high_resolution_clock::now();
	double t_ray_shoot = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
	std::cout << "Done shooting rays. Elapsed time: " << t_ray_shoot << " seconds.\n\n";


	/******************************************************************************
	create histograms of pixel values
	******************************************************************************/

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

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

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

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 512);
		set_blocks(threads, blocks, histogram_length);

		initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram, histogram_length);
		if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
		if (write_parities)
		{
			initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram_minima, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
			initialize_histogram_kernel<dtype> <<<blocks, threads>>> (histogram_saddles, histogram_length);
			if (cuda_error("initialize_histogram_kernel", true, __FILE__, __LINE__)) return -1;
		}

		/******************************************************************************
		redefine thread and block size to maximize parallelization
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

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
	/******************************************************************************
	done creating histograms of pixel values
	******************************************************************************/


	/******************************************************************************
	stream for writing output files
	set precision to 9 digits
	******************************************************************************/
	std::ofstream outfile;
	outfile.precision(9);
	std::string fname;


	std::cout << "Writing parameter info...\n";
	fname = outfile_prefix + "irs_parameter_info.txt";
	outfile.open(fname);
	if (!outfile.is_open())
	{
		std::cerr << "Error. Failed to open file " << fname << "\n";
		return -1;
	}
	outfile << "kappa_tot " << kappa_tot << "\n";
	outfile << "shear " << shear << "\n";
	outfile << "mu_ave " << mu_ave << "\n";
	outfile << "smooth_fraction " << smooth_fraction << "\n";
	outfile << "kappa_star " << kappa_star << "\n";
	if (starfile == "")
	{
		outfile << "kappa_star_actual " << kappa_star_actual << "\n";
	}
	outfile << "theta_e " << theta_e << "\n";
	if (starfile == "")
	{
		outfile << "mass_function " << mass_function_str << "\n";
		if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
		{
			outfile << "m_solar " << m_solar << "\n";
		}
		if (mass_function_str != "equal")
		{
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "m_lower_actual " << m_lower_actual << "\n";
			outfile << "m_upper_actual " << m_upper_actual << "\n";
		}
		outfile << "mean_mass " << mean_mass << "\n";
		outfile << "mean_mass2 " << mean_mass2 << "\n";
		if (mass_function_str != "equal")
		{
			outfile << "mean_mass_actual " << mean_mass_actual << "\n";
			outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		}
	}
	else
	{
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
	}
	outfile << "light_loss " << light_loss << "\n";
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


	std::cout << "Writing star info...\n";
	fname = outfile_prefix + "irs_stars" + outfile_type;
	if (!write_star_file<dtype>(stars, num_stars, fname))
	{
		std::cerr << "Error. Unable to write star info to file " << fname << "\n";
		return -1;
	}
	std::cout << "Done writing star info to file " << fname << "\n\n";


	/******************************************************************************
	histograms of magnification maps
	******************************************************************************/
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


	/******************************************************************************
	write magnifications for minima, saddle, and combined maps
	******************************************************************************/
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

