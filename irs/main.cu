/******************************************************************************

Please provide credit to Luke Weisenbach should this code be used.
Email: weisluke@alum.mit.edu

******************************************************************************/


#include "microlensing.cuh"
#include "util.hpp"

#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>


using dtype = float; //type to be used throughout this program. int, float, or double
Microlensing<dtype> microlensing;

/******************************************************************************
constants to be used
******************************************************************************/
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

/******************************************************************************
default input option values
******************************************************************************/
bool verbose = false;



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
		<< "  -k,--kappa_tot          Specify the total convergence. Default value: " << microlensing.kappa_tot << "\n"
		<< "  -y,--shear              Specify the external shear. Default value: " << microlensing.shear << "\n"
		<< "  -s,--smooth_fraction    Specify the fraction of convergence due to smoothly\n"
		<< "                          distributed mass. Default value: " << microlensing.smooth_fraction << "\n"
		<< "  -ks,--kappa_star        Specify the convergence in point mass lenses. If\n"
		<< "                          provided, this overrides any supplied value for the\n"
		<< "                          smooth fraction. Default value: " << microlensing.kappa_star << "\n"
		<< "  -t,--theta_e            Specify the size of the Einstein radius of a unit\n"
		<< "                          mass point lens in arbitrary units. Default value: " << microlensing.theta_e << "\n"
		<< "  -mf,--mass_function     Specify the mass function to use for the point mass\n"
		<< "                          lenses. Options are: equal, uniform, Salpeter, and\n"
		<< "                          Kroupa. Default value: " << microlensing.mass_function_str << "\n"
		<< "  -ms,--m_solar           Specify the solar mass in arbitrary units.\n"
		<< "                          Default value: " << microlensing.m_solar << "\n"
		<< "  -ml,--m_lower           Specify the lower mass cutoff in arbitrary units.\n"
		<< "                          Default value: " << microlensing.m_lower << "\n"
		<< "  -mh,--m_upper           Specify the upper mass cutoff in arbitrary units.\n"
		<< "                          Default value: " << microlensing.m_upper << "\n"
		<< "  -ll,--light_loss        Allowed average fraction of light lost due to scatter\n"
		<< "                          by the microlenses in the large deflection limit.\n"
		<< "                          Default value: " << microlensing.light_loss << "\n"
		<< "  -r,--rectangular        Specify whether the star field should be\n"
		<< "                          rectangular (1) or circular (0). Default value: " << microlensing.rectangular << "\n"
		<< "  -a,--approx             Specify whether terms for alpha_smooth should be\n"
		<< "                          approximated (1) or exact (0). Default value: " << microlensing.approx << "\n"
		<< "  -ss,--safety_scale      Specify the multiplicative safety factor over the\n"
		<< "                          shooting region to be used when generating the star\n"
		<< "                          field. Default value: " << microlensing.safety_scale << "\n"
		<< "  -sf,--starfile          Specify the location of a binary file containing\n"
		<< "                          values for num_stars, rectangular, corner, theta_e,\n"
		<< "                          and the star positions and masses, in an order as\n"
		<< "                          defined in this source code.\n"
		<< "  -hl,--half_length       Specify the half-length of the square source plane\n"
		<< "                          region to find the magnification in.\n"
		<< "                          Default value: " << microlensing.half_length_source << "\n"
		<< "  -px,--pixels            Specify the number of pixels per side for the\n"
		<< "                          magnification map. Default value: " << microlensing.num_pixels << "\n"
		<< "  -nr,--num_rays          Specify the average number of rays per pixel.\n"
		<< "                          Default value: " << microlensing.num_rays_source << "\n"
		<< "  -rs,--random_seed       Specify the random seed for star field generation.\n"
		<< "                          A value of 0 is reserved for star input files.\n"
		<< "  -wm,--write_maps        Specify whether to write magnification maps (1) or\n"
		<< "                          not (0). Default value: " << microlensing.write_maps << "\n"
		<< "  -wp,--write_parities    Specify whether to write parity specific\n"
		<< "                          magnification maps (1) or not (0). Default value: " << microlensing.write_parities << "\n"
		<< "  -wh,--write_histograms  Specify whether to write histograms (1) or not (0).\n"
		<< "                          Default value: " << microlensing.write_histograms << "\n"
		<< "  -ot,--outfile_type      Specify the type of file to be output. Valid options\n"
		<< "                          are binary (.bin). Default value: " << microlensing.outfile_type << "\n"
		<< "  -o,--outfile_prefix     Specify the prefix to be used in output file names.\n"
		<< "                          Default value: " << microlensing.outfile_prefix << "\n";
}



int main(int argc, char* argv[])
{
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
				set_param("kappa_tot", microlensing.kappa_tot, std::stod(cmdinput), verbose);
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
				set_param("shear", microlensing.shear, std::stod(cmdinput), verbose);
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
				set_param("smooth_fraction", microlensing.smooth_fraction, std::stod(cmdinput), verbose);
				if (microlensing.smooth_fraction < 0)
				{
					std::cerr << "Error. Invalid smooth_fraction input. smooth_fraction must be >= 0\n";
					return -1;
				}
				else if (microlensing.smooth_fraction >= 1)
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
				set_param("kappa_star", microlensing.kappa_star, std::stod(cmdinput), verbose);
				if (microlensing.kappa_star < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid kappa_star input. kappa_star must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("theta_e", microlensing.theta_e, std::stod(cmdinput), verbose);
				if (microlensing.theta_e < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid theta_e input. theta_e must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
			set_param("mass_function", microlensing.mass_function_str, make_lowercase(cmdinput), verbose);
			if (!massfunctions::MASS_FUNCTIONS.count(microlensing.mass_function_str))
			{
				std::cerr << "Error. Invalid mass_function input. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-ms") || argv[i] == std::string("--m_solar"))
		{
			try
			{
				set_param("m_solar", microlensing.m_solar, std::stod(cmdinput), verbose);
				if (microlensing.m_solar < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_solar input. m_solar must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("m_lower", microlensing.m_lower, std::stod(cmdinput), verbose);
				if (microlensing.m_lower < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_lower input. m_lower must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("m_upper", microlensing.m_upper, std::stod(cmdinput), verbose);
				if (microlensing.m_upper < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be >= " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (microlensing.m_upper > std::numeric_limits<dtype>::max())
				{
					std::cerr << "Error. Invalid m_upper input. m_upper must be <= " << std::numeric_limits<dtype>::max() << "\n";
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
				set_param("light_loss", microlensing.light_loss, std::stod(cmdinput), verbose);
				if (microlensing.light_loss < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid light_loss input. light_loss must be >= " << std::numeric_limits<dtype>::min() << "\n";
					return -1;
				}
				else if (microlensing.light_loss > 0.01)
				{
					std::cerr << "Error. Invalid light_loss input. light_loss must be <= 0.01\n";
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
				set_param("rectangular", microlensing.rectangular, std::stoi(cmdinput), verbose);
				if (microlensing.rectangular != 0 && microlensing.rectangular != 1)
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
				set_param("approx", microlensing.approx, std::stoi(cmdinput), verbose);
				if (microlensing.approx != 0 && microlensing.approx != 1)
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
				set_param("safety_scale", microlensing.safety_scale, std::stod(cmdinput), verbose);
				if (microlensing.safety_scale < 1.1)
				{
					std::cerr << "Error. Invalid safety_scale input. safety_scale must be >= 1.1\n";
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
			set_param("starfile", microlensing.starfile, cmdinput, verbose);
		}
		else if (argv[i] == std::string("-hl") || argv[i] == std::string("--half_length"))
		{
			try
			{
				set_param("half_length", microlensing.half_length_source, std::stod(cmdinput), verbose);
				if (microlensing.half_length_source < std::numeric_limits<dtype>::min())
				{
					std::cerr << "Error. Invalid half_length input. half_length must be >= " << std::numeric_limits<dtype>::min() << "\n";
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
				set_param("num_pixels", microlensing.num_pixels, std::stoi(cmdinput), verbose);
				if (microlensing.num_pixels < 1)
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
				set_param("num_rays", microlensing.num_rays_source, std::stoi(cmdinput), verbose);
				if (microlensing.num_rays_source < 1)
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
				set_param("random_seed", microlensing.random_seed, std::stoi(cmdinput), verbose);
				if (microlensing.random_seed == 0 && 
					!(cmd_option_exists(argv, argv + argc, "-sf") || cmd_option_exists(argv, argv + argc, "--star_file")))
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
				set_param("write_maps", microlensing.write_maps, std::stoi(cmdinput), verbose);
				if (microlensing.write_maps != 0 && microlensing.write_maps != 1)
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
				set_param("write_parities", microlensing.write_parities, std::stoi(cmdinput), verbose);
				if (microlensing.write_parities != 0 && microlensing.write_parities != 1)
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
				set_param("write_histograms", microlensing.write_histograms, std::stoi(cmdinput), verbose);
				if (microlensing.write_histograms != 0 && microlensing.write_histograms != 1)
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
			set_param("outfile_type", microlensing.outfile_type, make_lowercase(cmdinput), verbose);
			if (microlensing.outfile_type != ".bin")
			{
				std::cerr << "Error. Invalid outfile_type. outfile_type must be .bin\n";
				return -1;
			}
		}
		else if (argv[i] == std::string("-o") || argv[i] == std::string("--outfile_prefix"))
		{
			set_param("outfile_prefix", microlensing.outfile_prefix, cmdinput, verbose);
		}
	}

	if (cmd_option_exists(argv, argv + argc, "-ks") || cmd_option_exists(argv, argv + argc, "--kappa_star"))
	{
		set_param("smooth_fraction", microlensing.smooth_fraction, 1 - microlensing.kappa_star / microlensing.kappa_tot, verbose);
	}
	else
	{
		set_param("kappa_star", microlensing.kappa_star, (1 - microlensing.smooth_fraction) * microlensing.kappa_tot, verbose);
	}

	if (microlensing.mass_function_str == "equal")
	{
		set_param("m_lower", microlensing.m_lower, 1, verbose);
		set_param("m_upper", microlensing.m_upper, 1, verbose);
	}
	else if (microlensing.m_lower > microlensing.m_upper)
	{
		std::cerr << "Error. m_lower must be <= m_upper.\n";
		return -1;
	}

	std::cout << "\n";

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
	run microlensing and save files
	******************************************************************************/
	if (!microlensing.run(verbose)) return -1;
	if (!microlensing.save(verbose)) return -1;


	std::cout << "Done.\n";

	cudaDeviceReset();
	if (cuda_error("cudaDeviceReset", false, __FILE__, __LINE__)) return -1;

	return 0;
}

