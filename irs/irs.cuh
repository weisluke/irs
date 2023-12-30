#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "fmm.cuh"
#include "irs_functions.cuh"
#include "mass_function.cuh"
#include "star.cuh"
#include "stopwatch.hpp"
#include "tree_node.cuh"
#include "util.cuh"

#include <curand_kernel.h>

#include <algorithm> //for std::min and std::max
#include <chrono> //for setting random seed with clock
#include <cmath>
#include <fstream> //for writing files
#include <iostream>
#include <limits> //for std::numeric_limits
#include <string>
#include <vector>


template <typename T>
class IRS
{

public:
	/******************************************************************************
	default input variables
	******************************************************************************/
	T kappa_tot = static_cast<T>(0.3);
	T shear = static_cast<T>(0.3);
	T kappa_star = static_cast<T>(0.27);
	T theta_e = static_cast<T>(1);
	std::string mass_function_str = "equal";
	T m_solar = static_cast<T>(1);
	T m_lower = static_cast<T>(0.01);
	T m_upper = static_cast<T>(50);
	T light_loss = static_cast<T>(0.01);
	int rectangular = 1;
	int approx = 0;
	T safety_scale = static_cast<T>(1.37);
	std::string starfile = "";
	Complex<T> center_y = Complex<T>();
	T half_length_source = static_cast<T>(5);
	int num_pixels = 1000;
	int num_rays_source = 1000;
	int random_seed = 0;
	int write_maps = 1;
	int write_parities = 0;
	int write_histograms = 1;
	std::string outfile_prefix = "./";


	/******************************************************************************
	class initializer is empty
	******************************************************************************/
	IRS()
	{

	}


private:
	/******************************************************************************
	constant variables
	******************************************************************************/
	const T PI = static_cast<T>(3.1415926535898);
	const T E = static_cast<T>(2.718281828459);
	const std::string outfile_type = ".bin";

	/******************************************************************************
	variables for kernel threads and blocks
	******************************************************************************/
	dim3 threads;
	dim3 blocks;

	/******************************************************************************
	stopwatch for timing purposes
	******************************************************************************/
	Stopwatch stopwatch;
	double t_elapsed;
	double t_ray_shoot;

	/******************************************************************************
	derived variables
	******************************************************************************/
	massfunctions::massfunction mass_function;
	T mean_mass;
	T mean_mass2;

	int num_stars;
	T kappa_star_actual;
	T m_lower_actual;
	T m_upper_actual;
	T mean_mass_actual;
	T mean_mass2_actual;

	T mu_ave;
	T num_rays_lens;
	T ray_sep;
	Complex<T> center_x;
	Complex<int> num_ray_blocks;
	Complex<T> half_length_image;
	Complex<T> corner;
	int taylor_smooth;

	T error; //error in the ray position

	int expansion_order;

	T root_half_length;
	int rays_level; //ray_sep * 2 ^ rays_level = root_half_length
	int tree_levels = 0;
	std::vector<TreeNode<T>*> tree = {};
	std::vector<int> num_nodes = {};
	int ray_blocks_level = 0;

	/******************************************************************************
	dynamic memory
	******************************************************************************/
	curandState* states = nullptr;
	star<T>* stars = nullptr;
	star<T>* temp_stars = nullptr;

	int* binomial_coeffs = nullptr;

	int* pixels = nullptr;
	int* pixels_minima = nullptr;
	int* pixels_saddles = nullptr;

	int* min_rays = nullptr;
	int* max_rays = nullptr;
	int histogram_length = 0;
	int* histogram = nullptr;
	int* histogram_minima = nullptr;
	int* histogram_saddles = nullptr;



	bool check_input_params(bool verbose)
	{
		std::cout << "Checking input parameters...\n";


		if (kappa_tot < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_tot must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (kappa_star < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. kappa_star must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		if (kappa_star > kappa_tot)
		{
			std::cerr << "Error. kappa_star must be <= kappa_tot\n";
			return false;
		}

		if (theta_e < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. theta_e must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (!massfunctions::MASS_FUNCTIONS.count(mass_function_str))
		{
			std::cerr << "Error. mass_function must be equal, uniform, Salpeter, or Kroupa.\n";
			return false;
		}

		if (m_solar < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_solar must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (m_lower < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. m_lower must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (m_upper < m_lower)
		{
			std::cerr << "Error. m_upper must be >= m_lower.\n";
			return false;
		}

		if (light_loss < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. light_loss must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}
		else if (light_loss > 0.01)
		{
			std::cerr << "Error. light_loss must be <= 0.01\n";
			return false;
		}

		if (rectangular != 0 && rectangular != 1)
		{
			std::cerr << "Error. rectangular must be 1 (rectangular) or 0 (circular).\n";
			return false;
		}

		if (approx != 0 && approx != 1)
		{
			std::cerr << "Error. approx must be 1 (approximate) or 0 (exact).\n";
			return false;
		}
		
		if (safety_scale < 1.1)
		{
			std::cerr << "Error. safety_scale must be >= 1.1\n";
			return false;
		}

		if (half_length_source < std::numeric_limits<T>::min())
		{
			std::cerr << "Error. half_length must be >= " << std::numeric_limits<T>::min() << "\n";
			return false;
		}

		if (num_pixels < 1)
		{
			std::cerr << "Error. num_pixels must be an integer > 0\n";
			return false;
		}

		if (num_rays_source < 1)
		{
			std::cerr << "Error. num_rays must be an integer > 0\n";
			return false;
		}

		if (write_maps != 0 && write_maps != 1)
		{
			std::cerr << "Error. write_maps must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_parities != 0 && write_parities != 1)
		{
			std::cerr << "Error. write_parities must be 1 (true) or 0 (false).\n";
			return false;
		}

		if (write_histograms != 0 && write_histograms != 1)
		{
			std::cerr << "Error. write_histograms must be 1 (true) or 0 (false).\n";
			return false;
		}


		std::cout << "Done checking input parameters.\n\n";
		
		return true;		
	}
	
	bool calculate_derived_params(bool verbose)
	{
		std::cout << "Calculating derived parameters...\n";
		stopwatch.start();

		/******************************************************************************
		if star file is not specified, set the mass function, mean_mass, and
		mean_mass2
		******************************************************************************/
		if (starfile == "")
		{
			if (mass_function_str == "equal")
			{
				set_param("m_lower", m_lower, 1, verbose);
				set_param("m_upper", m_upper, 1, verbose);
			}

			/******************************************************************************
			determine mass function, <m>, and <m^2>
			******************************************************************************/
			mass_function = massfunctions::MASS_FUNCTIONS.at(mass_function_str);
			set_param("mean_mass", mean_mass, MassFunction<T>(mass_function).mean_mass(m_solar, m_lower, m_upper), verbose);
			set_param("mean_mass2", mean_mass2, MassFunction<T>(mass_function).mean_mass2(m_solar, m_lower, m_upper), verbose);
		}
		/******************************************************************************
		if star file is specified, check validity of values and set num_stars,
		rectangular, corner, theta_e, stars, kappa_star, m_lower, m_upper, mean_mass,
		and mean_mass2 based on star information
		******************************************************************************/
		else 
		{
			std::cout << "Calculating some parameter values based on star input file " << starfile << "\n";

			if (!read_star_file<T>(num_stars, rectangular, corner, theta_e, stars, 
				kappa_star, m_lower, m_upper, mean_mass, mean_mass2, starfile))
			{
				std::cerr << "Error. Unable to read star field parameters from file " << starfile << "\n";
				return false;
			}

			set_param("num_stars", num_stars, num_stars, verbose);
			set_param("rectangular", rectangular, rectangular, verbose);
			set_param("corner", corner, corner, verbose);
			set_param("theta_e", theta_e, theta_e, verbose);
			set_param("kappa_star", kappa_star, kappa_star, verbose);
			set_param("m_lower", m_lower, m_lower, verbose);
			set_param("m_upper", m_upper, m_upper, verbose);
			set_param("mean_mass", mean_mass, mean_mass, verbose);
			set_param("mean_mass2", mean_mass2, mean_mass2, verbose);

			std::cout << "Done calculating some parameter values based on star input file " << starfile << "\n";
		}

		/******************************************************************************
		average magnification of the system
		******************************************************************************/
		set_param("mu_ave", mu_ave, 1 / ((1 - kappa_tot) * (1 - kappa_tot) - shear * shear), verbose);

		/******************************************************************************
		number density of rays in the lens plane
		******************************************************************************/
		set_param("num_rays_lens", num_rays_lens, 
			num_rays_source / std::abs(mu_ave) * num_pixels * num_pixels / (2 * half_length_source * 2 * half_length_source),
			verbose);
		
		/******************************************************************************
		average separation between rays in one dimension is 1/sqrt(number density)
		******************************************************************************/
		set_param("ray_sep", ray_sep, 1 / std::sqrt(num_rays_lens), verbose);

		/******************************************************************************
		shooting region is greater than outer boundary for macro-mapping by the size of
		the region of images visible for a macro-image which on average loses no more
		than the desired amount of flux
		******************************************************************************/
		half_length_image = (half_length_source + theta_e * std::sqrt(kappa_star * mean_mass2 / (mean_mass * light_loss))) 
			* Complex<T>(
				1 / std::abs(1 - kappa_tot + shear),
				1 / std::abs(1 - kappa_tot - shear)
			);
		set_param("half_length_image", half_length_image, half_length_image, verbose);

		center_x = Complex<T>(center_y.re / (1 - kappa_tot + shear), center_y.im / (1 - kappa_tot - shear));
		set_param("center_x", center_x, center_x, verbose);

		/******************************************************************************
		if stars are not drawn from external file, calculate final number of stars to
		use and corner of the star field
		******************************************************************************/
		if (starfile == "")
		{
			if (rectangular)
			{
				num_stars = static_cast<int>((safety_scale * 2 * (std::abs(center_x.re) + half_length_image.re)) * (safety_scale * 2 * (std::abs(center_x.im) + half_length_image.im)) 
					* kappa_star / (PI * theta_e * theta_e * mean_mass)) + 1;
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = std::sqrt(PI * theta_e * theta_e * num_stars * mean_mass / (4 * kappa_star))
					* Complex<T>(
						std::sqrt(std::abs((1 - kappa_tot - shear) / (1 - kappa_tot + shear))),
						std::sqrt(std::abs((1 - kappa_tot + shear) / (1 - kappa_tot - shear)))
					);
				set_param("corner", corner, corner, verbose);
			}
			else
			{
				num_stars = static_cast<int>(safety_scale * safety_scale * (center_x + half_length_image).abs() * (center_x + half_length_image).abs()
					* kappa_star / (theta_e * theta_e * mean_mass)) + 1;
				set_param("num_stars", num_stars, num_stars, verbose);

				corner = std::sqrt(theta_e * theta_e * num_stars * mean_mass / (kappa_star * 2 * ((1 - kappa_tot) * (1 - kappa_tot) + shear * shear)))
					* Complex<T>(
						std::abs(1 - kappa_tot - shear),
						std::abs(1 - kappa_tot + shear)
					);
				set_param("corner", corner, corner, verbose);
			}
		}

		error = 2 * half_length_source / (10 * num_pixels); //error is 1/10 of a pixel

		taylor_smooth = std::max(
			static_cast<int>(std::log(2 * kappa_star * corner.abs() / (error * PI)) / std::log(safety_scale)),
			1
		);
		set_param("taylor_smooth", taylor_smooth, taylor_smooth, verbose && rectangular && approx);

		t_elapsed = stopwatch.stop();
		std::cout << "Done calculating derived parameters. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool allocate_initialize_memory(bool verbose)
	{
		std::cout << "Allocating memory...\n";
		stopwatch.start();

		/******************************************************************************
		allocate memory for stars
		******************************************************************************/
		cudaMallocManaged(&states, num_stars * sizeof(curandState));
		if (cuda_error("cudaMallocManaged(*states)", false, __FILE__, __LINE__)) return false;
		if (stars == nullptr) //if memory wasn't allocated already due to reading a star file
		{
			cudaMallocManaged(&stars, num_stars * sizeof(star<T>));
			if (cuda_error("cudaMallocManaged(*stars)", false, __FILE__, __LINE__)) return false;
		}
		cudaMallocManaged(&temp_stars, num_stars * sizeof(star<T>));
		if (cuda_error("cudaMallocManaged(*temp_stars)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for binomial coefficients
		******************************************************************************/
		cudaMallocManaged(&binomial_coeffs, (2 * treenode::MAX_EXPANSION_ORDER * (2 * treenode::MAX_EXPANSION_ORDER + 3) / 2 + 1) * sizeof(int));
		if (cuda_error("cudaMallocManaged(*binomial_coeffs)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		allocate memory for pixels
		******************************************************************************/
		cudaMallocManaged(&pixels, num_pixels * num_pixels * sizeof(int));
		if (cuda_error("cudaMallocManaged(*pixels)", false, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			cudaMallocManaged(&pixels_minima, num_pixels * num_pixels * sizeof(int));
			if (cuda_error("cudaMallocManaged(*pixels_minima)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&pixels_saddles, num_pixels * num_pixels * sizeof(int));
			if (cuda_error("cudaMallocManaged(*pixels_saddles)", false, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done allocating memory. Elapsed time: " << t_elapsed << " seconds.\n\n";


		/******************************************************************************
		initialize pixel values
		******************************************************************************/
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, num_pixels, num_pixels);

		std::cout << "Initializing pixel values...\n";
		stopwatch.start();

		initialize_array_kernel<T> <<<blocks, threads>>> (pixels, num_pixels, num_pixels);
		if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
		if (write_parities)
		{
			initialize_array_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels, num_pixels);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			initialize_array_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels, num_pixels);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
		}

		t_elapsed = stopwatch.stop();
		std::cout << "Done initializing pixel values. Elapsed time: " << t_elapsed << " seconds.\n\n";

		return true;
	}

	bool populate_star_array(bool verbose) 
	{
		/******************************************************************************
		BEGIN populating star array
		******************************************************************************/

		set_threads(threads, 512);
		set_blocks(threads, blocks, num_stars);

		if (starfile == "")
		{
			std::cout << "Generating star field...\n";
			stopwatch.start();

			/******************************************************************************
			if random seed was not provided, get one based on the time
			******************************************************************************/
			while (random_seed == 0)
			{
				set_param("random_seed", random_seed, static_cast<int>(std::chrono::system_clock::now().time_since_epoch().count()), verbose);
			}

			/******************************************************************************
			generate random star field if no star file has been given
			******************************************************************************/
			initialize_curand_states_kernel<T> <<<blocks, threads>>> (states, num_stars, random_seed);
			if (cuda_error("initialize_curand_states_kernel", true, __FILE__, __LINE__)) return false;
			generate_star_field_kernel<T> <<<blocks, threads>>> (states, stars, num_stars, rectangular, corner, mass_function, m_solar, m_lower, m_upper);
			if (cuda_error("generate_star_field_kernel", true, __FILE__, __LINE__)) return false;

			t_elapsed = stopwatch.stop();
			std::cout << "Done generating star field. Elapsed time: " << t_elapsed << " seconds.\n\n";
		}
		else
		{
			/******************************************************************************
			ensure random seed is 0 to denote that stars come from external file
			******************************************************************************/
			set_param("random_seed", random_seed, 0, verbose);
		}

		/******************************************************************************
		calculate kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual,
		and mean_mass2_actual based on star information
		******************************************************************************/
		calculate_star_params<T>(num_stars, rectangular, corner, theta_e, stars,
			kappa_star_actual, m_lower_actual, m_upper_actual, mean_mass_actual, mean_mass2_actual);

		set_param("kappa_star_actual", kappa_star_actual, kappa_star_actual, verbose);
		set_param("m_lower_actual", m_lower_actual, m_lower_actual, verbose);
		set_param("m_upper_actual", m_upper_actual, m_upper_actual, verbose);
		set_param("mean_mass_actual", mean_mass_actual, mean_mass_actual, verbose);
		set_param("mean_mass2_actual", mean_mass2_actual, mean_mass2_actual, verbose, true);

		/******************************************************************************
		END populating star array
		******************************************************************************/

		return true;
	}

	bool create_tree(bool verbose)
	{
		/******************************************************************************
		BEGIN create root node, then create children and sort stars
		******************************************************************************/

		if (rectangular)
		{
			root_half_length = corner.re > corner.im ? corner.re : corner.im;
		}
		else
		{
			root_half_length = corner.abs();
		}
		/******************************************************************************
		upscale root half length so it is a power of 2 multiple of the ray separation
		******************************************************************************/
		set_param("rays_level", rays_level, static_cast<int>(std::log2(root_half_length) - std::log2(ray_sep)) + 1, verbose);
		set_param("root_half_length", root_half_length, ray_sep * (1 << rays_level), verbose);

		/******************************************************************************
		push empty pointer into tree, add 1 to number of nodes, and allocate memory
		******************************************************************************/
		tree.push_back(nullptr);
		num_nodes.push_back(1);
		cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
		if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

		/******************************************************************************
		initialize root node
		******************************************************************************/
		tree[0][0] = TreeNode<T>(Complex<T>(0, 0), root_half_length, 0);
		tree[0][0].numstars = num_stars;


		int* max_num_stars_in_level;
		int* min_num_stars_in_level;
		int* num_nonempty_nodes;
		cudaMallocManaged(&max_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*max_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&min_num_stars_in_level, sizeof(int));
		if (cuda_error("cudaMallocManaged(*min_num_stars_in_level)", false, __FILE__, __LINE__)) return false;
		cudaMallocManaged(&num_nonempty_nodes, sizeof(int));
		if (cuda_error("cudaMallocManaged(*num_nonempty_nodes)", false, __FILE__, __LINE__)) return false;

		std::cout << "Creating children and sorting stars...\n";
		stopwatch.start();
		set_threads(threads, 512);
		do
		{
			print_verbose("\nProcessing level " + std::to_string(tree_levels) + "\n", verbose);

			*max_num_stars_in_level = 0;
			*min_num_stars_in_level = num_stars;
			*num_nonempty_nodes = 0;

			set_blocks(threads, blocks, num_nodes[tree_levels]);
			treenode::get_node_star_info_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels],
				num_nonempty_nodes, min_num_stars_in_level, max_num_stars_in_level);
			if (cuda_error("get_node_star_info_kernel", true, __FILE__, __LINE__)) return false;

			print_verbose("Maximum number of stars in a node and its neighbors is " + std::to_string(*max_num_stars_in_level) + "\n", verbose);
			print_verbose("Minimum number of stars in a node and its neighbors is " + std::to_string(*min_num_stars_in_level) + "\n", verbose);

			if (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT)
			{
				print_verbose("Number of non-empty children: " + std::to_string(*num_nonempty_nodes * 4) + "\n", verbose);

				print_verbose("Allocating memory for children...\n", verbose);
				tree.push_back(nullptr);
				num_nodes.push_back(*num_nonempty_nodes * 4);
				cudaMallocManaged(&tree.back(), num_nodes.back() * sizeof(TreeNode<T>));
				if (cuda_error("cudaMallocManaged(*tree)", false, __FILE__, __LINE__)) return false;

				print_verbose("Creating children...\n", verbose);
				(*num_nonempty_nodes)--; //subtract one since value is size of array, and instead needs to be the first allocatable element
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::create_children_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], num_nonempty_nodes, tree[tree_levels + 1]);
				if (cuda_error("create_children_kernel", true, __FILE__, __LINE__)) return false;

				print_verbose("Sorting stars...\n", verbose);
				set_blocks(threads, blocks, 512 * num_nodes[tree_levels]);
				treenode::sort_stars_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels], stars, temp_stars);
				if (cuda_error("sort_stars_kernel", true, __FILE__, __LINE__)) return false;

				tree_levels++;

				print_verbose("Setting neighbors...\n", verbose);
				set_blocks(threads, blocks, num_nodes[tree_levels]);
				treenode::set_neighbors_kernel<T> <<<blocks, threads>>> (tree[tree_levels], num_nodes[tree_levels]);
				if (cuda_error("set_neighbors_kernel", true, __FILE__, __LINE__)) return false;
			}
		} while (*max_num_stars_in_level > treenode::MAX_NUM_STARS_DIRECT);
		set_param("tree_levels", tree_levels, tree_levels, verbose);

		t_elapsed = stopwatch.stop();
		std::cout << "Done creating children and sorting stars. Elapsed time: " << t_elapsed << " seconds.\n\n";


		/******************************************************************************
		make shooting region a multiple of the lowest level node length, or a multiple
		of a factor of two smaller version of the root length that doesn't exceed the
		size of the corner
		******************************************************************************/
		ray_blocks_level = tree_levels;
		num_ray_blocks = Complex<int>(half_length_image / (2 * root_half_length) * (1 << ray_blocks_level)) + Complex<int>(1, 1);
		Complex<T> tmp_half_length_image = Complex<T>(2 * root_half_length / (1 << ray_blocks_level)) * num_ray_blocks;
		while (tmp_half_length_image.re > corner.re || tmp_half_length_image.im > corner.im)
		{
			ray_blocks_level++;
			num_ray_blocks = Complex<int>(half_length_image / (2 * root_half_length) * (1 << ray_blocks_level)) + Complex<int>(1, 1);
			tmp_half_length_image = Complex<T>(2 * root_half_length / (1 << ray_blocks_level)) * num_ray_blocks;
		}
		set_param("ray_blocks_level", ray_blocks_level, ray_blocks_level, verbose);
		if (ray_blocks_level > rays_level)
		{
			std::cerr << "Error. ray_blocks_level > rays_level\n";
			return false;
		}
		set_param("half_length_image", half_length_image, tmp_half_length_image, verbose);
		set_param("num_ray_blocks", num_ray_blocks, 2 * num_ray_blocks, verbose);

		/******************************************************************************
		END create root node, then create children and sort stars
		******************************************************************************/

		expansion_order = static_cast<int>(2 * std::log2(theta_e) - std::log2(root_half_length * error))
			 + tree_levels + 1;
		set_param("expansion_order", expansion_order, expansion_order, verbose, true);
		if (expansion_order < 3)
		{
			std::cerr << "Error. Expansion order needs to be >= 3\n";
			return false;
		}
		else if (expansion_order > treenode::MAX_EXPANSION_ORDER)
		{
			std::cerr << "Error. Maximum allowed expansion order is " << treenode::MAX_EXPANSION_ORDER << "\n";
			return false;
		}

		print_verbose("Calculating binomial coefficients...\n", verbose);
		calculate_binomial_coeffs(binomial_coeffs, 2 * expansion_order);
		print_verbose("Done calculating binomial coefficients.\n\n", verbose);


		/******************************************************************************
		BEGIN calculating multipole and local coefficients
		******************************************************************************/

		std::cout << "Calculating multipole and local coefficients...\n";
		stopwatch.start();

		set_threads(threads, 16, expansion_order + 1);
		set_blocks(threads, blocks, num_nodes[tree_levels], (expansion_order + 1));
		fmm::calculate_multipole_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[tree_levels], num_nodes[tree_levels], expansion_order, stars);

		set_threads(threads, 4, expansion_order + 1, 4);
		for (int i = tree_levels - 1; i >= 0; i--)
		{
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), 4);
			fmm::calculate_M2M_coeffs_kernel<T> <<<blocks, threads, 4 * 4 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}

		/******************************************************************************
		local coefficients are non zero only starting at the second level
		******************************************************************************/
		for (int i = 2; i <= tree_levels; i++)
		{
			set_threads(threads, 16, expansion_order + 1);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1));
			fmm::calculate_L2L_coeffs_kernel<T> <<<blocks, threads, 16 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);

			set_threads(threads, 1, expansion_order + 1, 27);
			set_blocks(threads, blocks, num_nodes[i], (expansion_order + 1), 27);
			fmm::calculate_M2L_coeffs_kernel<T> <<<blocks, threads, 1 * 27 * (expansion_order + 1) * sizeof(Complex<T>)>>> (tree[i], num_nodes[i], expansion_order, binomial_coeffs);
		}
		if (cuda_error("calculate_coeffs_kernels", true, __FILE__, __LINE__)) return false;

		t_elapsed = stopwatch.stop();
		std::cout << "Done calculating multipole and local coefficients. Elapsed time: " << t_elapsed << " seconds.\n\n";
		
		/******************************************************************************
		END calculating multipole and local coefficients
		******************************************************************************/

		return true;
	}

	bool shoot_rays(bool verbose)
	{
		set_threads(threads, 16, 16);
		set_blocks(threads, blocks, 16 * num_ray_blocks.re, 16 * num_ray_blocks.im);

		int* percentage = nullptr;
		cudaMallocManaged(&percentage, sizeof(int));
		if (cuda_error("cudaMallocManaged(*percentage)", false, __FILE__, __LINE__)) return false;

		*percentage = 1;

		/******************************************************************************
		shoot rays and calculate time taken in seconds
		******************************************************************************/
		std::cout << "Shooting rays...\n";
		stopwatch.start();
		shoot_rays_kernel<T> <<<blocks, threads, sizeof(TreeNode<T>) + treenode::MAX_NUM_STARS_DIRECT * sizeof(star<T>)>>> (kappa_tot, shear, theta_e, stars, kappa_star, tree[0], rays_level - ray_blocks_level,
			rectangular, corner, approx, taylor_smooth, center_x, half_length_image, num_ray_blocks,
			center_y, half_length_source, pixels_minima, pixels_saddles, pixels, num_pixels, percentage);
		if (cuda_error("shoot_rays_kernel", true, __FILE__, __LINE__)) return false;
		t_ray_shoot = stopwatch.stop();
		std::cout << "\nDone shooting rays. Elapsed time: " << t_ray_shoot << " seconds.\n\n";

		return true;
	}

	bool create_histograms(bool verbose)
	{
		/******************************************************************************
		create histograms of pixel values
		******************************************************************************/

		if (write_histograms)
		{
			std::cout << "Creating histograms...\n";
			stopwatch.start();

			cudaMallocManaged(&min_rays, sizeof(int));
			if (cuda_error("cudaMallocManaged(*min_rays)", false, __FILE__, __LINE__)) return false;
			cudaMallocManaged(&max_rays, sizeof(int));
			if (cuda_error("cudaMallocManaged(*max_rays)", false, __FILE__, __LINE__)) return false;

			*min_rays = std::numeric_limits<int>::max();
			*max_rays = 0;


			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels, num_pixels, min_rays, max_rays);
			if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels, min_rays, max_rays);
				if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
				histogram_min_max_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels, min_rays, max_rays);
				if (cuda_error("histogram_min_max_kernel", true, __FILE__, __LINE__)) return false;
			}

			histogram_length = *max_rays - *min_rays + 1;

			cudaMallocManaged(&histogram, histogram_length * sizeof(int));
			if (cuda_error("cudaMallocManaged(*histogram)", false, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				cudaMallocManaged(&histogram_minima, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_minima)", false, __FILE__, __LINE__)) return false;
				cudaMallocManaged(&histogram_saddles, histogram_length * sizeof(int));
				if (cuda_error("cudaMallocManaged(*histogram_saddles)", false, __FILE__, __LINE__)) return false;
			}

			
			set_threads(threads, 512);
			set_blocks(threads, blocks, histogram_length);

			initialize_array_kernel<T> <<<blocks, threads>>> (histogram, 1, histogram_length);
			if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				initialize_array_kernel<T> <<<blocks, threads>>> (histogram_minima, 1, histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
				initialize_array_kernel<T> <<<blocks, threads>>> (histogram_saddles, 1, histogram_length);
				if (cuda_error("initialize_array_kernel", true, __FILE__, __LINE__)) return false;
			}

			
			set_threads(threads, 16, 16);
			set_blocks(threads, blocks, num_pixels, num_pixels);

			histogram_kernel<T> <<<blocks, threads>>> (pixels, num_pixels, *min_rays, histogram);
			if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			if (write_parities)
			{
				histogram_kernel<T> <<<blocks, threads>>> (pixels_minima, num_pixels, *min_rays, histogram_minima);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
				histogram_kernel<T> <<<blocks, threads>>> (pixels_saddles, num_pixels, *min_rays, histogram_saddles);
				if (cuda_error("histogram_kernel", true, __FILE__, __LINE__)) return false;
			}
			t_elapsed = stopwatch.stop();
			std::cout << "Done creating histograms. Elapsed time: " + std::to_string(t_elapsed) + " seconds.\n\n";
		}

		/******************************************************************************
		done creating histograms of pixel values
		******************************************************************************/

		return true;
	}

	bool write_files(bool verbose)
	{
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
			return false;
		}
		outfile << "kappa_tot " << kappa_tot << "\n";
		outfile << "shear " << shear << "\n";
		outfile << "mu_ave " << mu_ave << "\n";
		outfile << "smooth_fraction " << (1 - kappa_star / kappa_tot) << "\n";
		outfile << "kappa_star " << kappa_star << "\n";
		if (starfile == "")
		{
			outfile << "kappa_star_actual " << kappa_star_actual << "\n";
		}
		outfile << "theta_e " << theta_e << "\n";
		outfile << "random_seed " << random_seed << "\n";
		if (starfile == "")
		{
			outfile << "mass_function " << mass_function_str << "\n";
			if (mass_function_str == "salpeter" || mass_function_str == "kroupa")
			{
				outfile << "m_solar " << m_solar << "\n";
			}
			outfile << "m_lower " << m_lower << "\n";
			outfile << "m_upper " << m_upper << "\n";
			outfile << "mean_mass " << mean_mass << "\n";
			outfile << "mean_mass2 " << mean_mass2 << "\n";
		}
		outfile << "m_lower_actual " << m_lower_actual << "\n";
		outfile << "m_upper_actual " << m_upper_actual << "\n";
		outfile << "mean_mass_actual " << mean_mass_actual << "\n";
		outfile << "mean_mass2_actual " << mean_mass2_actual << "\n";
		outfile << "light_loss " << light_loss << "\n";
		outfile << "num_stars " << num_stars << "\n";
		if (rectangular)
		{
			outfile << "corner_x1 " << corner.re << "\n";
			outfile << "corner_x2 " << corner.im << "\n";
			if (approx)
			{
				outfile << "taylor_smooth " << taylor_smooth << "\n";
			}
		}
		else
		{
			outfile << "rad " << corner.abs() << "\n";
		}
		outfile << "safety_scale " << safety_scale << "\n";
		outfile << "center_y1" << center_y.re << "\n";
		outfile << "center_y2" << center_y.im << "\n";
		outfile << "half_length_source " << half_length_source << "\n";
		outfile << "num_pixels " << num_pixels << "\n";
		outfile << "mean_rays_per_pixel " << num_rays_source << "\n";
		outfile << "half_length_image_x1 " << half_length_image.re << "\n";
		outfile << "half_length_image_x2 " << half_length_image.im << "\n";
		outfile << "ray_sep " << ray_sep << "\n";
		outfile << "t_ray_shoot " << t_ray_shoot << "\n";
		outfile.close();
		std::cout << "Done writing parameter info to file " << fname << "\n\n";


		std::cout << "Writing star info...\n";
		fname = outfile_prefix + "irs_stars" + outfile_type;
		if (!write_star_file<T>(num_stars, rectangular, corner, theta_e, stars, fname))
		{
			std::cerr << "Error. Unable to write star info to file " << fname << "\n";
			return false;
		}
		std::cout << "Done writing star info to file " << fname << "\n\n";


		/******************************************************************************
		histograms of magnification maps
		******************************************************************************/
		if (write_histograms)
		{
			std::cout << "Writing magnification histograms...\n";

			fname = outfile_prefix + "irs_numrays_numpixels.txt";
			if (!write_histogram<T>(histogram, histogram_length, *min_rays, fname))
			{
				std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
				return false;
			}
			std::cout << "Done writing magnification histogram to file " << fname << "\n";
			if (write_parities)
			{
				fname = outfile_prefix + "irs_numrays_numpixels_minima.txt";
				if (!write_histogram<T>(histogram_minima, histogram_length, *min_rays, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnification histogram to file " << fname << "\n";

				fname = outfile_prefix + "irs_numrays_numpixels_saddles.txt";
				if (!write_histogram<T>(histogram_saddles, histogram_length, *min_rays, fname))
				{
					std::cerr << "Error. Unable to write magnification histogram to file " << fname << "\n";
					return false;
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
				return false;
			}
			std::cout << "Done writing magnifications to file " << fname << "\n";
			if (write_parities)
			{
				fname = outfile_prefix + "irs_magnifications_minima" + outfile_type;
				if (!write_array<int>(pixels_minima, num_pixels, num_pixels, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnifications to file " << fname << "\n";

				fname = outfile_prefix + "irs_magnifications_saddles" + outfile_type;
				if (!write_array<int>(pixels_saddles, num_pixels, num_pixels, fname))
				{
					std::cerr << "Error. Unable to write magnifications to file " << fname << "\n";
					return false;
				}
				std::cout << "Done writing magnifications to file " << fname << "\n";
			}
			std::cout << "\n";
		}

		return true;
	}


public:

	bool run(bool verbose)
	{
		if (!check_input_params(verbose)) return false;
		if (!calculate_derived_params(verbose)) return false;
		if (!allocate_initialize_memory(verbose)) return false;
		if (!populate_star_array(verbose)) return false;
		if (!create_tree(verbose)) return false;
		if (!shoot_rays(verbose)) return false;

		return true;
	}

	bool save(bool verbose)
	{
		if (!create_histograms(verbose)) return false;
		if (!write_files(verbose)) return false;

		return true;
	}

};

