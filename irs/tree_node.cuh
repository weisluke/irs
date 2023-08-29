#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "star.cuh"

/******************************************************************************
template class for handling nodes of a tree
******************************************************************************/
template <typename T>
class TreeNode
{
public:
    Complex<T> center;
    T half_length;

    int level;
    int index;

    TreeNode* parent;
    TreeNode* children[4];
    TreeNode* neighbors[8];
    int numneighbors;
    TreeNode* interactionlist[27];
    int numinterlist;

    int stars;
    int numstars;

    int expansion_order;
    Complex<T> multipole_coeffs[21];
    Complex<T> taylor_coeffs[21];


    /******************************************************************************
	default constructor
	******************************************************************************/
	__host__ __device__ TreeNode(Complex<T> ctr, T hl, T lvl, int idx, TreeNode* p = nullptr)
	{
        center = ctr;
        half_length = hl;
        level = lvl;
        index = idx;

		parent = p;
		for (int i = 0; i < 4; i++)
        {
            children[i] = nullptr;
        }
        for (int i = 0; i < 8; i++)
        {
            neighbors[i] = nullptr;
        }
        numneighbors = 0;
        for (int i = 0; i < 27; i++)
        {
            interactionlist[i] = nullptr;
        }
        numinterlist = 0;

        stars = 0;
        numstars = 0;
        expansion_order = 0;

        for (int i = 0; i <= 20; i++)
        {
            multipole_coeffs[i] = Complex<T>();
            taylor_coeffs[i] = Complex<T>();
        }
	}
    
    /******************************************************************************
	assignment operator
	******************************************************************************/
	template <typename U> __host__ __device__ TreeNode& operator=(const TreeNode<U>& n)
	{
        center = n.center;
        half_length = n.half_length;
        level = n.level;
        index = n.index;

        parent = n.parent;
		for (int i = 0; i < 4; i++)
        {
            children[i] = n.children[i];
        }

        for (int i = 0; i < 8; i++)
        {
            neighbors[i] = n.neighbors[i];
        }
        numneighbors = n.numneighbors;
        for (int i = 0; i < 27; i++)
        {
            interactionlist[i] = n.interactionlist[i];
        }
        numinterlist = n.numinterlist;

        stars = n.stars;
        numstars = n.numstars;
        expansion_order = n.expansion_order;
        for (int i = 0; i <= 20; i++)
        {
            multipole_coeffs[i] = n.multipole_coeffs[i];
            taylor_coeffs[i] = n.taylor_coeffs[i];
        }

		return *this;
	}

    __host__ __device__ void make_child(int idx, TreeNode* nodes)
    {
        T new_half_length = half_length / 2;

        Complex<T> offsets[4] = 
        {
            Complex<T>(1, 1),
            Complex<T>(-1, 1),
            Complex<T>(-1, -1),
            Complex<T>(1, -1)
        };

        Complex<T> new_center = center + new_half_length * offsets[idx - 1];

        int new_index = 4 * index + idx;

        nodes[new_index] = TreeNode(new_center, new_half_length, level + 1, new_index, this);
        children[idx - 1] = &nodes[new_index];
    }

    __host__ __device__ void make_children(TreeNode* nodes)
    {
        for (int i = 1; i <= 4; i++)
        {
            make_child(i, nodes);
        }
    }
    
    __device__ void set_neighbors()
    {
        if (level == 0)
        {
            return;
        }

        for (int i = 0; i < 4; i++)
        {
            if (parent->children[i] != this)
            {
                neighbors[numneighbors] = parent->children[i];
                numneighbors++;
            }
        }
        for (int i = 0; i < parent->numneighbors; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                TreeNode* node = parent->neighbors[i]->children[j];

                if (fabs(node->center.re - center.re) < 3 * half_length
                    && fabs(node->center.im - center.im) < 3 * half_length)
                {
                    neighbors[numneighbors] = node;
                    numneighbors++;
                }
                else
                {
                    interactionlist[numinterlist] = node;
                    numinterlist++;
                }
            }
        }
    }

    __host__ __device__ void set_multipole_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            multipole_coeffs[i] = coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }
    
    __host__ __device__ void add_multipole_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            multipole_coeffs[i] += coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

    __host__ __device__ void set_taylor_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            taylor_coeffs[i] = coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

    __host__ __device__ void add_taylor_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            taylor_coeffs[i] += coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

};


__host__ __device__ int get_min_index(int level)
{
    /******************************************************************************
    min index for a level = ( 4 ^ n - 1 ) / 3 = ( 2 ^ (2n) - 1) / 3
    ******************************************************************************/
    int min_index = 1;
    min_index = min_index << (2 * level);
    min_index = (min_index - 1) / 3;
    return min_index;
}
__host__ __device__ int get_max_index(int level)
{
    /******************************************************************************
    max index for a level = ( 4 ^ (n + 1) - 4 ) / 3 = ( 4 * 2 ^ (2n) - 4) / 3
    ******************************************************************************/
    int max_index = 4;
    max_index = max_index << (2 * level);
    max_index = (max_index - 4) / 3;
    return max_index;
}

__host__ __device__ int get_num_nodes(int level)
{
    int min_index = get_min_index(level);
    int max_index = get_max_index(level);
    return max_index - min_index + 1;
}

template <typename T>
__global__ void create_children_kernel(TreeNode<T>* nodes, int level)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = blockDim.x * gridDim.x;

    int min_index = get_min_index(level);

    for (int i = x_index; i < get_num_nodes(level); i += x_stride)
    {
        nodes[min_index + i].make_children(nodes);
    }
}

template <typename T>
__global__ void set_neighbors_kernel(TreeNode<T>* nodes, int level)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int x_stride = blockDim.x * gridDim.x;

    int min_index = get_min_index(level);

    for (int i = x_index; i < get_num_nodes(level); i += x_stride)
    {
        nodes[min_index + i].set_neighbors();
    }
}

template <typename T>
__global__ void get_min_max_stars_kernel(TreeNode<T>* nodes, int level, int* min_n_stars, int* max_n_stars)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

    int min_index = get_min_index(level);

	for (int i = x_index; i < get_num_nodes(level); i += x_stride)
	{
        TreeNode<T>* node = &nodes[min_index + i];
        int nstars = node->numstars;
        for (int j = 0; j < node->numneighbors; j++)
        {
            nstars += node->neighbors[j]->numstars;
        }
        atomicMin(min_n_stars, nstars);
        atomicMax(max_n_stars, nstars);
	}
}

template <typename T>
__global__ void sort_stars_kernel(TreeNode<T>* nodes, int level, star<T>* stars, star<T>* temp_stars)
{
    /******************************************************************************
    each block is a node, and each thread is a star within the parent node
    ******************************************************************************/
    int node_index = blockIdx.x;

	int x_index = threadIdx.x;
	int x_stride = blockDim.x;

    __shared__ int n_stars_top;
    __shared__ int n_stars_bottom;
    __shared__ int n_stars_children[4];

    if (threadIdx.x == 0)
    {
        n_stars_top = 0;
        n_stars_bottom = 0;
        n_stars_children[0] = 0;
        n_stars_children[1] = 0;
        n_stars_children[2] = 0;
        n_stars_children[3] = 0;
    }
    __syncthreads();

    int min_index = get_min_index(level);

    if (node_index < get_num_nodes(level))
    {
        TreeNode<T>* node = &nodes[min_index + node_index];

        /******************************************************************************
        in the first pass, figure out whether the star is above or below the center
        ******************************************************************************/
        for (int i = x_index; i < node->numstars; i += x_stride)
        {
            if (stars[node->stars + i].position.im >= node->center.im)
            {
                temp_stars[node->stars + atomicAdd(&n_stars_top, 1)] = stars[node->stars + i];
            }
            else
            {
                temp_stars[node->stars + node->numstars - 1 - atomicAdd(&n_stars_bottom, 1)] = stars[node->stars + i];
            }
        }
        __syncthreads();

        /******************************************************************************
        in the second pass, figure out whether the star is left or right of the center
        ******************************************************************************/
        for (int i = x_index; i < node->numstars; i += x_stride)
        {
            /******************************************************************************
            if the star was in the top, then sort left and right
            ******************************************************************************/
            if (i < n_stars_top)
            {
                if (temp_stars[node->stars + i].position.re >= node->center.re)
                {
                    stars[node->stars + atomicAdd(&n_stars_children[0], 1)] = temp_stars[node->stars + i];
                }
                else
                {
                    stars[node->stars + n_stars_top - 1 - atomicAdd(&n_stars_children[1], 1)] = temp_stars[node->stars + i];
                }
            }
            /******************************************************************************
            if the star was in the bottom, then sort left and right
            ******************************************************************************/
            else
            {
                if (temp_stars[node->stars + i].position.re < node->center.re)
                {
                    stars[node->stars + n_stars_top + atomicAdd(&n_stars_children[2], 1)] = temp_stars[node->stars + i];
                }
                else
                {
                    stars[node->stars + node->numstars - 1 - atomicAdd(&n_stars_children[3], 1)] = temp_stars[node->stars + i];
                }
            }
        }
        __syncthreads();

        /******************************************************************************
        once the sorting is done, assign the starting position of stars to the children
        along with the number of stars
        ******************************************************************************/
        if (threadIdx.x == 0)
        {
            node->children[0]->stars = node->stars;
            node->children[0]->numstars = n_stars_children[0];

#pragma unroll
            for (int i = 1; i < 4; i++)
            {
                node->children[i]->stars = node->children[i - 1]->stars + node->children[i - 1]->numstars;
                node->children[i]->numstars = n_stars_children[i];
            }
        }
    }
}


template <typename T>
__device__ void calculate_multipole_coeff(TreeNode<T>* node, Complex<T>* coeffs, int power, star<T>* stars)
{
    Complex<T> result;

    if (power == 0)
    {
        for (int i = 0; i < node->numstars; i++)
        {
            result += stars[node->stars + i].mass;
        }
    }
    else
    {
        for (int i = 0; i < node->numstars; i++)
        {
            result -= (stars[node->stars + i].position - node->center).pow(power) * stars[node->stars + i].mass;
        }
        result /= power;
    }

    coeffs[power] = result;
}


template <typename T>
__global__ void calculate_multipole_coeffs_kernel(TreeNode<T>* nodes, int level, int power, star<T>* stars)
{
    /******************************************************************************
    each block is a node, and each thread calculates a multipole coefficient
    ******************************************************************************/
    int node_index = blockIdx.x;

	int x_index = threadIdx.x;
	int x_stride = blockDim.x;

    extern __shared__ Complex<T> coeffs[];

    int min_index = get_min_index(level);

    if (node_index < get_num_nodes(level))
    {
        TreeNode<T>* node = &nodes[min_index + node_index];

        for (int i = x_index; i <= power; i += x_stride)
        {
            calculate_multipole_coeff(node, coeffs, i, stars);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            node->set_multipole_coeffs(coeffs, power);
        }
    }
}

template <typename T>
__device__ void calculate_M2M_coeff(TreeNode<T>* node, Complex<T>* coeffs, int power, int* binomcoeffs)
{
    Complex<T> result;

    if (power == 0)
    {
        result = node->multipole_coeffs[0];
    }
    else
    {
        Complex<T> dz = node->center - node->parent->center;
        for (int i = power; i >= 1; i--)
        {
            result += node->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power - 1, i - 1);
            result /= dz;
        }
        result -= node->multipole_coeffs[0] / power;
        result *= dz.pow(power);
    }

    coeffs[power] = result;
}

template <typename T>
__global__ void calculate_M2M_coeffs_kernel(TreeNode<T>* nodes, int level, int power, int* binomcoeffs)
{
    /******************************************************************************
    each block is a node, and each thread calculates a multipole coefficient
    ******************************************************************************/
    int node_index = blockIdx.x;

    int x_index = threadIdx.x;
    int x_stride = blockDim.x;
    int y_index = threadIdx.y;
    int y_stride = blockDim.y;

    extern __shared__ Complex<T> coeffs[];

    int min_index = get_min_index(level);

    if (node_index < get_num_nodes(level))
    {
        TreeNode<T>* node = &nodes[min_index + node_index];

        for (int i = y_index; i < 4; i += y_stride)
        {
            TreeNode<T>* child = node->children[i];
            for (int j = x_index; j <= power; j += x_stride)
            {
                calculate_M2M_coeff(child, &coeffs[(power + 1) * i], j, binomcoeffs);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                node->add_multipole_coeffs(&coeffs[(power + 1) * i], power);
            }
        }
    }
}

template <typename T>
__device__ void calculate_L2L_coeff(TreeNode<T>* node, Complex<T>* coeffs, int power, int maxpower, int* binomcoeffs)
{
    Complex<T> result;
    Complex<T> dz = node->center - node->parent->center;

    for (int i = maxpower; i >= 1; i--)
    {
        result += node->parent->taylor_coeffs[i] * get_binomial_coeff(binomcoeffs, i, power);
        result *= dz;
    }
    result /= dz.pow(power);

    coeffs[power] = result;
}

template <typename T>
__global__ void calculate_L2L_coeffs_kernel(TreeNode<T>* nodes, int level, int power, int* binomcoeffs)
{
    /******************************************************************************
    each block is a node, and each thread calculates a multipole coefficient
    ******************************************************************************/
    int node_index = blockIdx.x;

    int x_index = threadIdx.x;
    int x_stride = blockDim.x;

    extern __shared__ Complex<T> coeffs[];

    int min_index = get_min_index(level);

    if (node_index < get_num_nodes(level))
    {
        TreeNode<T>* node = &nodes[min_index + node_index];

        for (int i = x_index; i <= power; i += x_stride)
        {
            calculate_L2L_coeff(node, coeffs, i, power, binomcoeffs);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            node->set_taylor_coeffs(coeffs, power);
        }
    }
}

template <typename T>
__device__ void calculate_M2L_coeff(TreeNode<T>* node_from, TreeNode<T>* node_to, Complex<T>* coeffs, int power, int maxpower, int* binomcoeffs)
{
    Complex<T> result;
    Complex<T> dz = node_from->center - node_to->center;

    if (power == 0)
    {
        for (int i = maxpower; i >= 1; i--)
        {
            result += node_from->multipole_coeffs[i];
            result /= -dz;
        }
        result += node_from->multipole_coeffs[0] * (-dz).log();
    }
    else
    {
        for (int i = maxpower; i >= 1; i--)
        {
            result += node_from->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power + i - 1, i - 1);
            result /= -dz;
        }
        result -= node_from->multipole_coeffs[0] / power;
        result /= dz.pow(power);
    }

    coeffs[power] = result;
}

template <typename T>
__global__ void calculate_M2L_coeffs_kernel(TreeNode<T>* nodes, int level, int power, int* binomcoeffs)
{
    /******************************************************************************
    each block is a node, and each thread calculates a Taylor coefficient
    ******************************************************************************/
    int node_index = blockIdx.x;

    int x_index = threadIdx.x;
    int x_stride = blockDim.x;
    int y_index = threadIdx.y;
    int y_stride = blockDim.y;

    extern __shared__ Complex<T> coeffs[];

    int min_index = get_min_index(level);

    if (node_index < get_num_nodes(level))
    {
        TreeNode<T>* node = &nodes[min_index + node_index];

        for (int i = y_index; i < node->numinterlist; i += y_stride)
        {
            for (int j = x_index; j <= power; j += x_stride)
            {
                calculate_M2L_coeff(node->interactionlist[i], node, &coeffs[(power + 1) * i], j, power, binomcoeffs);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0)
        {
            for (int i = 0; i < node->numinterlist; i++)
            {
                node->add_taylor_coeffs(&coeffs[(power + 1) * i], power);
            }
        }
    }
}