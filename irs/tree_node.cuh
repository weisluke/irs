#pragma once

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
    int stars;
    int numstars;
    int multipole_order;
    Complex<T> multipole_coeffs[21];


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
        stars = 0;
        numstars = 0;
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
        stars = n.stars;
        numstars = n.numstars;

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

    __host__ __device__ void set_multipole_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            multipole_coeffs[i] = coeffs[i];
        }
        multipole_order = power;
    }

};


__host__ __device__ int get_min_index(int level)
{
    if (level == 0)
    {
        return 0;
    }
    int min_index = 2;
    min_index = min_index << (2 * level - 1);
    min_index = (min_index - 1) / 3;
    return min_index;
}
__host__ __device__ int get_max_index(int level)
{
    if (level == 0)
    {
        return 0;
    }
    int max_index = 2;
    max_index = max_index << (2 * level + 1);
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
__global__ void get_min_max_stars_kernel(TreeNode<T>* nodes, int level, int* min_n_stars, int* max_n_stars)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

    int min_index = get_min_index(level);

	for (int i = x_index; i < get_num_nodes(level); i += x_stride)
	{
        atomicMin(min_n_stars, nodes[i + min_index].numstars);
        atomicMax(max_n_stars, nodes[i + min_index].numstars);
	}
}


template <typename T>
__global__ void create_children_kernel(TreeNode<T>* nodes, int level)
{
	int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	int x_stride = blockDim.x * gridDim.x;

    int min_index = get_min_index(level);

	for (int i = x_index; i < get_num_nodes(level); i += x_stride)
	{
        nodes[i + min_index].make_children(nodes);
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

        for (int i = x_index; i < node->numstars; i += x_stride)
        {
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

        if (threadIdx.x == 0)
        {
            node->children[0]->stars = node->stars;
            node->children[0]->numstars = n_stars_children[0];

#pragma unroll
            for (int i = 1; i < 4; i++)
            {
                node->children[i]->stars = node->children[i - 1]->stars + n_stars_children[i - 1];
                node->children[i]->numstars = n_stars_children[i];
            }
        }
    }
}


template <typename T>
__device__ void calculate_multipole_coeff(TreeNode<T>* node, star<T>* stars, Complex<T>* coeffs, int power)
{
    Complex<T> result;
    for (int i = 0; i < node->numstars; i++)
    {
        result += (stars[node->stars + i].position - node->center).pow(power) * stars[node->stars + i].mass;
    }
    coeffs[power] = result;
}


template <typename T>
__global__ void calculate_multipole_coeffs_kernel(TreeNode<T>* nodes, int level, star<T>* stars, int power)
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
            calculate_multipole_coeff(node, stars, coeffs, i);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            node->set_multipole_coeffs(coeffs, power);
        }
    }
}