#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "star.cuh"


/******************************************************************************
maximum expansion order for the fast multipole method
******************************************************************************/
namespace treenode
{
    const int MAX_EXPANSION_ORDER = 25;
}

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

    TreeNode* parent;
    TreeNode* children[4];
    int numchildren;
    TreeNode* neighbors[8];
    int numneighbors;
    TreeNode* interactionlist[27];
    int numinterlist;

    int stars;
    int numstars;

    int expansion_order;
    Complex<T> multipole_coeffs[treenode::MAX_EXPANSION_ORDER + 1];
    Complex<T> local_coeffs[treenode::MAX_EXPANSION_ORDER + 1];


    /******************************************************************************
	default constructor
	******************************************************************************/
	__host__ __device__ TreeNode(Complex<T> ctr, T hl, int lvl, TreeNode* p = nullptr)
	{
        center = ctr;
        half_length = hl;
        level = lvl;

		parent = p;
		for (int i = 0; i < 4; i++)
        {
            children[i] = nullptr;
        }
        numchildren = 0;
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

        for (int i = 0; i <= treenode::MAX_EXPANSION_ORDER; i++)
        {
            multipole_coeffs[i] = Complex<T>();
            local_coeffs[i] = Complex<T>();
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

        parent = n.parent;
		for (int i = 0; i < 4; i++)
        {
            children[i] = n.children[i];
        }
        numchildren = n.numchildren;
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
        for (int i = 0; i <= treenode::MAX_EXPANSION_ORDER; i++)
        {
            multipole_coeffs[i] = n.multipole_coeffs[i];
            local_coeffs[i] = n.local_coeffs[i];
        }

		return *this;
	}

    __host__ __device__ void make_child(TreeNode* nodes, int start, int idx)
    {
        T new_half_length = half_length / 2;

        Complex<T> offsets[4] = 
        {
            Complex<T>(1, 1),
            Complex<T>(-1, 1),
            Complex<T>(-1, -1),
            Complex<T>(1, -1)
        };

        Complex<T> new_center = center + new_half_length * offsets[idx];

        nodes[start + idx] = TreeNode(new_center, new_half_length, level + 1, this);
        children[idx] = &nodes[new_index];
    }

    __host__ __device__ void make_children(TreeNode* nodes, int start)
    {
        for (int i = 0; i < 4; i++)
        {
            make_child(nodes, start, i);
        }
        numchildren = 4;
    }


    /******************************************************************************
    set neighbors for a node
    ******************************************************************************/
    __device__ void set_neighbors()
    {
        /******************************************************************************
        root node has no neighbors
        ******************************************************************************/
        if (level == 0)
        {
            return;
        }

        /******************************************************************************
        add parent's children as neighbors
        ******************************************************************************/
        for (int i = 0; i < 4; i++)
        {
            if (parent->children[i] != this)
            {
                neighbors[numneighbors] = parent->children[i];
                numneighbors++;
            }
        }
        /******************************************************************************
        parent's neighbors' children are on the same level as the node
        add the close ones as neighbors, and the far ones to the interaction list
        ******************************************************************************/
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

    __host__ __device__ void set_local_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            local_coeffs[i] = coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

    __host__ __device__ void add_local_coeffs(Complex<T>* coeffs, int power)
    {
        for (int i = 0; i <= power; i++)
        {
            local_coeffs[i] += coeffs[i];
        }
        expansion_order = (expansion_order > power ? expansion_order : power);
    }

};


namespace treenode
{

    /******************************************************************************
    get the minimum and maximum number of stars contained within a node and its
    neighbors

    \param nodes -- pointer to tree
    \param level -- what level in the tree to get the number of stars for
    \param min_n_stars -- pointer to minimum number of stars
    \param max_n_stars -- pointer to maximum number of stars
    ******************************************************************************/
    template <typename T>
    __global__ void get_min_max_stars_kernel(TreeNode<T>* nodes, int numnodes, int* min_n_stars, int* max_n_stars, int* num_overfull_nodes, int MAX_NUM_STARS)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            TreeNode<T>* node = &nodes[i];
            int nstars = node->numstars;
            for (int j = 0; j < node->numneighbors; j++)
            {
                nstars += node->neighbors[j]->numstars;
            }
            atomicMin(min_n_stars, nstars);
            atomicMax(max_n_stars, nstars);
            if (nstars > MAX_NUM_STARS)
            {
                atomicAdd(num_overfull_nodes, 1);
            }
        }
    }

    /******************************************************************************
    create the children for nodes at a given level

    \param nodes -- pointer to tree
    \param level -- what level in the tree to make the children of
    ******************************************************************************/
    template <typename T>
    __global__ void create_children_kernel(TreeNode<T>* parents, int nparents, TreeNode<T>* children, int* nchildren, int MAX_NUM_STARS)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < nparents; i += x_stride)
        {
            TreeNode<T>* node = &parents[i];

            int nstars = node->numstars;
            for (int j = 0; j < node->numneighbors; j++)
            {
                nstars += node->neighbors[j]->numstars;
            }

            if (nstars > MAX_NUM_STARS)
            {
                node->make_children(children, atomicAdd(nchildren));
            }
        }
    }

    /******************************************************************************
    sort the stars into each node for a given level

    \param nodes -- pointer to tree
    \param level -- what level in the tree to sort the stars
    \param stars -- pointer to array of point mass lenses
    \param temp_stars -- pointer to temp array of point mass lenses, used for
                         swapping
    ******************************************************************************/
    template <typename T>
    __global__ void sort_stars_kernel(TreeNode<T>* parents, int nparents, TreeNode<T>* children, int nchildren, star<T>* stars, star<T>* temp_stars)
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

        if (node_index < nparents)
        {
            TreeNode<T>* node = &parents[node_index];

            /******************************************************************************
            in the first pass, figure out whether the star is above or below the center
            ******************************************************************************/
            for (int i = x_index; i < node->numstars; i += x_stride)
            {
                if (stars[node->stars + i].position.im >= node->center.im)
                {
                    /******************************************************************************
                    atomic addition returns the old value, so this is guaranteed to copy the star
                    to a unique location in the temp array
                    ******************************************************************************/
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

    /******************************************************************************
    set the neighbors for nodes at a given level

    \param nodes -- pointer to tree
    \param level -- what level in the tree to set the neighbors of
    ******************************************************************************/
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

    /******************************************************************************
    find the node nearest to a given position at a given level

    \param z -- position to find the nearest node to
    \param nodes -- pointer to tree
    \param level -- what level in the tree to find the nearest node

    \return pointer to nearest node
    ******************************************************************************/
    template <typename T>
    __device__ TreeNode<T>* get_nearest_node(Complex<T> z, TreeNode<T>* nodes, int level)
    {
        TreeNode<T>* node = &nodes[0];
        for (int i = 1; i <= level; i++)
        {
            if (z.re >= node->center.re && z.im >= node->center.im)
            {
                node = node->children[0];
            }
            else if (z.re < node->center.re && z.im >= node->center.im)
            {
                node = node->children[1];
            }
            else if (z.re < node->center.re && z.im < node->center.im)
            {
                node = node->children[2];
            }
            else
            {
                node = node->children[3];
            }
        }
        return node;
    }

}

