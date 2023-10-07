#pragma once

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
    TreeNode* children[4]; // a node has at most 4 children
    int numchildren;
    TreeNode* neighbors[8]; // a node has at most 8 neighbors
    int numneighbors;
    TreeNode* interactionlist[27]; // a node has at most 27 elements in its interaction list
    int numinterlist;

    int stars; // position of this node's stars in array of stars
    int numstars; // number of stars in this node

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

    /******************************************************************************
    create a child node

    \param nodes -- pointer to array of nodes for the next level
    \param start -- where to begin placing children
    \param idx -- index of the child to be created (0, 1, 2, or 3)
    ******************************************************************************/
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

        int new_index = start + idx;
        nodes[new_index] = TreeNode(new_center, new_half_length, level + 1, this);
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
        for (int i = 0; i < parent->numchildren; i++)
        {
            TreeNode* node = parent->children[i];

            /******************************************************************************
            if node is empty, skip it
            ******************************************************************************/
            if (node->numstars == 0)
            {
                continue;
            }

            if (node != this)
            {
                neighbors[numneighbors++] = node;
            }
        }
        /******************************************************************************
        parent's neighbors' children are on the same level as the node
        add the close ones as neighbors, and the far ones to the interaction list
        ******************************************************************************/
        for (int i = 0; i < parent->numneighbors; i++)
        {
            TreeNode* neighbor = parent->neighbors[i];
            
            for (int j = 0; j < neighbor->numchildren; j++)
            {
                TreeNode* node = neighbor->children[j];

                /******************************************************************************
                if node is empty, skip it
                ******************************************************************************/
                if (node->numstars == 0)
                {
                    continue;
                }

                if (fabs(node->center.re - center.re) < 3 * half_length
                    && fabs(node->center.im - center.im) < 3 * half_length)
                {
                    neighbors[numneighbors++] = node;
                }
                else
                {
                    interactionlist[numinterlist++] = node;
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
    get the number of non empty nodes, and the minimum and maximum number of stars
    contained within a node and its neighbors

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param num_nonempty_nodes -- pointer to number of non empty nodes
    \param min_n_stars -- pointer to minimum number of stars
    \param max_n_stars -- pointer to maximum number of stars
    ******************************************************************************/
    template <typename T>
    __global__ void get_node_star_info_kernel(TreeNode<T>* nodes, int numnodes, int* num_nonempty_nodes, int* min_n_stars, int* max_n_stars)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            TreeNode<T>* node = &nodes[i];
            int nstars = node->numstars;

            if (nstars > 0)
            {
                atomicAdd(num_nonempty_nodes, 1);
            }
            
            for (int j = 0; j < node->numneighbors; j++)
            {
                nstars += node->neighbors[j]->numstars;
            }

            atomicMin(min_n_stars, nstars);
            atomicMax(max_n_stars, nstars);
        }
    }

    /******************************************************************************
    create the children for nodes 

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param num_nonempty_nodes -- pointer to number of non empty nodes
    \param children -- pointer to child nodes
    ******************************************************************************/
    template <typename T>
    __global__ void create_children_kernel(TreeNode<T>* nodes, int numnodes, int* num_nonempty_nodes, TreeNode<T>* children)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            TreeNode<T>* node = &nodes[i];
            int nstars = node->numstars;

            if (nstars > 0)
            {
                /******************************************************************************
                assumes that array of child nodes has 4 * num_nonempty_nodes at the start
                every 4 elements are then the children of this node
                atomicSub ensures that children are placed at unique locations
                ******************************************************************************/
                node->make_children(children, 4 * atomicSub(num_nonempty_nodes, 1));
            }
        }
    }

    /******************************************************************************
    sort the stars into the children of each node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param stars -- pointer to array of point mass lenses
    \param temp_stars -- pointer to temp array of point mass lenses, used for
                         swapping
    ******************************************************************************/
    template <typename T>
    __global__ void sort_stars_kernel(TreeNode<T>* nodes, int numnodes, star<T>* stars, star<T>* temp_stars)
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

        if (node_index < numnodes)
        {
            TreeNode<T>* node = &nodes[node_index];

            /******************************************************************************
            if node is empty, skip it
            ******************************************************************************/
            if (node->numstars == 0)
            {
                return;
            }

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
    set the neighbors for nodes

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    ******************************************************************************/
    template <typename T>
    __global__ void set_neighbors_kernel(TreeNode<T>* nodes, int numnodes)
    {
        int x_index = blockIdx.x * blockDim.x + threadIdx.x;
        int x_stride = blockDim.x * gridDim.x;

        for (int i = x_index; i < numnodes; i += x_stride)
        {
            nodes[i].set_neighbors();
        }
    }

    /******************************************************************************
    find the node nearest to a given position at a given level

    \param z -- position to find the nearest node to
    \param root -- pointer to root node

    \return pointer to nearest node
    ******************************************************************************/
    template <typename T>
    __device__ TreeNode<T>* get_nearest_node(Complex<T> z, TreeNode<T>* root)
    {
        TreeNode<T>* node = root;
        while (node->numchildren > 0)
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

