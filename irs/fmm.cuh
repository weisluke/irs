#pragma once

#include "binomial_coefficients.cuh"
#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"


namespace fmm
{

    /******************************************************************************
    calculate the multipole coefficient for a given power

    \param node -- pointer to node
    \param coeffs -- pointer to array of multipole coefficients
    \param power -- what order to find the multipole coefficient of
    \param stars -- pointer to array of point mass lenses
    ******************************************************************************/
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
                result -= ((stars[node->stars + i].position - node->center) / node->half_length).pow(power) * stars[node->stars + i].mass;
            }
            result /= power;
        }

        coeffs[power] = result;
    }

    /******************************************************************************
    calculate the multipole coefficients for a given power

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the multipole coefficients of
    \param stars -- pointer to array of point mass lenses
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_multipole_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, star<T>* stars)
    {
        /******************************************************************************
        each block is a node
        each thread calculates a multipole coefficient in the x direction
        ******************************************************************************/
        int node_index = blockIdx.x;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;

        extern __shared__ Complex<T> coeffs[];

        if (node_index < numnodes)
        {
            TreeNode<T>* node = &nodes[node_index];

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

    /******************************************************************************
    calculate the shifted multipole coefficient for a given power
    assumes that the expansion is being shifted to the center of the parent node

    \param node -- pointer to node
    \param coeffs -- pointer to array of multipole coefficients
    \param power -- what order to find the multipole coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
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
            Complex<T> dz = (node->center - node->parent->center) / node->half_length;
            for (int i = power; i >= 1; i--)
            {
                result += node->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power - 1, i - 1);
                result /= dz;
            }
            result -= node->multipole_coeffs[0] / power;
            for (int i = 0; i < power; i++)
            {
                result *= dz;
                result /= 2;
            }
        }

        coeffs[power] = result;
    }

    /******************************************************************************
    calculate the shifted multipole coefficients for a given power
    assumes that the expansions are being shifted to the center of the parent node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the multipole coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_M2M_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, int* binomcoeffs)
    {
        /******************************************************************************
        each block is a node
        each thread calculates a shifted multipole coefficient in the x direction for a
         child node in the y direction
        ******************************************************************************/
        int node_index = blockIdx.x;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;
        int y_index = threadIdx.y;
        int y_stride = blockDim.y;

        extern __shared__ Complex<T> coeffs[];

        if (node_index < numnodes)
        {
            TreeNode<T>* node = &nodes[node_index];

            for (int i = y_index; i < node->numchildren; i += y_stride)
            {
                TreeNode<T>* child = node->children[i];
                for (int j = x_index; j <= power; j += x_stride)
                {
                    calculate_M2M_coeff(child, &coeffs[(power + 1) * i], j, binomcoeffs);
                }
            }
            __syncthreads();
            /******************************************************************************
            finally, add the children's shifted multipole coefficients onto the parent
            this must be sequentially carried out by one thread
            ******************************************************************************/
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                for (int i = 0; i < node->numchildren; i++)
                {
                    node->add_multipole_coeffs(&coeffs[(power + 1) * i], power);
                }
            }
        }
    }

    /******************************************************************************
    calculate the shifted local coefficient for a given power
    assumes that the expansion is being shifted from the center of the parent node

    \param node -- pointer to node
    \param coeffs -- pointer to array of local coefficients
    \param power -- what order to find the local coefficient of
    \param maxpower -- maximum order to find the local coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_L2L_coeff(TreeNode<T>* node, Complex<T>* coeffs, int power, int maxpower, int* binomcoeffs)
    {
        Complex<T> result;
        Complex<T> dz = (node->center - node->parent->center) / node->parent->half_length;

        for (int i = maxpower; i >= 1; i--)
        {
            result += node->parent->local_coeffs[i] * get_binomial_coeff(binomcoeffs, i, power);
            result *= dz;
        }
        for (int i = 0; i < power; i++)
        {
            result /= dz;
            result /= 2;
        }

        coeffs[power] = result;
    }

    /******************************************************************************
    calculate the shifted local coefficients for a given power
    assumes that the expansions are being shifted from the center of the parent
     node

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the local coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_L2L_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, int* binomcoeffs)
    {
        /******************************************************************************
        each block is a node
        each thread calculates a local coefficient in the x direction
        ******************************************************************************/
        int node_index = blockIdx.x;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;

        extern __shared__ Complex<T> coeffs[];

        if (node_index < numnodes)
        {
            TreeNode<T>* node = &nodes[node_index];

            for (int i = x_index; i <= power; i += x_stride)
            {
                calculate_L2L_coeff(node, coeffs, i, power, binomcoeffs);
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                node->set_local_coeffs(coeffs, power);
            }
        }
    }

    /******************************************************************************
    calculate the local coefficient for a given power from the multipole
    coefficient of a far node

    \param node_from -- pointer to node whose multipole expansion is being used
    \param node_to -- pointer to node whose local expansion is being calculated
    \param coeffs -- pointer to array of local coefficients
    \param power -- what order to find the local coefficient of
    \param maxpower -- maximum order to find the local coefficient of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __device__ void calculate_M2L_coeff(TreeNode<T>* node_from, TreeNode<T>* node_to, Complex<T>* coeffs, int power, int maxpower, int* binomcoeffs)
    {
        Complex<T> result;
        Complex<T> dz = (node_from->center - node_to->center) / node_from->half_length;

        if (power == 0)
        {
            for (int i = maxpower; i >= 1; i--)
            {
                result += node_from->multipole_coeffs[i];
                result /= -dz;
            }
            result += node_from->multipole_coeffs[0] * (-dz).log();
            /******************************************************************************
            include factor from true node size when calculating M2L in units of node length
            ******************************************************************************/
            result += node_from->multipole_coeffs[0] * log(node_from->half_length);
        }
        else
        {
            for (int i = maxpower; i >= 1; i--)
            {
                result += node_from->multipole_coeffs[i] * get_binomial_coeff(binomcoeffs, power + i - 1, i - 1);
                result /= -dz;
            }
            result -= node_from->multipole_coeffs[0] / power;
            for (int i = 0; i < power; i++)
            {
                result /= dz;
            }
        }

        coeffs[power] = result;
    }

    /******************************************************************************
    calculate the local coefficients for a given power from the multipole
    coefficients of far nodes

    \param nodes -- pointer to tree
    \param numnodes -- number of nodes in the tree
    \param power -- maximum expansion order to find the local coefficients of
    \param binomcoeffs -- pointer to array of binomial coefficients
    ******************************************************************************/
    template <typename T>
    __global__ void calculate_M2L_coeffs_kernel(TreeNode<T>* nodes, int numnodes, int power, int* binomcoeffs)
    {
        /******************************************************************************
        each block is a node
        each thread calculates a local coefficient in the x direction for a member of
         the interaction list in the y direction
        ******************************************************************************/
        int node_index = blockIdx.x;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;
        int y_index = threadIdx.y;
        int y_stride = blockDim.y;

        extern __shared__ Complex<T> coeffs[];

        if (node_index < numnodes)
        {
            TreeNode<T>* node = &nodes[node_index];

            for (int i = y_index; i < node->numinterlist; i += y_stride)
            {
                for (int j = x_index; j <= power; j += x_stride)
                {
                    calculate_M2L_coeff(node->interactionlist[i], node, &coeffs[(power + 1) * i], j, power, binomcoeffs);
                }
            }
            __syncthreads();
            /******************************************************************************
            finally, add the local coefficients from the interaction list onto the node
            this must be sequentially carried out by one thread
            ******************************************************************************/
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                for (int i = 0; i < node->numinterlist; i++)
                {
                    node->add_local_coeffs(&coeffs[(power + 1) * i], power);
                }
            }
        }
    }

}

