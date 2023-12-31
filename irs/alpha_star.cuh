#pragma once

#include "complex.cuh"
#include "star.cuh"
#include "tree_node.cuh"


/******************************************************************************
calculate the deflection angle due to nearby stars for a node

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- node within which to calculate the deflection angle

\return alpha_star = theta^2 * sum(m_i / (z - z_i)_bar)
******************************************************************************/
template <typename T>
__device__ Complex<T> alpha_star(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> a_star_bar;

	/******************************************************************************
	theta^2 * sum(m_i / (z - z_i))
	******************************************************************************/
	for (int i = 0; i < node->numstars; i++)
	{
		a_star_bar += stars[node->stars + i].mass / (z - stars[node->stars + i].position);
	}
	for (int j = 0; j < node->numneighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			a_star_bar += stars[neighbor->stars + i].mass / (z - stars[neighbor->stars + i].position);
		}
	}
	a_star_bar *= (theta * theta);

	return a_star_bar.conj();
}

/******************************************************************************
calculate the derivative of the deflection angle due to nearby stars for a
node with respect to zbar

\param z -- complex image plane position
\param theta -- size of the Einstein radius of a unit mass point lens
\param stars -- pointer to array of point mass lenses
\param node -- node within which to calculate the deflection angle

\return d_alpha_star_d_zbar = -theta^2 * sum(m_i / (z - z_i)_bar^2)
******************************************************************************/
template <typename T>
__device__ Complex<T> d_alpha_star_d_zbar(Complex<T> z, T theta, star<T>* stars, TreeNode<T>* node)
{
	Complex<T> d_a_star_bar_d_z;

	/******************************************************************************
	-theta^2 * sum(m_i / (z - z_i)^2)
	******************************************************************************/
	for (int i = 0; i < node->numstars; i++)
	{
		d_a_star_bar_d_z += stars[node->stars + i].mass / ((z - stars[node->stars + i].position) * (z - stars[node->stars + i].position));
	}
	for (int j = 0; j < node->numneighbors; j++)
	{
		TreeNode<T>* neighbor = node->neighbors[j];
		for (int i = 0; i < neighbor->numstars; i++)
		{
			d_a_star_bar_d_z += stars[neighbor->stars + i].mass / ((z - stars[neighbor->stars + i].position) * (z - stars[neighbor->stars + i].position));
		}
	}
	d_a_star_bar_d_z *= -(theta * theta);

	return d_a_star_bar_d_z.conj();
}

