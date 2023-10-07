#pragma once


/******************************************************************************
calculate the binomial coefficients up to a maximum order
assumes this function will be called once, otherwise the recursion is redundant

\param binom_coeffs -- pointer to array to hold coefficients
\param n -- maximum order to find the coefficients for
******************************************************************************/
void calculate_binomial_coeffs(int* binom_coeffs, int n)
{
	/******************************************************************************
	recursion limit
	******************************************************************************/
	if (n == 0)
	{
		binom_coeffs[0] = 1;
		return;
	}

	/******************************************************************************
	rows of Pascal's triangl
	******************************************************************************/
	int row_start = n * (n + 1) / 2;
	int prev_row_start = (n - 1) * n / 2;

	calculate_binomial_coeffs(binom_coeffs, n - 1);
	for (int i = 0; i <= n; i++)
	{
		if (i == 0 || i == n)
		{
			binom_coeffs[row_start + i] = 1;
		}
		else
		{
			binom_coeffs[row_start + i] = binom_coeffs[prev_row_start + i - 1] + binom_coeffs[prev_row_start + i];
		}
	}
}

/******************************************************************************
return the value of the binomial coefficient (n, k)

\param binom_coeffs -- pointer to array holding coefficients
\param n -- maximum order for the coefficients
\param k -- order to find the coefficient of
******************************************************************************/
__host__ __device__ int get_binomial_coeff(int* binom_coeffs, int n, int k)
{
	/******************************************************************************
	binomial coefficient only defined if n >= k
	******************************************************************************/
	if (n < k)
	{
		return 0;
	}
	int row_start = n * (n + 1) / 2;
	return binom_coeffs[row_start + k];
}

