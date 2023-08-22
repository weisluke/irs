#pragma once

/*assumes this function will be called once, otherwise the recursion is redundant*/
void calculate_binomial_coeffs(int* binom_coeffs, int n)
{
	/*recursion limit*/
	if (n == 0)
	{
		binom_coeffs[0] = 1;
		return;
	}

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

__host__ __device__ int get_binomial_coeff(int* binom_coeffs, int n, int k)
{
	int row_start = n * (n + 1) / 2;
	return binom_coeffs[row_start + k];
}