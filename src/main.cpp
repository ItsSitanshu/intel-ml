#include <iostream>
#include <tensor.hpp>

int main() {

	size_t N = 4;
	NTensor<float> tensor({N, N}, 0, NTensorConfig{0});

	for (size_t i = 0; i < N; i++) {
    	for (size_t j = 0; j < N; j++)
    		tensor.index({i, j}) = (float)(2.125 * (i + j) + (i/ (j + 1)));
	}

    tensor.print_flat();

	return 0;
}