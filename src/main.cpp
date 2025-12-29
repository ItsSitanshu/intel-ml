#include <iostream>
#include <tensor.hpp>

int main() {
	size_t N = 4;
	NTensor<float> tensor({N, N}, 0, NTensorConfig{0});
	NTensor<float> tensor2({N, N}, 0, NTensorConfig{0});

	for (size_t i = 0; i < N; i++) {
    	for (size_t j = 0; j < N; j++) {
    		tensor.index({i, j}) = (float)(2.125 * (i + j) + (i/ (j + 1)));
    		tensor2.index({i, j}) = (float)(i + j);
			}
	}
	tensor.matmul(2.0);
	tensor.matmul(tensor2);
	tensor.print_flat();

	VTensor<float> view = tensor.slice(0, 2, 0, 2);

	view.print_flat();

	return 0;
}		