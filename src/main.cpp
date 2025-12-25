#include <iostream>
#include <tensor.hpp>

int main() {
	NTensor<int> tensor({3, 2, 2}, 0, NTensorConfig{0});

	tensor.index({0, 0, 1}) = 1;
	tensor.index({0, 1, 1}) = 1;


	return 0;
}