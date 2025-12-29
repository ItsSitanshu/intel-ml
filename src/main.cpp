#include <iostream>
#include <tensor.hpp>

int main() {
	size_t N = 64;
	NTensor<int> T1({N, N}, 0, NTensorConfig{200000000});
	NTensor<int> T2({N, N}, 0, NTensorConfig{200000000});
	NTensor<int> T3({N, N}, 0, NTensorConfig{200000000});

	for (size_t i = 0; i < N; ++i) {
		for (size_t j = 0; j < N; ++j) {
			int f = static_cast<int>(i + j);
			int s = static_cast<int>(i - j);

			T1.index({i, j}) = f;  // top-left = 0+0=0, etc.
			T2.index({i, j}) = s;  // top-left = 0-0=0, etc.
		}
	}

    std::cout << T1.sum() << std::endl;
    std::cout << T2.sum() << std::endl;


    T3 = T1.matmul(T2);

    std::cout << T3.sum() << std::endl;

	return 0;
}		