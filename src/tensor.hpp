#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <initializer_list>

typedef struct NTensorConfig {
	size_t strassen_threshold = 48;
} NTensorConfig;

template<typename T>
class NTensor {
public:
    NTensor(std::initializer_list<size_t> shape, T fill, NTensorConfig cfg);

    T* data();
    const size_t* shape() const;
    size_t ndim() const;

private:
    NTensorConfig config_;

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::unique_ptr<T[]> data_;

    size_t size_;
};



class VTensor<T> {
private:
	T* data_;
	const size_t* shape_;
	const size_t* stride_;
	size_t ndim_;
}

#endif TENSOR_HPP