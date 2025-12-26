#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <log.hpp>

#include <initializer_list>
#include <vector>
#include <memory>

typedef struct NTensorConfig {
	size_t strassen_threshold = 48;
} NTensorConfig;

template<typename T = float>
class NTensor {
public:
    NTensor(std::initializer_list<size_t> shape, T fill, NTensorConfig cfg) {
        /**
         * @brief Initialize memory for a tensor object
         *
         * @param (initalizer_list<size_t>) shape: {highest order of abstraction -> scalar} 
         * @param (T) fill: default value of all scalars 
         * @param (struct NTensorConfig) cfg: configuration settings 
         *     ~ (size_t) strassen_threshold: limit .matmul() uses before switching to strassen's algorithm, default = 48 
        */

        shape_ = shape;
        config_ = cfg;

        calculate_size();
        calculate_stride();

        data_.resize(size_, fill);
    }

    T& index(std::initializer_list<size_t> pos) {
        /**
         * @brief Index a scalar at provided position 
         *
         * @param (initalizer_list<size_t>) pos: {highest order of abstraction -> scalar}
         * 
         * @reutrn (T) value at provided position
        */

        if (pos.size() != ndim_) {
            _log::log_fatal(
                "Provided position %s has %zu dimensions; tensor requires %zu @ %p",
                _log::format_init_list(pos).c_str(), pos.size(), ndim_, static_cast<void*>(this)
            );
        }

        size_t flat_pos = 0;
        int i = 0;

        for (size_t p : pos) {
            flat_pos += p * stride_[i];
        ++i;
        }

        if (flat_pos >= size_) {
            _log::log_fatal("Provided postition %s exceeds bounds of tensor @ %p",
                _log::format_init_list(pos).c_str(), static_cast<void*>(this));
        }

        return data_[flat_pos];
    }

    NTensor operator+(const NTensor& t) {
        check_size_eq(t);

        NTensor<T> out(shape_, 0, config_);

        for (size_t i = 0; i < size_; i++) {
            out.data_[i] = data_[i] + t.data_[i];
        }

        return out;
    }

    NTensor operator-(const NTensor& t) {
        check_size_eq(t);

        NTensor<T> out(shape_, 0, config_);

        for (size_t i = 0; i < size_; i++) {
            out.data_[i] = data_[i] - t.data_[i];
        }

        return out;
    }
    
    NTensor operator*(T scalar) {
        NTensor<T> out(shape_);

        for (size_t i = 0; i < size_; i++) {
            out.data_[i] = data_[i] * scalar;
        }

        return out;
    }

    NTensor flatten() {
        NTensor<T> out({1, size_});
        out.data_ = data_;
        
        return out;
    }

    T sum() { 
        T out;
        for (size_t i = 0; i < size_; i++) {
            out += data_[i];
        }
        return out;
    }

    T mean() {
        T out = (T)(sum() / size_);
        return out;
    }

    T median() {  };

    T min() {
        T out = data_[0];

        for (size_t i = 1; i < size_; i++) {
            out = (data_[i] < out) ? data_[i] : out;
        }

        return out;
    }  
    
    T max() {
        T out = data_[0];

        for (size_t i = 1; i < size_; i++) {
            out = (data_[i] > out) ? data_[i] : out;
        }

        return out;
    }

    void print_flat() {
        std::cout << "NTensor[size=" << size_ << ", data=";
        for (size_t i = 0; i < size_; i++) {
            std::cout << data_[i];
            if (i != size_) std::cout << ", "; 
        }
        std::cout << "]" << std::endl;
     
    }

    T* data() { return data_.data(); }
    const size_t* shape() { return shape_.data(); };
    size_t ndim() { return ndim_; };
private:
    NTensorConfig config_;

    std::vector<size_t> shape_;
    std::vector<size_t> stride_;
    std::vector<T> data_;

    size_t size_ = 1;
    size_t ndim_;

    void calculate_size() {
        ndim_ = shape_.size();
        for (int i = 0; i < ndim_; ++i)
            size_ *= shape_[i];
    }

    void calculate_stride() {
        stride_.resize(ndim_);

        stride_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i)
            stride_[i] = shape_[i + 1] * stride_[i + 1];
    }

    void check_size_eq(const NTensor& t) {
        if (shape_ != t.shape_) {
            std::ostringstream oss;
            oss << "Cannot operate on tensors: shapes differ. "
                << "LHS shape=[";
            for (size_t i = 0; i < shape_.size(); ++i) {
                oss << shape_[i];
                if (i + 1 < shape_.size()) oss << ", ";
            }
            oss << "] RHS shape=[";
            for (size_t i = 0; i < t.shape_.size(); ++i) {
                oss << t.shape_[i];
                if (i + 1 < t.shape_.size()) oss << ", ";
            }
            oss << "]";

            throw std::runtime_error(oss.str());
        }
    }     
};


template<typename T = float>
class VTensor {
private:
	T* data_;
	const size_t* shape_;
	const size_t* stride_;
	size_t ndim_;
};

#endif // TENSOR_HPP