#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <log.hpp>

#include <initializer_list>
#include <vector>
#include <memory>

typedef struct NTensorConfig {
	size_t strassen_threshold;
} NTensorConfig;

template<typename T = float>
class VTensor {
public:
    T* data_;
    const size_t* shape_;
    const size_t* stride_;
    size_t size_;
    size_t ndim_;

    T& index(size_t i, size_t j) {
        return data_[i * stride_[0] + j * stride_[1]];
    }

    void print_flat() {
        if (ndim_ == 2) {
            std::cout << "VTensor[";
            for (size_t i = 0; i < shape_[0]; ++i) {
                for (size_t j = 0; j < shape_[1]; ++j) {
                    std::cout << data_[i * stride_[0] + j * stride_[1]];
                    if (i != shape_[0]-1 || j != shape_[1]-1) std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
        } else {
            // fallback for 1D
            std::cout << "VTensor[";
            for (size_t i = 0; i < size_; ++i) {
                std::cout << data_[i];
                if (i != size_-1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

};

template<typename T = float>
class NTensor {
public:

    NTensor(const std::vector<size_t>& shape, T fill, NTensorConfig cfg)
        : shape_(shape), config_(cfg)
    {
        /**
         * @brief Initialize memory for a tensor object
         *
         * @param (std::vector<size_t>) shape: {highest order of abstraction -> scalar} 
         * @param (T) fill: default value of all scalars 
         * @param (struct NTensorConfig) cfg: configuration settings 
         *     ~ (size_t) strassen_threshold: limit .matmul() uses before switching to strassen's algorithm, default = 48 
        */
        calculate_size();
        calculate_stride();
        data_.resize(size_, fill);
    }

    NTensor(std::initializer_list<size_t> shape, T fill, NTensorConfig cfg)
        : NTensor(std::vector<size_t>(shape), fill, cfg)
    {}

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

    VTensor<T> slice(size_t a, size_t b, size_t c, size_t d) {
        if (ndim_ > 2) {
            throw std::runtime_error("Splitting not supported for tensors > 2D");
        }

        VTensor<T> out;

        size_t rows = b - a;
        size_t cols = d - c;

        out.data_ = data_.data() + a * stride_[0] + c * stride_[1];
        out.size_ = rows * cols;

        // Allocate new stride array for the view
        out.stride_ = new size_t[2]{ stride_[0], stride_[1] };
        out.shape_  = new size_t[2]{ rows, cols };
        out.ndim_ = 2;

        return out;
    }


    NTensor add(const NTensor& t) {
        check_size_eq(t);

        NTensor<T> out(shape_, (T)0, config_);

        for (size_t i = 0; i < size_; ++i) {
            out.data_[i] = data_[i] + t.data_[i];
        }

        return out;
    }

    void add(VTensor<T>& a, VTensor<T>& b) {
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                // _log::log_message(_log::DEBUG, "%zu, %zu", i, j); 
                index({i, j}) = a.index(i, j) + b.index(i, j);
            }
        }

        // _log::log_message(_log::DEBUG, "shape = {%zu, %zu}", shape_[0], shape_[1]); 
    }

    NTensor sub(NTensor& t) {
        check_size_eq(t);

        NTensor<T> out(shape_, 0, config_);

        for (size_t i = 0; i < size_; ++i) {
            out.data_[i] = data_[i] - t.data_[i];
        }

        return out;
    }

    void sub(VTensor<T>& a, VTensor<T>& b) {
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                // _log::log_message(_log::DEBUG, "%zu, %zu", i, j); 
                index({i, j}) = a.index(i, j) - b.index(i, j);
            }
        }
    }

    void eq(VTensor<T>& view) {
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < shape_[1]; ++j) {
                index({i, j}) = view.index(i, j);
            }
        }
    }    

    void matmul(T scalar) {
        T p = (T)0;
        for (size_t i = 0; i < size_; ++i) {
            p = data_[i];
            data_[i] = p * scalar;
        }
    }

    NTensor<T> matmul(NTensor<T> t) {
        if (ndim_ == 2) {
            if (size_ < config_.strassen_threshold) {
                return static_matmul(t);
            }

            _log::log_message(_log::DEBUG, "Strassen!");
            
            return strassen_matmul(*this, t);
        }

        // implement for n_ > 2
    }  

    NTensor<T> static_matmul(NTensor<T> t) {    
        size_t rows = shape_[1];
        size_t cols = t.shape_[0];
        _log::log_message(_log::DEBUG, "%zu, %zu", rows, cols);

        NTensor<T> out = NTensor({rows, cols}, (T)0, config_);


        for (size_t row = 0; row < rows; row++) {
            size_t col_off = cols * row;
            for (size_t col = 0; col < cols; col++) {
                T val = 0;

                for (size_t i = 0; i < shape_[0]; i++) {
                    // std::cout << "out(" << col << "," << row << ") += cur(" << col << "," << i << ")" <<" * other(" << i << "," << row << ")" << std::endl;
                    // std::cout << "\t [^...] = " << index({col, i}) << "*" << t.index({i, row}) << "=" << index({col, i}) * t.index({i, row}) << std::endl;

                    val += data_[col_off + i] * t.data_[(i * cols) + col];
                }

                out.data_[col_off + col] = val;
            }
        }

        return out;
    }


    NTensor<T> strassen_matmul(NTensor<T> A, NTensor<T> B) {
        if (A.shape_[0] <= 4 && B.shape_[1] <= 4) {
            _log::log_message(_log::DEBUG, "Static within strassen!"); 
            return A.static_matmul(B);
        }

        VTensor<T> a, b, c, d, e, f, g, h;
        

        // split 
        std::tie(a, b, c, d) = strassen_split(A);
        std::tie(e, f, g, h) = strassen_split(B);

        NTensor<T> buf1({a.shape_[0], a.shape_[1]}, (T)0, config_);
        NTensor<T> buf2({a.shape_[0], a.shape_[1]}, (T)0, config_);
        
        // m1 = strassen(a + d, e + h) 
        buf1.add(a, d);
        buf2.sub(e, h);
        NTensor<T> m1 = strassen_matmul(buf1, buf2);
        
        // m2 = strassen(d, g - e)
        buf2.eq(d);
        buf2.sub(g, e);
        NTensor<T> m2 = strassen_matmul(buf1, buf2);

        // m3 = strassen(a + b, h)
        buf1.add(a, b);
        buf2.eq(h);
        NTensor<T> m3 = strassen_matmul(buf1, buf2);
        
        // m4 = strassen(b - d, g + h)
        buf1.sub(b, d); 
        buf2.add(g, h);
        NTensor<T> m4 = strassen_matmul(buf1, buf2);
        
        // m5 = strassen(a, f - h)
        buf1.eq(a);
        buf2.sub(f, h);
        NTensor<T> m5 = strassen_matmul(buf1, buf2);

        // m6 = strassen(c + d, e)
        buf1.add(c, d);
        buf2.eq(e);
        NTensor<T> m6 = strassen_matmul(buf1, buf2);

        // m7 = strassen(a - c, e + f)
        buf1.sub(a, c); 
        buf2.add(e, f);
        NTensor<T> m7 = strassen_matmul(buf1, buf2);
        
        NTensor<T> c11 = m1.add(m2).sub(m3).add(m4);
        NTensor<T> c12 = m5.add(m3);
        NTensor<T> c21 = m6.add(m2);
        NTensor<T> c22 = m5.add(m1).sub(m6).sub(m7); 

        NTensor<T> C = strassen_stack(c11, c12, c21, c22);

        return C;
    }

    NTensor<T> strassen_stack(const NTensor<T>& c11, const NTensor<T>& c12, const NTensor<T>& c21, const NTensor<T>& c22) {
        const size_t R = c11.shape_[0];
        const size_t C = c11.shape_[1];


        const size_t R2 = 2 * R;
        const size_t C2 = 2 * C;

        NTensor<T> out({R2, C2}, T(0), config_);

        T* outp       = out.data_.data();
        const T* p11  = c11.data_.data();
        const T* p12  = c12.data_.data();
        const T* p21  = c21.data_.data();
        const T* p22  = c22.data_.data();

        for (size_t i = 0; i < R; ++i) {
            T* out_top    = outp + i * C2;
            T* out_bottom = outp + (i + R) * C2;

            const T* a = p11 + i * C;
            const T* b = p12 + i * C;
            const T* c = p21 + i * C;
            const T* d = p22 + i * C;

            for (size_t j = 0; j < C; ++j) {
                out_top[j]       = a[j];
                out_top[j + C]   = b[j];
                out_bottom[j]     = c[j];
                out_bottom[j + C] = d[j];
            }
        }

        return out;
    }


    std::tuple<VTensor<T>, VTensor<T>, VTensor<T>, VTensor<T>> strassen_split(NTensor<T> t){
        size_t rows = t.shape_[0];
        size_t columns = t.shape_[1];

        size_t half_rows = rows / 2;
        size_t half_columns = columns / 2;

        // std::cout << half_rows << half_columns << std::endl;

        VTensor<T> a = t.slice(0, half_rows, 0, half_columns);
        VTensor<T> b = t.slice(0, half_rows, half_columns, columns);
        VTensor<T> c = t.slice(half_rows, rows, 0, half_columns);
        VTensor<T> d = t.slice(half_rows, rows, half_columns, columns);
        
        return std::tuple<VTensor<T>, VTensor<T>, VTensor<T>, VTensor<T>> {a, b, c, d};
    }

    NTensor flatten() {
        NTensor<T> out({1, size_});
        out.data_ = data_;
        
        return out;
    }

    T sum() { 
        T out = 0;
        for (size_t i = 0; i < size_; ++i) {
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

        for (size_t i = 1; i < size_; ++i) {
            out = (data_[i] < out) ? data_[i] : out;
        }

        return out;
    }  
    
    T max() {
        T out = data_[0];

        for (size_t i = 1; i < size_; ++i) {
            out = (data_[i] > out) ? data_[i] : out;
        }

        return out;
    }

    void print_flat() {
        std::cout << "NTensor[size=" << size_ << ", data=";
        for (size_t i = 0; i < size_; ++i) {
            std::cout << data_[i];
            if (i != size_) std::cout << ", "; 
        }
        std::cout << "]" << std::endl;
    }

    T* data() { return data_.data(); };
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

#endif // TENSOR_HPP