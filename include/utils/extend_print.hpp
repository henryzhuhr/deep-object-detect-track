#ifndef __UTILS__EXTEND_PRINT_HPP__
#define __UTILS__EXTEND_PRINT_HPP__

#include <iostream>
#include <string>
#include <vector>

class Utils {
public:
    template <typename T>
    static void print_vector(const std::vector<T>& vec, std::string split = " ") {
        std::cout << "[";
        for (auto i = 0; i < vec.size() - 1; i++) {
            std::cout << vec[i] << split;
        }
        std::cout << vec[vec.size() - 1] << "]" << std::endl;
    };
};

#endif  // __UTILS__EXTEND_PRINT_HPP__