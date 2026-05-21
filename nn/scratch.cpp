#include <iostream>
#include <tuple>
#include <utility>

template <typename T> struct ArgGen;

template <> struct ArgGen<float*> {
    float* ptr;
    ArgGen() { ptr = new float[10]; }
    ~ArgGen() { delete[] ptr; }
    float* get() { return ptr; }
};

template <> struct ArgGen<float* __restrict> {
    float* ptr;
    ArgGen() { ptr = new float[10]; }
    ~ArgGen() { delete[] ptr; }
    float* __restrict get() { return ptr; }
};

template <> struct ArgGen<const float* __restrict> {
    float* ptr;
    ArgGen() { ptr = new float[10]; }
    ~ArgGen() { delete[] ptr; }
    const float* __restrict get() { return ptr; }
};

template<typename Func, typename... Args, size_t... Is>
void invoke_helper(Func f, std::tuple<ArgGen<Args>...>& tup, std::index_sequence<Is...>) {
    f(std::get<Is>(tup).get()...);
}

template<typename Ret, typename... Args>
void run_auto_benchmark(Ret(*func)(Args...)) {
    std::tuple<ArgGen<Args>...> args;
    invoke_helper(func, args, std::index_sequence_for<Args...>{});
}

void my_test_func(const float* __restrict a, float* __restrict b) {
    std::cout << "It works!\n";
}

int main() {
    run_auto_benchmark(&my_test_func);
    return 0;
}
