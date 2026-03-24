#include <chrono>
#include <iostream>

extern "C" float kernel();

int main(int argc, char* argv[]) {
    if (argc != 3) std::cout << "error: missing arguments! `./out warmup runs`";

    const int warmup = std::stoi(argv[1]);
    const int runs = std::stoi(argv[2]);

    for (int i = 0; i < warmup; i++) kernel();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < runs; i++) kernel();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;

    std::cout << "total time: " << diff.count() << " s\n";
    std::cout << "avg time: " << diff.count() / runs << " s\n";
}
