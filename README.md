# Sinter

A C++17 tensor compiler for custom GPU workloads. It lets you describe tensor computation graphs in a small text IR, checks that the graph and shapes are valid, and lowers the result toward optimized CUDA kernels. The goal is to make tensor programs easier to build, inspect, and optimize without writing every GPU kernel by hand.