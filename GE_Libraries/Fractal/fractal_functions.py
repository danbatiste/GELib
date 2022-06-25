from cmath import *
import cupy as cp



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void mandelbrot_kernel(const complex<float>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<float> z = 0;
        int i = 0;
        while (i < iter) {
            z = z*z + c[tid];
            i++;
            if (abs(z) > 2) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
mandelbrot_kernel = cp.RawKernel(code, 'mandelbrot_kernel')



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void mandelbrot_double_kernel(const complex<double>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<double> z = 0;
        int i = 0;
        while (i < iter) {
            z = z*z + c[tid];
            i++;
            if (abs(z) > 2) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
mandelbrot_double_kernel = cp.RawKernel(code, 'mandelbrot_double_kernel')



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void mandelbrot_long_double_kernel(const complex<long double>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<long double> z = 0;
        int i = 0;
        while (i < iter) {
            z = z*z + c[tid];
            i++;
            if (abs(z) > 2) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
mandelbrot_long_double_kernel = cp.RawKernel(code, 'mandelbrot_long_double_kernel')



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void timebrot_kernel(const complex<double>* c, int* output_fractal_values,
                       const double t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<double> I = -1.0;
        I = sqrt(I);
        complex<double> z = 2.0*sin(3.1415926535*t/2);// - 2.0*cos(3.1415926535*t/2)*I;
        int i = 0;
        while (i < iter) {
            z = z*z + c[tid]; //pow(z, 1.99+1600*t*sqrt(t)) + pow(c[tid], 1.0/(1+t*t));
            i++;
            if (abs(z) > 2) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
timebrot_kernel = cp.RawKernel(code, 'timebrot_kernel')



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void spikybrot_kernel(const complex<float>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<float> z = 0;
        int i = 0;
        while (i < iter) {
            z = (z*z*z + 4.0f*z*z + z + c[tid]);
            i++;
            if (abs(z) > (4.0f)) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
spikybrot_kernel = cp.RawKernel(code, 'spikybrot_kernel')


code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void star_kernel(const complex<float>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<float> z = 0;
        int i = 0;
        complex<float> z_denominator;
        while (i < iter) {
            z_denominator = (2.0f*z*z + 8.0f*z*z + 1.0f);
            if (z_denominator == 0.0f) {
                z_denominator = 1.0f;
            }
            z = (z*z*z + 4.0f*z*z + z + c[tid])/z_denominator;
            i++;
            if (abs(z) > (4.0f)) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
star_kernel = cp.RawKernel(code, 'star_kernel')



code = r'''
#include <cupy/complex.cuh>
extern "C" __global__
void x_kernel(const complex<float>* c, int* output_fractal_values,
                       const float t, const int iter, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        complex<float> z = 0;
        int i = 0;
        complex<float> z_denominator;
        while (i < iter) {
            z_denominator = (3.0f*z*z + 8.0f*z + 1.0f);
            if (z_denominator == 0.0f) {
                z_denominator = 1.0f;
            }
            z = (z*z*z + 4.0f*z*z + z + c[tid])/z_denominator;
            i++;
            if (abs(z) > (4.0f)) {
                output_fractal_values[tid] = i;
                break;
            }
        }
        if (i == iter) {
            output_fractal_values[tid] = -1;
        }
    }
}
'''
x_kernel = cp.RawKernel(code, 'x_kernel')