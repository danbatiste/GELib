from cmath import *
import cupy as cp

# Color functions
modulo_coloring_kernel = cp.RawKernel(r'''
extern "C" __global__
void modulo_coloring(const int* iterations, int* output_rgbx_ints, const float t, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id/index?
    if (tid < N) {
        if (iterations[tid] == -1.0) {
            output_rgbx_ints[tid] = 0;
        } else {
            int iteration_mod = 15;
            float shade = (float) (iterations[tid] % iteration_mod);
            shade = 1 - shade/((float) iteration_mod);

            int iteration_wave = iterations[tid]/iteration_mod;
            int colors = 10;
            float color = (float) (iteration_wave%colors) / colors;

            float huedeg = color*360;

            // HSV to RGB conversion formula from HSV and HSL wikipedia page
            // https://en.wikipedia.org/wiki/HSL_and_HSV#section=19
            float k_5 = fmod((5 + huedeg/60), 6);
            float k_3 = fmod((3 + huedeg/60), 6);
            float k_1 = fmod((1 + huedeg/60), 6);

            k_5 = min(k_5, 4.0 - k_5);
            //!(k_5<4-k_5)?4-k_5:k_5;
            k_5 = min(k_5, 1.0);
            //!(k_5<1)?1:k_5;
            k_5 = max(k_5, 0.0);
            //(k_5<0)?0:k_5;

            k_3 = min(k_3, 4.0 - k_3);
            k_3 = min(k_3, 1.0);
            k_3 = max(k_3, 0.0);

            k_1 = min(k_1, 4.0 - k_1);
            k_1 = min(k_1, 1.0);
            k_1 = max(k_1, 0.0);

            // k_3 = !(k_3<4-k_3)?4-k_3:k_3;
            // k_3 = !(k_3<1)?1:k_3;
            // k_3 = (k_3<0)?0:k_3;

            // k_1 = !(k_1<4-k_1)?4-k_1:k_1;
            // k_1 = !(k_1<1)?1:k_1;
            // k_1 = (k_1<0)?0:k_1;

            int R = round(shade*255*( 1 - k_5 ));
            int G = round(shade*255*( 1 - k_3 ));
            int B = round(shade*255*( 1 - k_1 ));

            output_rgbx_ints[tid] = (B << 16) | (G << 8) | R;
        }
    }
}
''', 'modulo_coloring') # adds two numbers together and outputs to y array


modulo_coloring_sqrt_shading_kernel = cp.RawKernel(r'''
extern "C" __global__
void modulo_coloring(const int* iterations, int* output_rgbx_ints, const float t, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // thread id/index?
    if (tid < N) {
        if (iterations[tid] == -1.0) {
            output_rgbx_ints[tid] = 0;
        } else {
            int iteration_mod = 15;
            float shade = (float) (iterations[tid] % iteration_mod);
            shade = 1 - shade/((float) iteration_mod);
            shade = sqrt(sqrt(shade));

            int iteration_wave = iterations[tid]/iteration_mod;
            int colors = 10;
            float color = (float) (iteration_wave%colors) / colors;

            float huedeg = color*360;

            // HSV to RGB conversion formula from HSV and HSL wikipedia page
            // https://en.wikipedia.org/wiki/HSL_and_HSV#section=19
            float k_5 = fmod((5 + huedeg/60), 6);
            float k_3 = fmod((3 + huedeg/60), 6);
            float k_1 = fmod((1 + huedeg/60), 6);

            k_5 = min(k_5, 4.0 - k_5);
            //!(k_5<4-k_5)?4-k_5:k_5;
            k_5 = min(k_5, 1.0);
            //!(k_5<1)?1:k_5;
            k_5 = max(k_5, 0.0);
            //(k_5<0)?0:k_5;

            k_3 = min(k_3, 4.0 - k_3);
            k_3 = min(k_3, 1.0);
            k_3 = max(k_3, 0.0);

            k_1 = min(k_1, 4.0 - k_1);
            k_1 = min(k_1, 1.0);
            k_1 = max(k_1, 0.0);

            // k_3 = !(k_3<4-k_3)?4-k_3:k_3;
            // k_3 = !(k_3<1)?1:k_3;
            // k_3 = (k_3<0)?0:k_3;

            // k_1 = !(k_1<4-k_1)?4-k_1:k_1;
            // k_1 = !(k_1<1)?1:k_1;
            // k_1 = (k_1<0)?0:k_1;

            int R = round(shade*255*( 1 - k_5 ));
            int G = round(shade*255*( 1 - k_3 ));
            int B = round(shade*255*( 1 - k_1 ));

            output_rgbx_ints[tid] = (B << 16) | (G << 8) | R;
        }
    }
}
''', 'modulo_coloring')
