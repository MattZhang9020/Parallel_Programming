#include <cuda.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PAD 1

void input(char *input_filename);
void output(char *output_filename);
void flash_attention();

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    flash_attention();

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(O);

    fclose(file);
}

__global__ void flash_attention_kernel(
    float *__restrict__ q, float *__restrict__ k, float *__restrict__ v, float *__restrict__ o,
    const float scalar,
    const int N, const int d,
    const int bc, const int br,
    const int tc, const int tr) {
    int tx = threadIdx.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int num_threads = blockDim.x;
    int batch_row = bx * br + tx;

    int qkvo_offset = by * N * d;

    extern __shared__ float sram[]; 
    float *const kj = sram;
    float *const vj = &sram[bc * (d + PAD)];
    float *const qi = &sram[bc * (d + PAD) * 2];
    float *const spij = &sram[bc * (d + PAD) * 2 + br * (d + PAD)];

    const int offset_si = tx * bc;

    if (batch_row < N) {
        float4 *q4 = reinterpret_cast<float4 *>(&q[qkvo_offset + batch_row * d]);

        float4 *shared_q4 = reinterpret_cast<float4 *>(&qi[tx * (d + PAD)]);

        #pragma unroll
        for (int x = 0; x < d / 4; x++) {
            shared_q4[x] = q4[x];
        }

        float li = 0.0f;
        float mi = -INFINITY;

        #pragma unroll
        for (int j = 0; j < tc; j++) {
            #pragma unroll
            for (int y = tx; y < bc; y += num_threads) {
                int global_col = j * bc + y;
                if (global_col < N) {
                    float4 *k4 = reinterpret_cast<float4 *>(&k[qkvo_offset + global_col * d]);
                    float4 *v4 = reinterpret_cast<float4 *>(&v[qkvo_offset + global_col * d]);

                    float4 *shared_k4 = reinterpret_cast<float4 *>(&kj[y * (d + PAD)]);
                    float4 *shared_v4 = reinterpret_cast<float4 *>(&vj[y * (d + PAD)]);

                    #pragma unroll
                    for (int x = 0; x < d / 4; x++) {
                        shared_k4[x] = k4[x];
                        shared_v4[x] = v4[x];
                    }
                }
            }
            __syncthreads();

            const int num_cols = fminf(bc, N - (bc * j));
            float local_max = -INFINITY;

            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                float qk_sum = 0.0f;
                float4 *qi_vec = reinterpret_cast<float4 *>(&qi[tx * (d + PAD)]);
                float4 *kj_vec = reinterpret_cast<float4 *>(&kj[c * (d + PAD)]);

                #pragma unroll
                for (int x = 0; x < d / 4; x++) {
                    float4 q_val = qi_vec[x];
                    float4 k_val = kj_vec[x];
                    qk_sum += q_val.x * k_val.x + q_val.y * k_val.y + q_val.z * k_val.z + q_val.w * k_val.w;
                }

                spij[offset_si + c] = scalar * qk_sum;
                local_max = fmaxf(local_max, spij[offset_si + c]);
            }

            float mi_tilde = __shfl_sync(0xFFFFFFFF, local_max, 0);
            float li_tilde = 0.0f;

            #pragma unroll
            for (int c = 0; c < num_cols; c++) {
                spij[offset_si + c] = __expf(spij[offset_si + c] - mi_tilde);
                li_tilde += spij[offset_si + c];
            }

            float mi_new = fmaxf(mi, mi_tilde);
            float li_new = __expf(mi - mi_new) * li + __expf(mi_tilde - mi_new) * li_tilde;

            #pragma unroll
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;

                #pragma unroll
                for (int c = 0; c < num_cols; c++) {
                    pv += spij[offset_si + c] * vj[c * (d + PAD) + x];
                }

                o[qkvo_offset + batch_row * d + x] = (1 / li_new) * ((li * __expf(mi - mi_new) * o[qkvo_offset + batch_row * d + x]) + __expf(mi_tilde - mi_new) * pv);
            }

            li = li_new;
            mi = mi_new;
            __syncthreads();
        }
    }
}

void flash_attention() {
    float *d_q, *d_k, *d_v, *d_o;

    size_t qkvo_size = B * N * d * sizeof(float);

    cudaMalloc(&d_q, qkvo_size);
    cudaMalloc(&d_k, qkvo_size);
    cudaMalloc(&d_v, qkvo_size);
    cudaMalloc(&d_o, qkvo_size);

    cudaMemcpy(d_q, Q, qkvo_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, K, qkvo_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, V, qkvo_size, cudaMemcpyHostToDevice);

    free(Q);
    free(K);
    free(V);
    
    const int bc = 11;
    const int br = 128;

    const int tc = ceil(float(N) / bc);
    const int tr = ceil(float(N) / br);

    dim3 grid((N + br - 1) / br, B);
    dim3 block(br);

    size_t shared_mem_size = ((bc * (d + PAD) * 2) + (br * (d + PAD)) + (bc * br)) * sizeof(float);

    const float scalar = 1.0f / sqrt(d);

    flash_attention_kernel<<<grid, block, shared_mem_size>>>(d_q, d_k, d_v, d_o, scalar, N, d, bc, br, tc, tr);

    cudaMemcpy(O, d_o, qkvo_size, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
}
