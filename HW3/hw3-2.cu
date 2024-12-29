#include <cuda.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>

#define DEV_NO 0
#define FW_B 76
#define TILE 4
#define CUDA_B FW_B / TILE

__device__ const int INF = ((1 << 30) - 1);

int N, n, m;
int* h_D;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");

    fread(&N, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    n = N + (FW_B - (N % FW_B));
    h_D = (int*)malloc(n * n * sizeof(int));

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                h_D[i * n + j] = 0;
            } else {
                h_D[i * n + j] = INF;
            }
        }
    }

    int* edges = (int*)malloc(m * 3 * sizeof(int));

    fread(edges, sizeof(int), m * 3, file);
    fclose(file);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        h_D[edges[i * 3] * n + edges[i * 3 + 1]] = edges[i * 3 + 2];
    }

    free(edges);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < N; ++i) {
        fwrite(&h_D[i * n], sizeof(int), N, outfile);
    }
    fclose(outfile);
}

__global__ void phase1_kernel(int* __restrict__ d_D, int n, int r) {
    __shared__ int s_D[FW_B][FW_B];

    const int start = FW_B * r;

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = start + i;
            int global_j = start + j;

            s_D[i][j] = d_D[global_i * n + global_j];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < FW_B; k++) {
        #pragma unroll
        for (int bi = 0; bi < TILE; bi++) {
            #pragma unroll
            for (int bj = 0; bj < TILE; bj++) {
                int i = threadIdx.y + bi * CUDA_B;
                int j = threadIdx.x + bj * CUDA_B;

                s_D[i][j] = min(s_D[i][j], s_D[i][k] + s_D[k][j]);
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = start + i;
            int global_j = start + j;

            d_D[global_i * n + global_j] = s_D[i][j];
        }
    }
}

__global__ void phase2_row_kernel(int* __restrict__ d_D, int n, int r) {
    __shared__ int s_pivot[FW_B][FW_B];
    __shared__ int s_row[FW_B][FW_B];

    const int start = FW_B * r;

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            s_pivot[i][j] = d_D[(start + i) * n + start + j];

            int global_i = blockIdx.x * FW_B + i;
            int global_j = start + j;

            s_row[i][j] = d_D[global_i * n + global_j];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < FW_B; k++) {
        #pragma unroll
        for (int bi = 0; bi < TILE; bi++) {
            #pragma unroll
            for (int bj = 0; bj < TILE; bj++) {
                int i = threadIdx.y + bi * CUDA_B;
                int j = threadIdx.x + bj * CUDA_B;

                s_row[i][j] = min(s_row[i][j], s_row[i][k] + s_pivot[k][j]);
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = blockIdx.x * FW_B + i;
            int global_j = start + j;

            d_D[global_i * n + global_j] = s_row[i][j];
        }
    }
}

__global__ void phase2_col_kernel(int* __restrict__ d_D, int n, int r) {
    __shared__ int s_pivot[FW_B][FW_B];
    __shared__ int s_col[FW_B][FW_B];

    const int start = FW_B * r;

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            s_pivot[i][j] = d_D[(start + i) * n + start + j];

            int global_i = start + i;
            int global_j = blockIdx.x * FW_B + j;

            s_col[i][j] = d_D[global_i * n + global_j];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < FW_B; k++) {
        #pragma unroll
        for (int bi = 0; bi < TILE; bi++) {
            #pragma unroll
            for (int bj = 0; bj < TILE; bj++) {
                int i = threadIdx.y + bi * CUDA_B;
                int j = threadIdx.x + bj * CUDA_B;

                s_col[i][j] = min(s_col[i][j], s_pivot[i][k] + s_col[k][j]);
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = start + i;
            int global_j = blockIdx.x * FW_B + j;

            d_D[global_i * n + global_j] = s_col[i][j];
        }
    }
}

__global__ void phase3_kernel(int* __restrict__ d_D, int n, int r) {
    __shared__ int s_row[FW_B][FW_B];
    __shared__ int s_col[FW_B][FW_B];

    register int result[TILE][TILE];

    const int start = FW_B * r;

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = blockIdx.y * FW_B + i;
            int global_j = blockIdx.x * FW_B + j;

            result[bi][bj] = d_D[global_i * n + global_j];

            s_row[i][j] = d_D[global_i * n + start + j];
            s_col[i][j] = d_D[(start + i) * n + global_j];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < FW_B; k++) {
        #pragma unroll
        for (int bi = 0; bi < TILE; bi++) {
            #pragma unroll
            for (int bj = 0; bj < TILE; bj++) {
                int i = threadIdx.y + bi * CUDA_B;
                int j = threadIdx.x + bj * CUDA_B;

                result[bi][bj] = min(result[bi][bj], s_row[i][k] + s_col[k][j]);
            }
        }
    }

    #pragma unroll
    for (int bi = 0; bi < TILE; bi++) {
        #pragma unroll
        for (int bj = 0; bj < TILE; bj++) {
            int i = threadIdx.y + bi * CUDA_B;
            int j = threadIdx.x + bj * CUDA_B;

            int global_i = blockIdx.y * FW_B + i;
            int global_j = blockIdx.x * FW_B + j;

            d_D[global_i * n + global_j] = result[bi][bj];
        }
    }
}

void blocked_floyd_warshall() {
    int* d_D;
    cudaStream_t stream1, stream2, stream3;

    cudaMalloc(&d_D, n * n * sizeof(int));
    cudaMemcpy(d_D, h_D, n * n * sizeof(int), cudaMemcpyHostToDevice);

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    const int round = n / FW_B;
    dim3 grid(round, round);
    dim3 block(CUDA_B, CUDA_B);

    for (int r = 0; r < round; ++r) {
        phase1_kernel<<<1, block, 0, stream1>>>(d_D, n, r);
        cudaStreamSynchronize(stream1);

        phase2_row_kernel<<<round, block, 0, stream2>>>(d_D, n, r);
        phase2_col_kernel<<<round, block, 0, stream3>>>(d_D, n, r);

        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);

        phase3_kernel<<<grid, block, 0, stream1>>>(d_D, n, r);
        cudaStreamSynchronize(stream1);
    }

    cudaMemcpy(h_D, d_D, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaFree(d_D);
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;

    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    omp_set_num_threads(CPU_COUNT(&cpu_set));

    cudaSetDevice(DEV_NO);

    input(argv[1]);

    blocked_floyd_warshall();

    output(argv[2]);

    free(h_D);
    return 0;
}