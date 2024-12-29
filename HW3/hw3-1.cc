#include <omp.h>
#include <sched.h>
#include <smmintrin.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>

const int INF = ((1 << 30) - 1);
const int B = 64;

int n, m;

static int D[50010][50010];

void input(char* infile) {
    FILE* file = fopen(infile, "rb");

    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                D[i][j] = 0;
            } else {
                D[i][j] = INF;
            }
        }
    }

    int* edges = (int*)malloc(m * 3 * sizeof(int));

    fread(edges, sizeof(int), m * 3, file);
    fclose(file);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        D[edges[i * 3]][edges[i * 3 + 1]] = edges[i * 3 + 2];
    }

    free(edges);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(D[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void process_block_sse(int block_internal_start_x, int block_internal_end_x, int block_internal_start_y, int block_internal_end_y, int k) {
    const int SIMD_WIDTH = 4;
    const int UNROLL = 4;

    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        __m128i v_d_ik = _mm_set1_epi32(D[i][k]);

        int j;
        for (j = block_internal_start_y; j + SIMD_WIDTH * UNROLL <= block_internal_end_y; j += SIMD_WIDTH * UNROLL) {
            __m128i v0_d_kj = _mm_loadu_si128((__m128i*)&D[k][j]);
            __m128i v1_d_kj = _mm_loadu_si128((__m128i*)&D[k][j + SIMD_WIDTH]);
            __m128i v2_d_kj = _mm_loadu_si128((__m128i*)&D[k][j + SIMD_WIDTH * 2]);
            __m128i v3_d_kj = _mm_loadu_si128((__m128i*)&D[k][j + SIMD_WIDTH * 3]);

            __m128i v0_d_ij = _mm_loadu_si128((__m128i*)&D[i][j]);
            __m128i v1_d_ij = _mm_loadu_si128((__m128i*)&D[i][j + SIMD_WIDTH]);
            __m128i v2_d_ij = _mm_loadu_si128((__m128i*)&D[i][j + SIMD_WIDTH * 2]);
            __m128i v3_d_ij = _mm_loadu_si128((__m128i*)&D[i][j + SIMD_WIDTH * 3]);

            __m128i v0_sum = _mm_add_epi32(v_d_ik, v0_d_kj);
            __m128i v1_sum = _mm_add_epi32(v_d_ik, v1_d_kj);
            __m128i v2_sum = _mm_add_epi32(v_d_ik, v2_d_kj);
            __m128i v3_sum = _mm_add_epi32(v_d_ik, v3_d_kj);

            v0_sum = _mm_min_epi32(v0_sum, v0_d_ij);
            v1_sum = _mm_min_epi32(v1_sum, v1_d_ij);
            v2_sum = _mm_min_epi32(v2_sum, v2_d_ij);
            v3_sum = _mm_min_epi32(v3_sum, v3_d_ij);

            _mm_storeu_si128((__m128i*)&D[i][j], v0_sum);
            _mm_storeu_si128((__m128i*)&D[i][j + SIMD_WIDTH], v1_sum);
            _mm_storeu_si128((__m128i*)&D[i][j + SIMD_WIDTH * 2], v2_sum);
            _mm_storeu_si128((__m128i*)&D[i][j + SIMD_WIDTH * 3], v3_sum);
        }

        for (; j < block_internal_end_y; ++j) {
            int d_ik_kj = D[i][k] + D[k][j];
            D[i][j] = std::min(D[i][j], d_ik_kj);
        }
    }
}

void cal(int round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int k_start = round * B;
    int k_end = std::min((round + 1) * B, n);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = k_start; k < k_end; ++k) {
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;

                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                process_block_sse(block_internal_start_x, block_internal_end_x, block_internal_start_y, block_internal_end_y, k);
            }
        }
    }
}

void blocked_floyd_warshall() {
    int round = (n + B - 1) / B;

    for (int r = 0; r < round; ++r) {
        cal(r, r, r, 1, 1);

        cal(r, r, 0, r, 1);
        cal(r, r, r + 1, round - r - 1, 1);
        cal(r, 0, r, 1, r);
        cal(r, r + 1, r, 1, round - r - 1);

        cal(r, 0, 0, r, r);
        cal(r, 0, r + 1, round - r - 1, r);
        cal(r, r + 1, 0, r, round - r - 1);
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;

    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    omp_set_num_threads(CPU_COUNT(&cpu_set));

    input(argv[1]);

    blocked_floyd_warshall();

    output(argv[2]);

    return 0;
}