#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std::chrono;

const int INF = ((1 << 30) - 1);
const int B = 512;

int n, m;

static int D[50010][50010];

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                D[i][j] = 0;
            } else {
                D[i][j] = INF;
            }
        }
    }

    int pair[3];

    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        D[pair[0]][pair[1]] = pair[2];
    }
    
    fclose(file);
}

inline int ceil(int a, int b) { return (a + b - 1) / b; }

void cal(int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (D[i][k] + D[k][j] < D[i][j]) {
                            D[i][j] = D[i][k] + D[k][j];
                        }
                    }
                }
            }
        }
    }
}

void blocked_floyd_warshall() {
    int round = ceil(n, B);

    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(r, r, r, 1, 1);

        /* Phase 2*/
        cal(r, r, 0, r, 1);
        cal(r, r, r + 1, round - r - 1, 1);
        cal(r, 0, r, 1, r);
        cal(r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        cal(r, 0, 0, r, r);
        cal(r, 0, r + 1, round - r - 1, r);
        cal(r, r + 1, 0, r, round - r - 1);
        cal(r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (D[i][j] >= INF) D[i][j] = INF;
        }
        fwrite(D[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {
    auto start = high_resolution_clock::now();

    input(argv[1]);

    blocked_floyd_warshall();

    output(argv[2]);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<seconds>(stop - start);

    printf("%d\n", duration.count());

    return 0;
}