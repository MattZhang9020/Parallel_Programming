#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);
void self_attention(float *q, float *k, float *v, float *o);

void QKDotAndScalar(float *out, float *q, float *k, float scalar);
void SoftMax(float *out, float *in);
void MulAttV(float *out, float *att, float *v);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }

    input(argv[1]);

    double start, end;
    start = getTimeStamp();

    for (int i = 0; i < B; i++) {
        self_attention(
            Q + (i * N * d), 
            K + (i * N * d), 
            V + (i * N * d), 
            O + (i * N * d)
        );
    }

    end = getTimeStamp();
    printf("(B, N, d): (%d, %d, %d)\n", B, N, d);
    printf("Time: %.3f seconds\n", end - start);

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
    memset(O, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);

    fclose(file);
}

void self_attention(float *q, float *k, float *v, float *o) {
    float *Attn = (float *)malloc(N * N * sizeof(float));
    memset(Attn, 0x00, N * N * sizeof(float));

    QKDotAndScalar(Attn, q, k, 1.0 / sqrt(d));
    SoftMax(Attn, Attn);
    MulAttV(o, Attn, v);

    free(Attn);
}

void QKDotAndScalar(float *out, float *q, float *k, float scalar) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            out[i * N + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * N + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * N + j] *= scalar;
        }
    }
}

void SoftMax(float *out, float *in) {
    for (int i = 0; i < N; i++) {
        float max_value = in[i * N];
        for (int j = 0; j < N; j++) {
            max_value = _max(max_value, in[i * N + j]);
        }
        for (int j = 0; j < N; j++) {
            out[i * N + j] = exp(in[i * N + j] - max_value);
        }

        float sum_value = 0.0F;
        for (int j = 0; j < N; j++) {
            sum_value += out[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            out[i * N + j] = out[i * N + j] / sum_value;
        }
    }
}

void MulAttV(float *out, float *att, float *v) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            out[i * d + j] = 0.0F;
            for (int t = 0; t < N; t++) {
                out[i * d + j] += att[i * N + t] * v[t * d + j];
            }
        }
    }
}

