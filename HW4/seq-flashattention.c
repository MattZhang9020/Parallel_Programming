#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);

void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar);
void RowMax(float *out, float *in, int br, int bc);
void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
void RowSum(float *out, float *in, int br, int bc);
void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
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
        flash_attention(
            Q + (i * N * d),
            K + (i * N * d),
            V + (i * N * d),
            O + (i * N * d));
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

void flash_attention(float *q, float *k, float *v, float *o) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        m[i] = FLT_MIN;
    }

    int br = 32, bc = 32;
    int tr = N / br, tc = N / bc;
    float *kj = (float *)malloc(bc * d * sizeof(float));
    float *vj = (float *)malloc(bc * d * sizeof(float));
    float *qi = (float *)malloc(br * d * sizeof(float));
    float *oi = (float *)malloc(br * d * sizeof(float));
    float *li = (float *)malloc(br * sizeof(float));
    float *mi = (float *)malloc(br * sizeof(float));

    float *sij = (float *)malloc(br * bc * sizeof(float));
    float *pij = (float *)malloc(br * bc * sizeof(float));
    float *mij = (float *)malloc(br * sizeof(float));
    float *lij = (float *)malloc(br * sizeof(float));

    for (int j = 0; j < tc; j++) {
        memcpy(kj, k + j * bc * d, bc * d * sizeof(float));
        memcpy(vj, v + j * bc * d, bc * d * sizeof(float));
        for (int i = 0; i < tr; i++) {
            memcpy(qi, q + i * br * d, br * d * sizeof(float));
            memcpy(oi, o + i * br * d, br * d * sizeof(float));
            memcpy(li, l + i * br, br * sizeof(float));
            memcpy(mi, m + i * br, br * sizeof(float));

            QKDotAndScalar(sij, qi, kj, br, bc, 1.0 / sqrt(d));
            RowMax(mij, sij, br, bc);
            MinusMaxAndExp(pij, sij, mij, br, bc);
            RowSum(lij, pij, br, bc);

            UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, br, bc);

            memcpy(o + i * br * d, oi, br * d * sizeof(float));
            memcpy(l + i * br, li, br * sizeof(float));
            memcpy(m + i * br, mi, br * sizeof(float));
        }
    }

    free(sij);
    free(pij);
    free(mij);
    free(lij);

    free(kj);
    free(vj);
    free(qi);
    free(oi);
    free(li);
    free(mi);

    free(l);
    free(m);
}

void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * bc + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * bc + j] *= scalar;
        }
    }
}

void RowMax(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = in[i * bc];
        for (int j = 0; j < bc; j++) {
            out[i] = _max(out[i], in[i * bc + j]);
        }
    }
}

void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = exp(in[i * bc + j] - mx[i]);
        }
    }
}

void RowSum(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = 0.0F;
        for (int j = 0; j < bc; j++) {
            out[i] += in[i * bc + j];
        }
    }
}

void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc) {
    float *mi_new = (float *)malloc(br * sizeof(float));
    float *li_new = (float *)malloc(br * sizeof(float));

    for (int i = 0; i < br; i++) {
        mi_new[i] = _max(mi[i], mij[i]);
        li_new[i] = exp(mi[i] - mi_new[i]) * li[i] + exp(mij[i] - mi_new[i]) * lij[i];
    }

    for (int i = 0; i < br; i++) {
        for (int j = 0; j < d; j++) {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++) {
                pv += pij[i * bc + t] * vj[t * d + j];
            }
            oi[i * d + j] = (li[i] * exp(mi[i] - mi_new[i]) * oi[i * d + j] + exp(mij[i] - mi_new[i]) * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, br * sizeof(float));
    memcpy(li, li_new, br * sizeof(float));

    free(mi_new);
    free(li_new);
}
