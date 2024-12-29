#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define PNG_NO_SETJMP

#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <immintrin.h>

typedef struct {
    FILE* fp;

    png_structp png_ptr;
    png_infop info_ptr;
    png_bytep* rows;

    int width;
    int height;
    int iters;
} PngWriter;

PngWriter* init_png_writer(const char* filename, int width, int height, int iters) {
    PngWriter* writer = (PngWriter*) malloc(sizeof(PngWriter));

    writer->fp = fopen(filename, "wb");
    assert(writer->fp);

    writer->png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(writer->png_ptr);

    writer->info_ptr = png_create_info_struct(writer->png_ptr);
    assert(writer->info_ptr);

    writer->width = width;
    writer->height = height;
    writer->iters = iters;
    
    png_init_io(writer->png_ptr, writer->fp);
    png_set_IHDR(writer->png_ptr, writer->info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(writer->png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(writer->png_ptr, writer->info_ptr);
    png_set_compression_level(writer->png_ptr, 1);
    
    writer->rows = (png_bytep*) malloc(height * sizeof(png_bytep));

    return writer;
}

void write_png_row(PngWriter* writer, const int* buffer, int local_index, int y) {
    writer->rows[y] = (png_bytep) malloc(3 * writer->width * sizeof(png_byte));
    memset(writer->rows[y], 0, 3 * writer->width * sizeof(png_byte));
    
    for (int x = 0; x < writer->width; x++) {
        int p = buffer[local_index * writer->width + x];

        png_bytep color = writer->rows[y] + x * 3;

        if (p != writer->iters) {
            if (p & 16) {
                color[0] = 240;
                color[1] = color[2] = p % 16 * 16;
            } else {
                color[0] = p % 16 * 16;
            }
        }
    }
}

void finish_png_writer(PngWriter* writer) {
    for (int y = writer->height-1; y >= 0; y--) {
        png_write_row(writer->png_ptr, writer->rows[y]);
    }
    
    png_write_end(writer->png_ptr, NULL);
    png_destroy_write_struct(&writer->png_ptr, &writer->info_ptr);

    for (int i = 0; i < writer->height; i++) {
        free(writer->rows[i]);
    }
    free(writer->rows);

    fclose(writer->fp);
    
    free(writer);
}

int main(int argc, char** argv) {
    assert(argc == 9);

    int rank, size, rows, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    omp_set_num_threads(CPU_COUNT(&cpu_set));

    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    int local_height = (height + size - 1) / size;
    int* local_rows = (int*) malloc(local_height * sizeof(int));

    local_height = 0;
    for (int i = rank; i < height; i += size) {
        local_rows[local_height++] = i;
    }

    int* local_image = (int*) malloc(width * local_height * sizeof(int));
    assert(local_image);

    #pragma omp parallel
    {
        __m512d v_four = _mm512_set1_pd(4.0);
        __m512d v_two = _mm512_set1_pd(2.0);
        __m512i v_one = _mm512_set1_epi32(1);
        
        double x_scale = (right - left) / width;
        double y_scale = (upper - lower) / height;
        
        alignas(64) double x_coords[32];
        alignas(64) int repeats[32];
        
        __m512d v_x[4];
        __m512d v_y[4];
        __m512d v_x0[4];
        __m512i v_repeats[4];
        __mmask8 masks[4];
        __mmask8 base_masks[4];

        #pragma omp for schedule(dynamic) nowait
        for (int j = 0; j < local_height; j++) {
            double y0 = local_rows[j] * y_scale + lower;
            __m512d v_y0 = _mm512_set1_pd(y0);
            
            for (int i = 0; i < width; i += 32) {
                int remaining = width - i;
                if (remaining <= 0) break;
                
                int pixels_to_process = (remaining < 32) ? remaining : 32;
                
                for (int k = 0; k < pixels_to_process; k++) {
                    x_coords[k] = (i + k) * x_scale + left;
                }

                for (int k = pixels_to_process; k < 32; k++) {
                    x_coords[k] = x_coords[pixels_to_process - 1];
                }
                
                for (int v = 0; v < 4; v++) {
                    v_x0[v] = _mm512_load_pd(&x_coords[v * 8]);
                    
                    v_x[v] = _mm512_setzero_pd();
                    v_y[v] = _mm512_setzero_pd();
                    v_repeats[v] = _mm512_setzero_si512();
                    
                    int vector_pixels = pixels_to_process - (v * 8);
                    if (vector_pixels > 8) vector_pixels = 8;
                    if (vector_pixels > 0) {
                        base_masks[v] = (1u << vector_pixels) - 1;
                    } else {
                        base_masks[v] = 0;
                    }
                    masks[v] = base_masks[v];
                }
                
                __mmask8 combined_mask = 0;
                for (int v = 0; v < 4; v++) {
                    combined_mask |= masks[v];
                }
                
                for (int iter = 0; iter < iters && combined_mask; iter++) {
                    combined_mask = 0;
                    
                    for (int v = 0; v < 4; v++) {
                        if (masks[v]) {
                            __m512d v_x2 = _mm512_mul_pd(v_x[v], v_x[v]);
                            __m512d v_y2 = _mm512_mul_pd(v_y[v], v_y[v]);
                            __m512d v_length_squared = _mm512_add_pd(v_x2, v_y2);
                            
                            masks[v] &= _mm512_cmp_pd_mask(v_length_squared, v_four, _CMP_LT_OS) & base_masks[v];
                            
                            if (masks[v]) {
                                __m512d v_xy2 = _mm512_mul_pd(_mm512_mul_pd(v_x[v], v_y[v]), v_two);
                                v_x[v] = _mm512_add_pd(_mm512_sub_pd(v_x2, v_y2), v_x0[v]);
                                v_y[v] = _mm512_add_pd(v_xy2, v_y0);
                                v_repeats[v] = _mm512_mask_add_epi32(v_repeats[v], masks[v], v_repeats[v], v_one);
                            }
                            
                            combined_mask |= masks[v];
                        }
                    }
                }
                
                for (int v = 0; v < 4; v++) {
                    _mm512_store_si512(&repeats[v * 8], v_repeats[v]);
                }
                
                for (int k = 0; k < pixels_to_process; k++) {
                    local_image[j * width + i + k] = repeats[k];
                }
            }
        }
    }

    if (rank == 0) {
        PngWriter* writer = init_png_writer(filename, width, height, iters);
        
        #pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < local_height; j++) {
            write_png_row(writer, local_image, j, local_rows[j]);
        }
        
        for (int src = 1; src < size; src++) {
            int src_height;
            MPI_Recv(&src_height, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int* src_rows = (int*) malloc(src_height * sizeof(int));
            MPI_Recv(src_rows, src_height, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int* src_buffer = (int*) malloc(width * src_height * sizeof(int));
            MPI_Recv(src_buffer, width * src_height, MPI_INT, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < src_height; j++) {
                write_png_row(writer, src_buffer, j, src_rows[j]);
            }
            
            free(src_rows);
            free(src_buffer);
        }
        
        finish_png_writer(writer);
    } else {
        MPI_Send(&local_height, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(local_rows, local_height, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(local_image, local_height * width, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }

    free(local_rows);
    free(local_image);
    MPI_Finalize();
    return 0;
}
