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
#include <pthread.h>
#include <immintrin.h>

typedef struct {
    int width;
    int height;
    int iters;

    int* image;

    double left;
    double right;
    double lower;
    double upper;
    double x_scale;
    double y_scale;
} mandelbrot_params_t;

typedef struct {
    int start_row;
    int end_row;
} mandelbrot_task_t;

typedef struct {
    int num_threads;
    int task_capacity;
    int task_count;
    int head;
    int tail;
    int shutdown;

    mandelbrot_task_t* tasks;
    mandelbrot_params_t params;

    pthread_t* threads;

    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_not_empty;
    pthread_cond_t queue_not_full;
} thread_pool_t;

void mandelbrot_worker(void* arg1, void* arg2) {
    mandelbrot_task_t* task = (mandelbrot_task_t*) arg1;
    mandelbrot_params_t* params = (mandelbrot_params_t*) arg2;
    
    __m512d v_four = _mm512_set1_pd(4.0);
    __m512d v_two = _mm512_set1_pd(2.0);
    __m512i v_one = _mm512_set1_epi32(1);
    
    alignas(64) double x_coords[32];
    alignas(64) int repeats[32];
    
    __m512d v_x[4];
    __m512d v_y[4];
    __m512d v_x0[4];
    __m512i v_repeats[4];
    __mmask8 masks[4];
    __mmask8 base_masks[4];
    
    for (int j = task->start_row; j < task->end_row; j++) {
        double y0 = j * params->y_scale + params->lower;
        __m512d v_y0 = _mm512_set1_pd(y0);
        
        for (int i = 0; i < params->width; i += 32) {
            int remaining = params->width - i;
            if (remaining <= 0) break;
            
            int pixels_to_process = (remaining < 32) ? remaining : 32;
            
            for (int k = 0; k < pixels_to_process; k++) {
                x_coords[k] = (i + k) * params->x_scale + params->left;
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
            
            for (int iter = 0; iter < params->iters && combined_mask; iter++) {
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
                params->image[j * params->width + i + k] = repeats[k];
            }
        }
    }
}

void* thread_pool_worker(void* arg) {
    thread_pool_t* pool = (thread_pool_t*) arg;
    
    while (1) {
        mandelbrot_task_t task;
        mandelbrot_params_t params;

        pthread_mutex_lock(&pool->queue_mutex);
        
        while (pool->task_count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->queue_not_empty, &pool->queue_mutex);
        }
        
        if (pool->shutdown && pool->task_count == 0) {
            pthread_mutex_unlock(&pool->queue_mutex);
            pthread_exit(NULL);
        }
        
        task = pool->tasks[pool->head];
        pool->head = (pool->head + 1) % pool->task_capacity;
        pool->task_count--;

        params = pool->params;
        
        pthread_cond_signal(&pool->queue_not_full);
        pthread_mutex_unlock(&pool->queue_mutex);
        
        mandelbrot_worker(&task, &params);
    }
    return NULL;
}

void thread_pool_destroy(thread_pool_t* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->queue_mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->queue_not_empty);
    pthread_cond_broadcast(&pool->queue_not_full);
    pthread_mutex_unlock(&pool->queue_mutex);
    
    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->queue_mutex);
    pthread_cond_destroy(&pool->queue_not_empty);
    pthread_cond_destroy(&pool->queue_not_full);
    
    free(pool->threads);
    free(pool->tasks);
    free(pool);
}

thread_pool_t* thread_pool_init(int num_threads, int task_capacity, mandelbrot_params_t* params) {
    thread_pool_t* pool = (thread_pool_t*) malloc(sizeof(thread_pool_t));
    assert(pool);
    
    pool->num_threads = num_threads;
    pool->task_capacity = task_capacity;
    pool->task_count = 0;
    pool->head = 0;
    pool->tail = 0;
    pool->shutdown = 0;
    pool->threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    pool->tasks = (mandelbrot_task_t*) malloc(task_capacity * sizeof(mandelbrot_task_t));
    pool->params = *params;

    assert(pool->threads && pool->tasks);
    
    pthread_mutex_init(&pool->queue_mutex, NULL);
    pthread_cond_init(&pool->queue_not_empty, NULL);
    pthread_cond_init(&pool->queue_not_full, NULL);
    
    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&pool->threads[i], NULL, thread_pool_worker, (void*)pool);
        if (rc) {
            thread_pool_destroy(pool);
            exit(-1);
        }
    }
    
    return pool;
}

void thread_pool_add_task(thread_pool_t* pool, int start_row, int end_row) {
    mandelbrot_task_t task;

    task.start_row = start_row;
    task.end_row = end_row;
    
    pthread_mutex_lock(&pool->queue_mutex);
    
    while (pool->task_count == pool->task_capacity && !pool->shutdown) {
        pthread_cond_wait(&pool->queue_not_full, &pool->queue_mutex);
    }
    
    if (pool->shutdown) {
        pthread_mutex_unlock(&pool->queue_mutex);
        return;
    }
    
    pool->tasks[pool->tail] = task;
    pool->tail = (pool->tail + 1) % pool->task_capacity;
    pool->task_count++;
    
    pthread_cond_signal(&pool->queue_not_empty);
    pthread_mutex_unlock(&pool->queue_mutex);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);

    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }

    free(row);

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);

    fclose(fp);
}

int main(int argc, char** argv) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    assert(argc == 9);

    const char* filename = argv[1];

    int num_threads = CPU_COUNT(&cpu_set);

    mandelbrot_params_t shared_params;

    shared_params.iters = strtol(argv[2], 0, 10);
    shared_params.left = strtod(argv[3], 0);
    shared_params.right = strtod(argv[4], 0);
    shared_params.lower = strtod(argv[5], 0);
    shared_params.upper = strtod(argv[6], 0);
    shared_params.width = strtol(argv[7], 0, 10);
    shared_params.height = strtol(argv[8], 0, 10);
    shared_params.x_scale = (shared_params.right - shared_params.left) / shared_params.width;
    shared_params.y_scale = (shared_params.upper - shared_params.lower) / shared_params.height;

    shared_params.image = (int*) malloc(shared_params.width * shared_params.height * sizeof(int));
    assert(shared_params.image);

    thread_pool_t* pool = thread_pool_init(num_threads, num_threads * 16, &shared_params);

    for (int row = 0; row < shared_params.height; row++) {
        thread_pool_add_task(pool, row, row + 1);
    }

    thread_pool_destroy(pool);

    write_png(filename, shared_params.iters, shared_params.width, shared_params.height, shared_params.image);
    free(shared_params.image);

    exit(0);
}