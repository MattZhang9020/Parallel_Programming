#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = atoi(argv[1]);

    char *input_filename = argv[2];
    char *output_filename = argv[3];
    
    int elements_per_process = n / size;
    int remainder = n % size;

    int local_n = (rank < remainder) ? elements_per_process + 1 : elements_per_process;
    int offset = rank * elements_per_process + (rank < remainder ? rank : remainder);

    int left_n = local_n + (rank == remainder);
    int right_n = local_n - (rank + 1 == remainder);

    int max_n = std::max(left_n, right_n);

    float* local_data;
    float* partner_data;
    float* merged;

    MPI_Alloc_mem(sizeof(float) * local_n, MPI_INFO_NULL, &local_data);
    MPI_Alloc_mem(sizeof(float) * max_n, MPI_INFO_NULL, &partner_data);
    MPI_Alloc_mem(sizeof(float) * local_n, MPI_INFO_NULL, &merged);

    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    MPI_File_read_at(input_file, offset * sizeof(float), local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    boost::sort::spreadsort::float_sort(local_data, local_data+local_n);

    int phase = 0;
        
    while (phase < size+1) {
        int partner, partner_n;

        if (phase % 2 == 0) {
            if (rank % 2 == 0) {
                partner = rank + 1;
                partner_n = right_n;
            } else {
                partner = rank - 1;
                partner_n = left_n;
            }
        } else {
            if (rank % 2 == 1) {
                partner = rank + 1;
                partner_n = right_n;
            } else {
                partner = rank - 1;
                partner_n = left_n;
            }
        }
        
        if (partner >= 0 && partner < size) {
            if (local_n <= 0 || partner_n <= 0) {
                phase++;
                continue;
            }

            MPI_Request send_request, recv_request;

            int local_break;

            if (rank < partner) {
                float local_back=(local_data)[local_n-1], partner_front;

                MPI_Send(&local_back, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
                MPI_Recv(&partner_front, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (local_back <= partner_front) {
                    phase++;
                    continue;
                }

                (partner_data)[0] = partner_front;
                
                MPI_Isend((local_data), local_n-1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
                MPI_Irecv((partner_data)+1, partner_n-1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);
                
                int low = 0, high = local_n, bound = local_n/4;
                while (low < high) {
                    int mid = low + (high - low) / 2;
                    if (mid < local_n && (local_data)[mid] < partner_front) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }

                    if (low < bound) break;
                }
                local_break = low >= bound ? low : 0;

                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                if (local_break != 0) memcpy(merged, local_data, local_break * sizeof(float));

                float* i     = local_data + local_break;
                float* i_end = local_data + local_n;

                float* j     = partner_data;
                float* j_end = partner_data + partner_n;
                
                float* k     = merged + local_break;
                float* k_end = merged + local_n;

                while (k < k_end) {
                    if (i < i_end && (j >= j_end || *i <= *j)) {
                        *k++ = *i++;
                    } else if (j < j_end) {
                        *k++ = *j++;
                    }
                }
            } else {
                float local_front=(local_data)[0], partner_back;

                MPI_Recv(&partner_back, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&local_front, 1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
                
                if (local_front >= partner_back) {
                    phase++;
                    continue;
                }

                (partner_data)[partner_n - 1] = partner_back;

                MPI_Isend((local_data)+1, local_n-1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &send_request);
                MPI_Irecv((partner_data), partner_n-1, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &recv_request);

                int low = 0, high = local_n, bound = local_n - local_n/4;
                while (low > high) {
                    int mid = low + (high - low) / 2;
                    if (mid < local_n && (local_data)[mid] > partner_back) {
                        high = mid - 1;
                    } else {
                        low = mid;
                    }

                    if (high > bound) break;
                }
                local_break = high <= bound ? high : local_n;

                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                if (local_break != local_n) memcpy(merged + local_break, local_data + local_break, (local_n - local_break) * sizeof(float));

                float* i     = local_data + local_break-1;
                float* i_end = local_data;

                float* j     = partner_data + partner_n-1;
                float* j_end = partner_data;

                float* k     = merged + local_break-1;
                float* k_end = merged;
                
                while (k >= k_end) {
                    if (i >= i_end && (j < j_end || *i > *j)) {
                        *k-- = *i--;
                    } else if (j >= j_end) {
                        *k-- = *j--;
                    }
                }
            }

            float* temp = local_data;
            local_data = merged;
            merged = temp;
        }
        phase++;
    }
    
    MPI_File_write_at(output_file, offset * sizeof(float), local_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    MPI_Free_mem(local_data);
    MPI_Free_mem(partner_data);
    MPI_Free_mem(merged);

    MPI_Finalize();
    return 0;
}