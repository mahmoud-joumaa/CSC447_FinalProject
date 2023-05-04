#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


#define M_PI 3.14159265358979323846
#define chunk_size 128

int bitReverse(unsigned int x, int log2n)
{
    int n = 0;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

void fft(double* real, double* imag, int n) {
    int i, j, k, m;
    double tempreal, tempimag, theta, wreal, wimag, wtempreal, wtempimag;
 
    int s = log2(n); // Compute the number of stages
 
    for (i = 0; i < n; i++) {
        j = bitReverse(i, s); // Compute the bit-reversed index
        if (j > i) {
            // Swap the real and imaginary parts
            tempreal = real[i];
            real[i] = real[j];
            real[j] = tempreal;
            tempimag = imag[i];
            imag[i] = imag[j];
            imag[j] = tempimag;
        }
    }
 
    for (i = 2; i <= n; i *= 2) {
        // Compute the twiddle factors
        theta = 2 * M_PI / i;
        wtempreal = cos(theta);
        wtempimag = sin(theta);
 
        for (j = 0; j < n; j += i) {
            wreal = 1.0;
            wimag = 0.0;
            for (k = 0; k < i / 2; k++) {
                // Compute the butterfly
                m = j + k;
                tempreal = wreal * real[m + i / 2] - wimag * imag[m + i / 2];
                tempimag = wreal * imag[m + i / 2] + wimag * real[m + i / 2];
                real[m + i / 2] = real[m] - tempreal;
                imag[m + i / 2] = imag[m] - tempimag;
                real[m] += tempreal;
                imag[m] += tempimag;
                // Update the twiddle factor
                tempreal = wreal;
                wreal = wreal * wtempreal - wimag * wtempimag;
                wimag = tempreal * wtempimag + wimag * wtempreal;
            }
        }
    }
}



void ifft(double* real, double* imag, int n) {

    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
    }
    

    fft(real, imag, n);
    
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i] / n;
        real[i] = real[i] / n;
    }
}


int main(int argc, char** argv)
{
    int rank, size;
    int num_samples;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    clock_t t = clock();
    char filename[] = "audio1.wav";
    char output[] = "file_altered.wav";
    SNDFILE *audio_file;
    SF_INFO audio_info;
    double* buffer_real;
    double* buffer_imag;
    int chunk_per;
    
    

    if(rank == size-1){
        

        
        audio_file = sf_open(filename, SFM_READ, &audio_info);
        if (audio_file == NULL) {
            printf("Error opening file.\n");
            return 1;
        }

        int num_channels = audio_info.channels;
        num_samples = audio_info.frames * num_channels;
        
        printf("Sample rate: %d\n", audio_info.samplerate);
        printf("Number of channels: %d\n", audio_info.channels);
        printf("Number of frames: %ld\n", audio_info.frames);

        buffer_real = (double*) malloc(num_samples * sizeof(double));
        buffer_imag = (double*) malloc(num_samples * sizeof(double));

        
        
        
        sf_count_t read;
        if(read = sf_readf_double(audio_file, buffer_real, num_samples/num_channels) != num_samples/num_channels){
            printf("%ld\n", read);
            printf("ERROR: %s\n", sf_strerror(audio_file));
            return 1;
            
        }
        for(int i=0; i<num_samples; i++){
            buffer_imag[i] = i;
        }
        
        int num_chunk = num_samples/chunk_size;
        chunk_per = num_chunk/(size-1);


        for(int i=0; i<size-1; i++){
            
            int send_status = MPI_Send(&chunk_per, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            if (send_status != MPI_SUCCESS) {
                printf("Error sending data.\n");
                return 1;
            }

            // double* ptr
            
            send_status = MPI_Send((buffer_real+i*chunk_per*chunk_size), chunk_size*chunk_per, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            if (send_status != MPI_SUCCESS) {
                printf("Error sending data.\n");
                return 1;
            }

            send_status = MPI_Send((buffer_imag+i*chunk_per*chunk_size), chunk_size*chunk_per, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            
            
            if (send_status != MPI_SUCCESS) {
                printf("Error sending data.\n");
                return 1;
            }
            
  
        }

    }
    
    if(rank != size-1){
        
        int num_chunks;
        
        MPI_Recv(&num_chunks, 1, MPI_INT, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        

        double* chunk_real = (double*) malloc(chunk_size*num_chunks*sizeof(double));
        double* chunk_imag = (double*) malloc(chunk_size*num_chunks*sizeof(double));
        
        MPI_Recv(chunk_real, chunk_size*num_chunks, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(chunk_imag, chunk_size*num_chunks, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        

        for(int i=0; i<num_chunks; i++){
            fft((chunk_real+i*chunk_size), (chunk_imag+i*chunk_size), chunk_size);
            
            chunk_real[i*chunk_size + 1] *= 1.02;
            ifft((chunk_real+i*chunk_size), (chunk_imag+i*chunk_size), chunk_size);

            if(rank == 0 && i == 0){
                
            }
        }
        printf("rank=%d\n", rank);

        MPI_Send(chunk_real, chunk_size*num_chunks, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD);
        MPI_Send(chunk_imag, chunk_size*num_chunks, MPI_DOUBLE, size-1, 0, MPI_COMM_WORLD);
        
        free(chunk_real);
        free(chunk_imag);
    }
   

    if(rank == size-1){
        for(int i=0; i<size-1; i++){
            
            MPI_Recv((buffer_real+i*chunk_per*chunk_size), chunk_size*chunk_per, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv((buffer_imag+i*chunk_per*chunk_size), chunk_size*chunk_per, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            
        }
        
    }
    

    MPI_Barrier(MPI_COMM_WORLD);
    

    if(rank == size-1){
        for(int j=0; j<1; j++){
            printf("%d\n", num_samples);
        }
        SNDFILE* SNoutput = sf_open(output, SFM_WRITE, &audio_info);
        sf_count_t written;
        if(written = sf_write_double(SNoutput, buffer_real, num_samples) != num_samples){
            printf("%ld\n", written);
            printf("ERROR1: %s\n", sf_strerror(SNoutput));
            return 1;
        }
        
        t = clock()-t;

        double time_taken = ((double)t)/CLOCKS_PER_SEC;

        printf("Time taken: %f\n", time_taken);

        sf_close(SNoutput);
        sf_close(audio_file);
        
        //free(buffer_real);
    }
    

    MPI_Finalize();

    
    return 0;
}
