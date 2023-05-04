// %%cuda --name CUDATEST.cu

#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADNUM 1024


#define M_PI 3.14159265358979323846

_device_ int bitReverse(unsigned int x, int log2n)
{
    int n = 0;
    for (int i = 0; i < log2n; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

_device_ void fft(double* real, double* imag, int n) {
    
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




_device_ void ifft(double* real, double* imag, int n) {

    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
    }
    
    //const int shared_mem_size = 128*64*sizeof(double);
    fft(real, imag, n);

    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i] / n;
        real[i] = real[i] / n;
    }

    
}



_global_ void fft_parallel(double* buffer_real_d, double* buffer_imag_d, int num_samples, int n){
    
    int i = blockIdx.x;
    int chunks_per_thread = (num_samples/n)/THREADNUM;

    for(int k=0; k<chunks_per_thread; k++){

      int index = i*n*chunks_per_thread+k*n;
      //const int shared_mem_size = 128*64*sizeof(double);
      fft((buffer_real_d+index), (buffer_imag_d+index), n);
      
      

      buffer_real_d[i*n*chunks_per_thread+k*n + 1] *= 1.03;

    }
    

    
}

_global_ void ifft_parallel(double* buffer_real_d, double* buffer_imag_d, int num_samples, int n){
    
    int i = blockIdx.x;
    int chunks_per_thread = (num_samples/n)/THREADNUM;


    for(int k=0; k<chunks_per_thread; k++){



        ifft((buffer_real_d+(i*n*chunks_per_thread)+k*n), (buffer_imag_d+(i*n*chunks_per_thread)+k*n), n);

        
    }

    
}



int main()
{
    clock_t t = clock();

    SNDFILE *audio_file;
    SF_INFO audio_info;

    char filename[] = "audio1.wav";
    char output[] = "file_altered.wav";
    
    audio_file = sf_open(filename, SFM_READ, &audio_info);
    if (audio_file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }
    
    printf("Sample rate: %d\n", audio_info.samplerate);
    printf("Number of channels: %d\n", audio_info.channels);
    printf("Number of frames: %ld\n", audio_info.frames);

    int num_channels = audio_info.channels;
    int num_samples = audio_info.frames * num_channels;

    
    double* buffer_real_h = (double*) malloc(num_samples * sizeof(double));
    double* buffer_imag_h = (double*) malloc(num_samples * sizeof(double));
    double* buffer_real_d; double* buffer_imag_d;

    cudaMalloc((double**) &buffer_real_d, num_samples*sizeof(double));
    cudaMalloc((double**) &buffer_imag_d, num_samples*sizeof(double));


    sf_count_t read;
    if(read = sf_readf_double(audio_file, buffer_real_h, num_samples/num_channels) != num_samples/num_channels){
        printf("%ld\n", read);
        printf("ERROR: %s\n", sf_strerror(audio_file));
        return 1;
    }
    
    sf_close(audio_file);

    
    for(int i=0; i<num_samples; i++){
        buffer_imag_h[i] = i;
    }



    cudaMemcpy(buffer_real_d, buffer_real_h, num_samples*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer_imag_d, buffer_imag_h, num_samples*sizeof(double), cudaMemcpyHostToDevice);

    int chunk_size = 128;
 

    
    
    
 
    fft_parallel<<<THREADNUM, 1>>>(buffer_real_d, buffer_imag_d, num_samples, chunk_size);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    

    ifft_parallel<<<THREADNUM, 1>>>(buffer_real_d, buffer_imag_d, num_samples, chunk_size);
    cudaDeviceSynchronize();
 
    
    
    
    cudaMemcpy(buffer_real_h, buffer_real_d, num_samples*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(buffer_imag_h, buffer_imag_d, num_samples*sizeof(double), cudaMemcpyDeviceToHost);
 

    SNDFILE* SNoutput = sf_open(output, SFM_WRITE, &audio_info);
    sf_count_t written;
    if(written = sf_write_double(SNoutput, buffer_real_h, num_samples) != num_samples){
        printf("%ld\n", written);
        printf("ERROR1: %s\n", sf_strerror(SNoutput));
        return 1;
    }

    t = clock()-t;

    double time_taken = ((double)t)/CLOCKS_PER_SEC;
 
    cudaDeviceSynchronize();

    printf("Time taken: %f\n", time_taken);
 
    
    
    sf_close(SNoutput);

    free(buffer_real_h);
    free(buffer_imag_h);
    cudaFree(buffer_real_d); 
    cudaFree(buffer_imag_d);

    
    return 0;
}
