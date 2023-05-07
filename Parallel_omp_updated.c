// %%writefile test.c

#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <math.h>
#include <time.h>
#include <omp.h>


#define M_PI 3.14159265358979323846

#define OMP_NUM_THREADS 2

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
        
       #pragma omp parallel for
        for (j = 0; j < n; j += i) {
            wreal = 1.0;
            wimag = 0.0;
            #pragma omp for
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

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
}
    

    fft(real, imag, n);
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i] / n;
        real[i] = real[i] / n;
}
}


int main()
{

    clock_t t = clock();

    SNDFILE *audio_file;
    SF_INFO audio_info;

    char filename[] = "audio_glitched.wav";
    char output[] = "audio_fixed.wav";
    
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
    double* buffer_real = (double*) malloc(num_samples * sizeof(double));
    double* buffer_imag = (double*) malloc(num_samples * sizeof(double));


    // printf("%d\n", num_samples);
    

    sf_count_t read;
    if(read = sf_readf_double(audio_file, buffer_real, num_samples/num_channels) != num_samples/num_channels){
        printf("%ld\n", read);
        printf("ERROR: %s\n", sf_strerror(audio_file));
        return 1;
}
    
   

    #pragma omp parallel for
    for(int i=0; i<num_samples; i++){
        buffer_imag[i] = i;
}

    int chunk_size = 128;

    double* chunk_real = (double*) malloc(chunk_size * sizeof(double));
    double* chunk_imag = (double*) malloc(chunk_size * sizeof(double));

    #pragma omp parallel for
    for(int i = 0; i < num_samples/chunk_size; i++){
      // if (i==0)
      //  printf("%d\n", omp_get_num_threads());
      // printf("%d ", omp_get_thread_num());

      int index = i*chunk_size;
      fft((buffer_real+index), (buffer_imag+index), chunk_size);
      buffer_real[index + 1] /= 1.03;
      ifft((buffer_real+index), (buffer_imag+index), chunk_size);

}

  printf("\n");
    

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
    free(buffer_imag);
    free(chunk_imag);
    free(chunk_real);

    
    return 0;
}
