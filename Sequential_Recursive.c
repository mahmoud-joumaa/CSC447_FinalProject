#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

int n;

typedef struct complex {
	double real;
	double img;
} complex;

complex addComplex(complex a, complex b) {
	complex c;
	c.real = a.real + b.real;
	c.img = a.img + b.img;
	return c;
}
complex mulComplex(complex a, complex b) {
	complex c;
	c.real = a.real*b.real - a.img*b.img;
	c.img = a.real*b.img + a.img*b.real;
	return c;
}
complex powComplex(complex c, int k) {
	complex product;
	product.real = 1;
	product.img = 1;
	for (int i = 0; i < k; i++) product = mulComplex(product, c);
	return product;
}
complex oppComplex(complex c) {
	c.real = -c.real;
	c.img = -c.img;
	return c;
}

void fft(complex* c, int start, int end, complex* y) {
	int p = end-start;
	if (p == 1)
		y = c;
	complex w; // = exp((double)2*PI/p);
	w.real = cos(2*PI/p);
	w.img = sin(2*PI/p);
	complex ceven[p/2];
	complex codd[p/2];
	for (int i = 0; i < p/2; i++) ceven[i] = c[i*2];
	for (int i = 0; i < p/2; i++) codd[n/2] = ceven[i] = c[i*2+1];
	complex* yeven; complex y1[p/2];
	fft(ceven, 0, p/2, y1);
	complex* yodd; complex y2[p/2];
	fft(codd, 0, p/2, y2);
	for (int j = 0; j < n/2; j++) {
		y[j] = addComplex(yeven[j], mulComplex(powComplex(w, j), yodd[j]));
		y[j+n/2] = addComplex(yeven[j], oppComplex(mulComplex(powComplex(w, j), yodd[j])));
	}
}

int main() {
	n = 4; // cin >> n;
	complex input[n];
	for (int i = 1; i <= n; i++) {
		input[i-1].real = i;
		input[i-1].img = 0;
	}
	for (int i = 0; i < n; i++) printf("%f %f\n", input[i].real, input[i].img);
	complex output[n];
	fft(input, 0, n, output);
	printf("\n");
	for (int i = 0; i < n; i++) printf("%f %f\n", output[i].real, output[i].img);
	return 0;
}
