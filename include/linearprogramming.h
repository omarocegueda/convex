#ifndef LINEARPROGRAMMING_H
#define LINEARPROGRAMMING_H
double linprog_pd(double *c, double *A, int n, int m, double *b, double *x, int maxIter, double tolerance);
#endif
