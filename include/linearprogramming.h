#ifndef LINEARPROGRAMMING_H
#define LINEARPROGRAMMING_H
int linprogPDSmall(double *c, double *A, int n, int m, double *b, double *x, int maxIter, double tolerance);
int linprogPDLarge(char *cname, char* Aname, int n, int m, char *bname, double *x, int maxIter, double tolerance);
#endif
