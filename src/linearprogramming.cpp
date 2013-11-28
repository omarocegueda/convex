#include "linearprogramming.h"
#include "macros.h"
#include <stdlib.h>
#include <time.h>
void multMatrixVector(double *A, int n, int m, double *x, double *y){
    double *a=A;
    for(int i=0;i<n;++i, a+=m){
        double &ss=y[i];
        ss=0;
        for(int j=0;j<m;++j){
            ss+=a[j]*x[j];
        }
    }
}

void multVectorMatrix(double *x, double *A, int n, int m, double *y){
    for(int j=0;j<m;++j){
        double *a=&A[j];
        double &ss=y[j];
        ss=0;
        for(int i=0;i<n;++i, a+=m){
            ss+=x[i]*(*a);
        }
    }
}

void fillRandom(double *x, int n){
    srand(time(NULL));
    for(int i=0;i<n;++i){
        x[i]=(rand()%1001)/1000.0;
    }
}

double getVectorNorm(double *x, int n){
    double ss=0;
    for(int i=0;i<n;++i,x++){
        ss+=(*x)*(*x);
    }
    return sqrt(ss);
}

double estimateMatrixNorm(double *A, int n, int m, double tol){
    double *x=new double[m];
    double *y=new double[n];
    fillRandom(x, m);
    double e=getVectorNorm(x,m);
    if(e>1e-9){
        for(int i=0;i<m;++i){
            x[i]/=e;
        }
    }
    int cnt=0;
    double e0=0;
    while(fabs(e-e0)>tol*e){
        e0=e;
        multMatrixVector(A, n, m, x, y);
        e=getVectorNorm(y,n);
        multVectorMatrix(y, A, n, m, x);
        double nn=getVectorNorm(x,m);
        if(nn>1e-9){
            for(int i=0;i<m;++i){
                x[i]/=nn;
            }
        }
        ++cnt;
    }
    delete[] x;
    delete[] y;
    return e;
}

/*
    Solves 
        min <c,x> s.t. Ax=b, x>=0
    using the Chambolle's primal-dual algorithm
    Returns the solution in x0
    A is given explicitly. It is intended for small-scale problems
*/
double linprog_pd(double *c, double *A, int n, int m, double *b, double *x, int maxIter, double tolerance){
    double *y=new double[n];
    double *tmpPrimal=new double[m];
    double *tmpDual=new double[n];
    double error=1+tolerance;
    memset(y, 0, sizeof(double)*n);
    double theta=1;
    double L=estimateMatrixNorm(A, n, m, 1e-9);//this estimation is not necessary for the diagonally preconditioned algorithm. Kept it here just as reference
    double *sigma=new double[n];
    double *tau=new double[m];
    for(int i=0;i<n;++i){
        sigma[i]=0;
        for(int j=0;j<m;++j){
            sigma[i]+=fabs(A[i*m+j]);
        }
        if(sigma[i]>1e-9){
            sigma[i]=1.0/sigma[i];
        }
    }
    for(int j=0;j<m;++j){
        tau[j]=0;
        for(int i=0;i<n;++i){
            tau[j]+=fabs(A[i*m+j]);
        }
        if(tau[j]>1e-9){
            tau[j]=1.0/tau[j];
        }
    }
    /*for(int j=0;j<m;++j){
        x[j]=-c[j];
    }*/
    int iter;
    for(iter=0;(iter<maxIter)&&(tolerance<error);++iter){
        //---iterate primal variable---
        multVectorMatrix(y,A,n,m,tmpPrimal);
        error=0;
        for(int j=0;j<m;++j){
            double newX=x[j]-tau[j]*(tmpPrimal[j]+c[j]);
            if(newX<0){
                newX=0;
            }
            tmpPrimal[j]=newX+theta*(newX-x[j]);
            double diff=fabs(x[j]-newX);
            if(error<diff){
                error=diff;
            }
            x[j]=newX;
        }
        //---iterate dual variable---
        multMatrixVector(A, n, m, tmpPrimal, tmpDual);
        for(int i=0;i<n;++i){
            y[i]=y[i]+sigma[i]*(tmpDual[i]-b[i]);
        }
    }
    delete[] y;
    delete[] tmpPrimal;
    delete[] tmpDual;
    delete[] sigma;
    delete[] tau;
    return iter;
}
