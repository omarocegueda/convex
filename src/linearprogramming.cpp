#include <stdio.h>
#include "linearprogramming.h"
#include "macros.h"
#include <stdlib.h>
#include <time.h>
#include "SparseMatrix.h"
#include <vector>
using namespace std;
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
int linprogPDSmall(double *c, double *A, int n, int m, double *b, double *x, int maxIter, double tolerance){
    double *y=new double[n];
    double *tmpPrimal=new double[m];
    double *tmpDual=new double[n];
    double error=1+tolerance;
    memset(y, 0, sizeof(double)*n);
    double theta=1;
    //double L=estimateMatrixNorm(A, n, m, 1e-9);//this estimation is not necessary for the diagonally preconditioned algorithm. Kept it here just as reference
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
    /*FILE *F=fopen("preconditionerSmall.txt","w");
    fprintf(F, "Sigma (preconditioner for each row):\n");
    for(int i=0;i<n;++i){
        fprintf(F,"%lf\n", sigma[i]);
    }
    fprintf(F, "Sigma (preconditioner for each column):\n");
    for(int j=0;j<m;++j){
        fprintf(F,"%lf\n", tau[j]);
    }
    fclose(F);*/
    for(int j=0;j<m;++j){
        x[j]=-c[j];
    }
    //F=fopen("iterations.txt","w");
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
        //fprintf(F,"%lf\n", error);
    }
    //fclose(F);
    delete[] y;
    delete[] tmpPrimal;
    delete[] tmpDual;
    delete[] sigma;
    delete[] tau;
    return iter;
}

int readVector(const char *fname, double *x, int n){
    FILE *F=fopen(fname,"r");
    double val;
    vector<double> v;
    int len=0;
    while(fscanf(F,"%lf",&val)!=EOF){
        v.push_back(val);
        ++len;
    }
    fclose(F);
    if(len!=n){
        return -1;
    }
    for(int i=0;i<n;++i){
        x[i]=v[i];
    }
    return 0;
}

int linprogPDLarge(char *cname, char* Aname, int n, int m, char *bname, double *x, int maxIter, double tolerance){
    double *c=new double[m];
    int retVal=readVector(cname, c, m);
    if(retVal<0){
        delete[] c;
        return -1;
    }
    double *b=new double[n];
    retVal=readVector(bname, b, n);
    if(retVal<0){
        delete[] c;
        delete[] b;
        return -2;
    }
    SparseMatrix A;
    retVal=A.loadFromFile(Aname);
    if((retVal<0)||(A.n!=n)){
        delete[] c;
        delete[] b;
        return -3;
    }
    double *y=new double[n];
    double *tmpPrimal=new double[m];
    double *tmpDual=new double[n];
    double error=1+tolerance;
    memset(y, 0, sizeof(double)*n);
    double theta=1;
    double *sigma=new double[n];
    double *tau=new double[m];
    retVal=A.sumRowAbsValues(sigma);
    if(retVal<0){
        delete[] c;
        delete[] b;
        return -4;
    }
    retVal=A.sumColumnAbsValues(tau, m);
    if(retVal<0){
        delete[] c;
        delete[] b;
        return -5;
    }
    for(int i=0;i<n;++i)if(sigma[i]>1e-9){
        sigma[i]=1.0/sigma[i];
    }
    for(int j=0;j<m;++j)if(tau[j]>1e-9){
        tau[j]=1.0/tau[j];
    }
    /*FILE *F=fopen("preconditionerLarge.txt","w");
    fprintf(F, "Sigma (preconditioner for each of the %d rows):\n", A.n);
    for(int i=0;i<n;++i){
        fprintf(F,"%lf\n", sigma[i]);
    }
    fprintf(F, "Sigma (preconditioner for each of the %d columns):\n", A.mObserved);
    for(int j=0;j<m;++j){
        fprintf(F,"%lf\n", tau[j]);
    }
    fclose(F);*/
    for(int j=0;j<m;++j){
        x[j]=-c[j];
    }
    //F=fopen("iterations.txt","w");
    int iter;
    for(iter=0;(iter<maxIter)&&(tolerance<error);++iter){
        //---iterate primal variable---
        retVal=A.multVecLeft(y, tmpPrimal, m);
        if(retVal<0){
            delete[] c;
            delete[] b;
            return -6;
        }
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
        retVal=A.multVecRight(tmpPrimal, tmpDual);
        if(retVal<0){
            delete[] c;
            delete[] b;
            return -7;
        }
        for(int i=0;i<n;++i){
            y[i]=y[i]+sigma[i]*(tmpDual[i]-b[i]);
        }
        //fprintf(F,"%lf\n", error);
    }
    //fclose(F);
    delete[] y;
    delete[] tmpPrimal;
    delete[] tmpDual;
    delete[] sigma;
    delete[] tau;
    delete[] c;
    delete[] b;
    return iter;
}

