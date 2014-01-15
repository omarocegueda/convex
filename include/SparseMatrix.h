#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
struct Edge{
    int destination;
    double w;
    Edge();
    Edge(int dest, double _w);
};


class SparseMatrix{
    public:
        int n;
        int k;
        int mObserved;
        /*int *indexToPosition;
        int *positionToIndex;*/
        int *degree;
        Edge **edges;
        double *diagonal;
        SparseMatrix();
        SparseMatrix(int _n, int _k);
        ~SparseMatrix();
        void create(int _n, int _k/*, int totalPositions*/);
        void dellocate(void);
        void draw(unsigned char *img_data, int rows, int cols);
        int multVecRight(double *in, double *out);
        int multVecLeft(double *in, double *out, int m);
        void multDiagLeftRight(double *diagLeft, double *diagRight);
        int sumRowAbsValues(double *sums);
        int sumRowValues(double *sums);
        int sumColumnAbsValues(double *sums, int m);
        int sumColumnValues(double *sums, int m);
        int addEdge(int from, int to, double weight);
        int sumToDiagonal(double *d);
        int testEigenPair(double *evec, double eval);
        double computeAsymetry(void);
        double retrieve(int r, int c);
        int copyFrom(SparseMatrix &S);
        int copySubmatrix(SparseMatrix &S, int k, int *labels, int *newToOldMapping, int *oldToNewMapping);
        int loadFromFile(const char *fname);
};

#endif
