#include <iostream>
#include <vector>
#include <armadillo>
#include "symm_matrix_mult.h"

using namespace std;
using namespace arma;

#ifdef MKL
#define dspmv_ DSPMV
#define dgemv_ DGEMV
#endif

extern "C" {
    void dspmv_(const char * UPLO, const int * N, const double * ALPHA, double * AP, double * X,
                const int * INCX, const double * BETA, double * Y, const int * INCY);
    void dgemv_(const char * TRANS, const int * M, const int * N, const double * ALPHA, double * A, const int * LDA,
                double * X, const int * INCX, const double * BETA, double * Y, const int * INCY);
}

int main()
{
    /* Compare multiplication using Binary tree-packed storage and single-pass on packed storage algorithms */
    size_t n, m;
    cout << "Enter n: ";    cin >> n;
    cout << "Enter m: ";    cin >> m;

    unsigned depth;
    cout << "Enter tree depth (or 0 to automatically figure depth): ";
    cin >> depth;
    if (depth==0 || pow(2,depth) > n) {
        depth = figureTreePackedDepth(n);
        cout << "Using depth " << depth << "\n";
    }

    vec AP(n*(n+1)/2, fill::randn);
    vector<double> A = conv_to< vector<double> >::from(AP);

    clock_t clck = clock();
    SymmetricMatrix<double> A_BTPS(n, depth);
    A_BTPS.setFromPacked(AP);
    clck = clock() - clck;
    printf("Setting A_BTPS took %f seconds.\n", ((float)clck)/CLOCKS_PER_SEC);

    mat X(n, m, fill::randn);

    clck = clock();
    mat Y = A_BTPS * X;
    clck = clock() - clck;
    printf("BTPS multiplication took %f seconds.\n", ((float)clck)/CLOCKS_PER_SEC);



    mat A_full(n,n,fill::none);
    vector<double>::iterator itr = A.begin();
    for (size_t i = 0; i < n; i++) {
        double a_ij = *itr++;
        A_full(i,i) = a_ij;
        for (size_t j = i+1; j < n; j++) {
            double a_ij = *itr++;
            A_full(i,j) = a_ij;
            A_full(j,i) = a_ij;
        }
    }

    clck = clock();
    mat Y_fullt = A_full.t() * X;
    clck = clock() - clck;
    printf("Transposed full symmetric matrix multiplication took %f seconds.\n", ((float)clck)/CLOCKS_PER_SEC);



    /* BLAS */
    int N = 1000, M = 100, INCX = 1, INCY = 1;
    double ALPHA = 1, BETA = 0;
    char UPLO = 'L';
    double Y_dspmv[N*M];

    cout << "Check -1 ";
    X.reshape(n*m, 1);
    cout << "Check 0 ";
    vector<double> Xvec = conv_to< vector<double> >::from( conv_to<vec>::from(X) );
    cout << "Check 1 ";
    clck = clock();
    cout << "Check 2 ";
    for (int i = 0; i < M; i++)
        dspmv_(&UPLO, &N, &ALPHA, &A[0], &Xvec[N*i], &INCX, &BETA, &Y_dspmv[N*i], &INCY);
    clck = clock() - clck;
    printf("dspmv took %f seconds.\n", ((float)clck)/CLOCKS_PER_SEC);



    /*  Check correctness of multiplication algorithms
    n = 8;
    vector<double> A={1,0,0,0,0,0,0,0,
                        1,0,0,2,0,0,0,
                          1,0,0,0,0,0,
                            1,0,0,0,0,
                              1,0,0,0,
                                1,0,0,
                                  1,0,
                                    1};  // A.size() = n*(n+1)/2

    vector<double> x={0,1,2,3,4,5,6,7};
    vector<double> y(n);

    SymmMat_Vec_Mult(n, A, x, y);
    cout << "\nA * x = ( ";
    for (size_t i = 0; i < n-1; i++)   cout << y[i] << ", ";
    cout << y[n-1] << ")^T\n";

    // Test setting A_BTPS from A
    vec AP = conv_to<vec>::from(A);
    A_BTPS.setFromPacked(AP);

    A_BTPS.diagUpper_->fullPart_->print("diagUpper");
    A_BTPS.fullPart_->print("fullPart");
    A_BTPS.diagLower_->fullPart_->print("diagLower");

    mat X = conv_to<mat>::from(x);
    (A_BTPS * X).t().print("A_BTPS * X = ");

    */


    return 0;
}
