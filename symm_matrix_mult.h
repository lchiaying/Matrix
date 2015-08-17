#ifndef SYMM_MATRIX_MULT_H_INCLUDED
#define SYMM_MATRIX_MULT_H_INCLUDED

#include <iostream>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace std;


/*
*   Symmetric matrix class, implemented in binary-tree-packed storage (BTPS)
*/
unsigned figureTreePackedDepth(unsigned dim, unsigned max_base_dim = 64, unsigned max_depth = 5) {
    return min(max_depth, ((unsigned)ceil( log2(dim * 1. / max_base_dim) ) ) );
}


template <class T>
class SymmetricMatrix {
private:
    arma::Mat<T> fullPart_;         // )    A = [ diagUpper     fullPart  ]
    SymmetricMatrix * diagUpper_;   // )        [ fullPart.t    diagLower ]
    SymmetricMatrix * diagLower_;   // )
    size_t dim_;                    // dimension of matrix
    unsigned depth_;                // recursive depth of tree-packed storage structure

public:
    SymmetricMatrix() {};
    SymmetricMatrix(size_t dim, unsigned depth);
    ~SymmetricMatrix() { if (depth_) { delete [] diagLower_, diagUpper_;} };

    arma::Mat<T> operator* (const arma::Mat<T> & X);
};

template <class T>
SymmetricMatrix<T>::SymmetricMatrix(size_t dim, unsigned depth) : dim_(dim), depth_(depth) {
    if (depth_) {
        size_t mid = dim_ / 2 + 1;
        fullPart_ = arma::randu< arma::Mat<T> >(mid, dim_ - mid);
        diagUpper_ = new SymmetricMatrix(mid, depth_ - 1);
        diagLower_ = new SymmetricMatrix(dim_ - mid, depth_ - 1);
    } else {
        fullPart_ = arma::randu< arma::Mat<T> >(dim_, dim_);
    }
}

template <class T>
arma::Mat<T> SymmetricMatrix<T>::operator* (const arma::Mat<T> & X) {
    if (depth_)
    {
        size_t mid = dim_ / 2 + 1;

        arma::Mat<T> Y(X.n_rows, X.n_cols, arma::fill::none);
        arma::Mat<T> Xupper = X.submat( arma::span(0, mid-1), arma::span::all );
        arma::Mat<T> Xlower = X.submat( arma::span(mid, dim_-1), arma::span::all );

        Y.submat( arma::span(0, mid-1), arma::span::all ) = diagUpper_ * Xupper;
        Y.submat( arma::span(0, mid-1), arma::span::all ) += fullPart_ * Xlower;

        Y.submat( arma::span(mid, dim_-1), arma::span::all ) = diagLower_ * Xlower;
        Y.submat( arma::span(mid, dim_-1), arma::span::all ) += fullPart_.t() * Xupper;

        return Y;
    }
    else
    {
        return fullPart_ * X;
    }
}








/*
*   Symmetric matrix- vector multiplication.
*/
void SymmMat_Vec_Mult(const size_t n, const vector<double> &A, const vector<double> &x, vector<double> &y) {
    // Symmetric matrix-vector multiplication: A * x = y
    // A is n x n, represented in packed storage (including the diagonal)
    // Output is written into y.
    // Lengths must be at least n*(n+1)/2 for A and n for x,y. Extra entries are ignored, but note that
    // the packed storage assumption of A as a vector means that a submatrix of A is *not* the same as a
    // single (entry-stepwise) pass through the first few entries of A
    if (A.size() < n*(n+1)/2 || x.size() < n || y.size() < n) {
        cout << "\nWrong dimension of inputs.\n";
        return;
    }

    vector<double>::const_iterator  itr_A = A.begin(), itr_xi = x.begin(), itr_xj = itr_xi;
    vector<double>::iterator        itr_yi = y.begin(), itr_yj = itr_yi;

    for (size_t i = 0; i < n; i++, itr_xi++, itr_yi++) {
        double xi = *itr_xi;
        *itr_yi += (*itr_A++) * xi;

        itr_xj = itr_xi + 1;
        itr_yj = itr_yi + 1;

        for (size_t j = i+1; j < n; j++, itr_xj++, itr_yj++) {
            double a_ij = *itr_A++;
            *itr_yi += a_ij * (*itr_xj);
            *itr_yj += a_ij * xi;
        }
    }

    return;
}




#endif // SYMM_MATRIX_MULT_H_INCLUDED
