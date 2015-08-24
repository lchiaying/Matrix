#ifndef SYMM_MATRIX_MULT_H_INCLUDED
#define SYMM_MATRIX_MULT_H_INCLUDED

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;


// Choose a suitable depth. The smallest subblock will have dimension no larger than max_base_dim.
unsigned figureTreePackedDepth(unsigned dim, unsigned max_base_dim = 8) {
    return ((unsigned) ceil( log2(dim * 1. / max_base_dim) )) ;
}



/*
*   Symmetric matrix class, implemented in binary-tree-packed storage (BTPS)
*/

template <class T>
class SymmetricMatrix {
public:
    Mat<T> * fullPart_;                 // )    A = [ diagUpper   fullPart.t ]
    SymmetricMatrix<T> * diagUpper_;    // )        [ fullPart    diagLower  ]
    SymmetricMatrix<T> * diagLower_;    // )
    size_t dim_;                            // dimension of matrix
    unsigned depth_;                        // recursive depth of tree-packed storage structure
    size_t idx_begin_, idx_mid_, idx_end_;  // If part of a supermatrix, keep track of subindices within the supermatrix

public:
    SymmetricMatrix() {};
    SymmetricMatrix(size_t dim, unsigned depth, string fill_ty, size_t idx_begin);
    ~SymmetricMatrix() { delete fullPart_; if (depth_) {delete diagLower_; delete diagUpper_; }  };

    Mat<T> operator* (const Mat<T> & X);

    T getij(size_t i, size_t j);
    void setij(size_t i, size_t j, T value);
    void setFromPacked(const Col<T> & AP);
};


template <class T>
SymmetricMatrix<T>::SymmetricMatrix(size_t dim, unsigned depth, string fill_ty = "none", size_t idx_begin = 0) :
    dim_(dim), depth_(depth), idx_begin_(idx_begin), idx_end_(idx_begin + dim) {
    if (depth_) {
        size_t mid = dim_/2;
        idx_mid_ = idx_begin + mid;

        fullPart_ = new Mat<T>(dim_ - mid, mid, fill::none);
        if (fill_ty == "randn")         fullPart_->randn();
        else if (fill_ty == "zeros")    fullPart_->zeros();

        diagUpper_ = new SymmetricMatrix<T>(mid, depth_ - 1, fill_ty, idx_begin_);
        diagLower_ = new SymmetricMatrix<T>(dim_ - mid, depth_ - 1, fill_ty, idx_begin_ + mid);
    } else
        fullPart_ = new Mat<T>(dim_, dim_, fill::none);
        if (fill_ty == "zeros")         fullPart_->zeros();
        else if (fill_ty == "randn"){   fullPart_->randn(); *fullPart_ += (*fullPart_).t(); }
}


/* Algebraic operator */
template <class T>
Mat<T> SymmetricMatrix<T>::operator* (const Mat<T> & X) {
    if (depth_)
    {
        size_t mid = dim_ / 2;

        Mat<T> Y(X.n_rows, X.n_cols, fill::none);
        Mat<T> Xupper = X.submat( span(0, mid-1), span::all );
        Mat<T> Xlower = X.submat( span(mid, dim_-1), span::all );

        Y.submat( span(0, mid-1), span::all ) = *diagUpper_ * Xupper;
        Y.submat( span(0, mid-1), span::all ) += (*fullPart_).t() * Xlower;

        Y.submat( span(mid, dim_-1), span::all ) = *diagLower_ * Xlower;
        Y.submat( span(mid, dim_-1), span::all ) += *fullPart_ * Xupper;

        return Y;
    }
    else return *fullPart_ * X;
}

/* Element access */
template <class T>
T SymmetricMatrix<T>::getij(size_t i, size_t j) {
    if (i < idx_begin_ || i >= idx_end_ || j < idx_begin_ || j >= idx_end_) {
        printf("Index error for (%i,%i)!\n", i, j);
        return 0;
    }
    if (i < j) swap(i,j);

    if (depth_) {
        if (i < idx_mid_)       return diagUpper_->getij(i, j);
        else if (j < idx_mid_)  return (*fullPart_)(i-idx_mid_, j-idx_begin_);
        else                    return diagLower_->getij(i, j);
    }
    else return (*fullPart_)(i-idx_begin_, j-idx_begin_);
}

template <class T>
void SymmetricMatrix<T>::setij(size_t i, size_t j, T value) {
    if (i < idx_begin_ || i >= idx_end_ || j < idx_begin_ || j >= idx_end_) {
        printf("Index error for (%i,%i)!\n", i, j);
        return;
    }
    if (i < j) swap(i,j);

    if (depth_) {
        if (i < idx_mid_)       diagUpper_->setij(i, j, value);
        else if (j < idx_mid_)  (*fullPart_)(i-idx_mid_, j-idx_begin_) = value;
        else                    diagLower_->setij(i, j, value);
    } else {
        (*fullPart_)(i-idx_begin_, j-idx_begin_) = value;
        (*fullPart_)(j-idx_begin_, i-idx_begin_) = value;
    }
}


template <class T>
void SymmetricMatrix<T>::setFromPacked(const Col<T> & AP) {
    if (AP.size() != dim_ * (dim_+1) / 2) {cout << "*** Error! Wrong input size. ***\n"; return;}

    if (depth_)
    {
        size_t mid = dim_ / 2;

        // Full part and upper diagonal
        Col<T> AP_upper(mid * (mid+1) / 2);
        for (size_t j = 0; j < mid; j++) {
            size_t col_last = (2*dim_ - j) * (j+1) / 2;
            size_t col_mid = col_last - (dim_-mid);
            size_t col_first = col_last - (dim_-j); //(2*dim_ - j + 1) * j / 2;
            size_t upper_col_last = (2*mid - j) * (j+1) / 2;
            size_t upper_col_first = upper_col_last - (mid-j); // (2*mid - j + 1) * j / 2;

            fullPart_->col(j) = AP.subvec(span( col_mid, col_last-1 ));
            AP_upper(span( upper_col_first, upper_col_last-1 )) = AP.subvec(span( col_first, col_mid-1 ));
        }
        diagUpper_->setFromPacked(AP_upper);

        // Lower diagonal part
        Col<T> AP_lower = AP.subvec( span( (2*dim_ - mid + 1) * mid/2 , AP.size() - 1 ) );
        diagLower_->setFromPacked(AP_lower);
    }
    else
    {
        typename Col<T>::const_iterator itr = AP.begin();
        for (size_t j = 0; j < dim_; j++) {
            (*fullPart_)(j,j) = *itr++;
            for (size_t i = j + 1; i < dim_; i++) {
                T a_ij = *itr++;
                (*fullPart_)(i,j) = a_ij;
                (*fullPart_)(j,i) = a_ij;
            }
        }
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
