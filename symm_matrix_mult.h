#ifndef SYMM_MATRIX_MULT_H_INCLUDED
#define SYMM_MATRIX_MULT_H_INCLUDED

#include <iostream>
#include <vector>

using namespace std;

void SymmMat_Vec_Mult(const int n, const vector<double> &A, const vector<double> &x, vector<double> &y) {
    // Symmetric matrix-vector multiplication: A * x = y
    // A is n x n, represented in packed storage (including the diagonal)
    // Output is written into y.
    // Lengths must be at least n*(n+1)/2 for A and n for x,y. Extra entries are ignored, but note that
    // the packed storage assumption of A as a vector means that a submatrix of A is *not* the same as a
    // single (entry-stepwise) pass through the first few entries of A
    if (A.size() < n*(n+1)/2 || x.size() < n || y.size() < n) {
            cout << "\nWrong dimension of inputs.\n"; return;
    }

    vector<double>::const_iterator  ptr_A = A.begin(), ptr_xi = x.begin(), ptr_xj = ptr_xi;
    vector<double>::iterator        ptr_yi = y.begin(), ptr_yj = ptr_yi;

    for (int i = 0; i < n; i++) {
        double xi = *ptr_xi;
        *ptr_yi += (*ptr_A) * xi;
        ptr_xj++;   ptr_yj++;   ptr_A++;

        for (int j = i+1; j < n; j++) {
            double a_ij = *ptr_A;
            *ptr_yi += a_ij * (*ptr_xj);
            *ptr_yj += a_ij * xi;
            ptr_xj++;   ptr_yj++;   ptr_A++;
        }

        ptr_xj = ++ptr_xi;
        ptr_yj = ++ptr_yi;
    }

    return;
}

#endif // SYMM_MATRIX_MULT_H_INCLUDED
