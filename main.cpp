#include <iostream>
#include <vector>
#include "symm_matrix_mult.h"

using namespace std;

int main()
{
    const int n = 5;
    vector<double> A={1,0,0,0,0,
                        1,0,0,2,
                          1,0,0,
                            1,0,
                              1};  // A.size() = n*(n+1)/2
    vector<double> x={0,1,2,3,4};
    vector<double> y(n);

    SymmMat_Vec_Mult(n, A, x, y);

    // print result
    cout << "\nA * x = ( ";
    for (int i = 0; i < n-1; i++) {
        cout << y[i] << ", ";
    }
    cout << y[n-1] << ")^T";

    return 0;
}
