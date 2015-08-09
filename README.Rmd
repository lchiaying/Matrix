# Symmetric matrix-vector multiplicaton

A simple, single-pass algorithm for multiplication of a symmetric matrix $A$ in packed storage with a vector $x$. The entries of $A$ are traversed exactly once, making it efficient for large matrices, or streamed input.


### Implementation and usage
The entries of symmetric $n \times n$ matrix $A$, *including the diagonal*, are given in a single array `a` that fills the matrix column-/row-wise. Specifically, the diagonal and upper triangular part of $A$ is

\[\begin{matrix} 
\text{a[0]} & \text{a[1]} & \text{a[2]}   & \dots & \text{a[n-1]} \\ 
            & \text{a[n]} & \text{a[n+1]} & \dots & \text{a[2n-2]}\\
            &             & \ddots        & \vdots& \vdots        \\
            &             &               & \ddots& \vdots        \\
            &             &               &       & \text{a[n*(n+1)/2]}
\end{matrix}\]

The lower triangular part, of course, is the reflection across the diagonal. In general, $A_{ij}=A_{ji}=$`a[(2*n-i+1)*i/2+(j-i)]`, for $1\leq i\leq j\leq n$.

For $y = Ax$, the entry $A_{ij} = A_{ji}$ is used exactly twice: once in the sum $y_i = \sum_{j} A_{ij} x_j$ and once in the sum $y_j = \sum_{i} A_{ji} x_i$. This allows for a single pass through $A$, employing two pairs of pointers to `x[i]`, `x[j]` and `y[i]`, `y[j]`.


