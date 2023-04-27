using Symbolics, LinearAlgebra
A = randn(5,5)
det(A)
L,U = lu(A, NoPivot()) # LU without row swaps
I₅ = I(5) * 1
print(I₅)