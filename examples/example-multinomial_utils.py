
import numpy as np
import torch


class MatrixOps(object):
    """Collection of matrix operations for positive definite matrices."""
    
    @classmethod
    def is_positive_definite(cls, A: torch.Tensor) -> bool:
        try:
            _ = torch.linalg.cholesky(A)
            return True
        except RuntimeError:
            return False

    @classmethod
    def is_semipositive_definite(cls, A: torch.Tensor, tol: float = 1e-7) -> bool:
        """A matrix is p.s.d if all its eigenvalues are non-negative."""
        eigvals, _ = torch.linalg.eigh(A)
        return eigvals.add(tol).ge(0.).all()
    
    @classmethod
    def _getAplus(cls, A: torch.Tensor) -> torch.Tensor:
        eigval, eigvec = torch.linalg.eigh(A)
        _zeros = torch.zeros_like(eigval)
        return eigvec @ torch.diag(torch.maximum(eigval, _zeros)) @ eigvec.T

    @classmethod
    def _getPs(cls, A: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        W = torch.eye(A.shape[0], device=A.device) if W is None else W
        W05 = W ** 0.5
        W05_inv = torch.linalg.inv(W05)
        return W05_inv @ cls._getAplus(W05 @ A @ W05) @ W05_inv

    @classmethod
    def _getPu(cls, A: torch.Tensor, W: torch.Tensor = None) -> torch.Tensor:
        Aret = A.clone()
        Aret[W > 0] = W[W > 0]
        return Aret

    @classmethod
    def get_nearest_psd_correlation_matrix(cls, A: torch.Tensor, nit: int = 5, tol: float = 1e-7) -> torch.Tensor:
        """
        Get nearest psd correlation matrix (where diagonals equal 1).
        Reference:
            # https://stackoverflow.com/questions/10939213/how-can-i-calculate-the-nearest-positive-semi-definite-matrix/18542094#18542094
        """
        n: int = A.shape[0]
        W = torch.eye(n, device=A.device)
        deltaS = 0
        Yk = A.clone()
        for _ in range(nit):
            Rk = Yk - deltaS
            Xk = cls._getPs(Rk, W=W)
            deltaS = Xk - Rk
            Yk = cls._getPu(Xk, W=W)
            if cls.is_semipositive_definite(Yk, tol=tol):
                return Yk
        return Yk

    @classmethod
    def make_positive_definite(cls, A: torch.Tensor) -> torch.Tensor:
        """
        Find the nearest positive-definite matrix to input.
        A Python/Pytorch port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        [3] https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194#43244194
        """
        
        if cls.is_positive_definite(A):
            return A
        
        B = (A + A.T) / 2  # symmetrize
        _, s, V = torch.linalg.svd(B)
        H = V.T @ torch.diag(s) @ V
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if cls.is_positive_definite(A3):
            return A3

        norm = np.linalg.norm(A.detach().cpu().numpy())
        eps: float = np.spacing(norm)
        I = torch.eye(A.shape[0], device=A.device)

        count: int = 1
        while not cls.is_positive_definite(A3):
            eigvals, _ = torch.linalg.eigh(A3)
            A3 -= (eigvals.min() * count ** 2 + eps) * I
            count += 1

        return A3

    @classmethod
    def delete_row_from_matrix(cls, m: torch.Tensor, index: int) -> torch.Tensor:
        """(R,C) -> (R-1,C)"""
        if (index < 0) or (index > m.size(0) - 1):
            raise IndexError(f"Index {index} out of bounds.")
        return torch.cat([m[:index, :], m[index+1:, :]], dim=0)

    @classmethod
    def delete_column_from_matrix(cls, m: torch.Tensor, index: int) -> torch.Tensor:
        """(R,C) -> (R,C-1)"""
        if (index < 0) or (index > m.size(1) - 1):
            raise IndexError(f"Index {index} out of bounds.")
        return torch.cat([m[:, :index], m[:, index+1:]], dim=1)

    @classmethod
    def delete_index_from_matrix(cls, m: torch.Tensor, index: int) -> torch.Tensor:
        """(R,C) -> (R-1,C-1)"""
        m = cls.delete_row_from_matrix(m, index=index)
        m = cls.delete_column_from_matrix(m, index=index)
        return m

    @classmethod
    def compute_cov_of_error_differences(cls, M: torch.FloatTensor, j: int = 0) -> torch.FloatTensor:
        """(J,J) -> (J-1,J-1)"""
        out = M[j,j] \
            - cls.delete_row_from_matrix(M, index=j)[:, j].view(-1, 1) \
            - cls.delete_column_from_matrix(M, index=j)[j, :].view(1, -1) \
            + cls.delete_index_from_matrix(M, index=j)

        return out

    @classmethod
    def cov_to_corr(cls, A: torch.Tensor):
        # print('A: ', A)
        D = torch.sqrt(A.diag().diag() + 1e-7)
        # print('D: ', D)
        DInv = torch.linalg.inv(D)
        return DInv @ A @ DInv
