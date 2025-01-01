
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve, bicg
from scipy.sparse import csr_matrix, csc_matrix, issparse
from scipy.interpolate import RegularGridInterpolator

class Fluid(object):
    def __init__(self,
                 N=64,
                 dt=0.1,
                 diff=0.001,
                 visc=0.01,
                 force=5.0,
                 dens_source=100.0,
                 solver='iter'):
        self.N = N
        self.dt = dt
        self.diff = diff
        self.visc = visc
        self.force = force
        self.dens_source = dens_source
        if solver in ['direct', 'iter']:
            self.solver = solver
        else:
            print(f'Unsupported solver \'{solver}\' specified.  Reverted back to iter')
        # TODO: Extend to > 2d support
        self.v = np.zeros((self.N + 2, self.N + 2, 2))
        self.dens = np.zeros((self.N + 2, self.N + 2, 1))
        self.A = self._build_Amat()
        self.coeff_density = self._build_density_coeff()
        self.coeff_velocity = self._build_velocity_coeff()
        self.coeff_project = self._build_project_coeff()

    def _build_Amat(self):
        # TODO extend to > 2d support
        A = np.fromfunction(self._build_matrix, [self.N + 2] * 4)
        # Convert to 2d matrix for linear solve
        A = A.reshape(A.shape[0] ** 2, -1)
        A = A.astype('int8')
        '''
        A[0] = 0
        A[-1] = 0
        A[..., 0] = 0
        A[..., -1] = 0
        '''
        return A


    @staticmethod
    # TODO: Extend to > 2d support
    def _build_matrix(x, y, i, j):
        return ((x == i + 1) & (y == j)) | \
               ((x == i - 1) & (y == j)) | \
               ((x == i) & (y == j + 1)) | \
               ((x == i) & (y == j - 1))

    def _build_density_coeff(self):
        '''
        x0 = x - a(xt + xb + xl + xr - 4x)
        x0 = (1+4a)x - axt - axb - axl - axr
        Ax = x0
        '''
        a = self.dt * self.diff * pow(self.N, 2)
        coeff = -a * self.A + np.eye(self.A.shape[0]) * (1 + 4 * a)
        return csc_matrix(coeff)

    def _build_velocity_coeff(self):
        '''
        x0 = x - a(xt + xb + xl + xr - 4x)
        x0 = (1+4a)x - axt - axb - axl - axr
        Ax = x0
        '''
        a = self.dt * self.visc * pow(self.N, 2)
        coeff = -a * self.A + np.eye(self.A.shape[0]) * (1 + 4 * a)
        return csc_matrix(coeff)

    def _build_project_coeff(self):
        '''
        x0 = 4x - xt - xb - xl - xr
        '''
        coeff = -self.A + np.eye(self.A.shape[0]) * 4
        return csc_matrix(coeff)

    def _itsolve2(self, x, x0, N, a, c):
        iter_method = 'jacobi'
        for k in range(0, 20):
            # Jacobi method rather than gauss siedel method used
            # by in the original c implementation
            if iter_method == 'jacobi':
                x[1:N + 1, 1:N + 1] = (x0[1:N + 1, 1:N + 1] + a *
                                       (x[0:N, 1:N + 1] +
                                       x[2:N + 2, 1:N + 1] +
                                       x[1:N + 1, 0:N] +
                                       x[1:N + 1, 2:N + 2])) / c
            else: # gauss siedel
                for i in range(1, N+1):
                    for j in range(1, N+1):
                        x[i, j] = (x0[i, j] + a * (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) / c
            self._set_bound(x)
        return x

    def _solve(self, X, N, coeff):
        X_ori_shape = X.shape
        X = X.reshape(coeff.shape[0], -1)
        if self.solver == 'direct':
            X = spsolve(coeff, X)
        elif self.solver == 'iter':
            for axis in range(X.shape[-1]):
                X[..., axis] = bicg(coeff, X[..., axis], x0=X[..., axis].copy())[0]
        else:
            raise ValueError(f'Unsupported solver: {self.solver}.  Supported solver type is [\'direct\', \'iter\'].')
        X = X.reshape(X_ori_shape)
        return X

    def _advect(self, X):
        # linear backtrace
        # TODO How to extend to RK2 solver as mentioned in the paper
        coords = [coord[..., np.newaxis] for coord in np.meshgrid(*[np.arange(self.N)] * (X.ndim - 1), indexing='ij')]
        coords = np.concat(coords, axis=X.ndim - 1)
        #coords = np.einsum('ij...->ji...', coords)
        coords = coords + 1.0
        transport = coords - self.dt * self.N * self.v[1:-1,1:-1,:] # Leaving out boundary cells
        transport[transport < 0.5] = 0.5
        transport[transport > (self.N + 0.5)] = self.N + 0.5
        grids = list((np.arange(i) for i in X.shape[:-1]))
        interp = RegularGridInterpolator(grids, X)
        X[1:-1,1:-1,:] = interp(transport)
        return X

    def add_density_source(self, source):
        self.dens += source[..., [0]] * self.dt

    def diffuse_density(self):
        if self.solver == 'fallback':
            a = self.dt * self.diff * self.N * self.N
            c = 1 + 4 * a
            self.dens = self._itsolve2(self.dens.copy(), self.dens, self.N, a, c)
        else:
            self.dens = self._solve(self.dens, self.N, self.coeff_density)

    def advect_density(self):
        self.dens = self._advect(self.dens)

    @staticmethod
    def _set_bound(x):
        # Processing along the edges first
        x[0] = x[1]
        x[-1] = x[-2]
        x[:, 0] = x[:, 1]
        x[:, -1] = x[:, -2]
        # Velocity bounces back after hitting the wall
        # Assuming density is scalar so there should not be ambiguity here
        # TODO How to generalize to 3D and higher dimensional
        if x.shape[-1] > 1:
            # u component
            x[0, ..., 0] = -x[0, ..., 0]
            x[-1, ..., 0] = -x[-1, ..., 0]
            # v component
            x[..., 0, 1] = -x[..., 0, 1]
            x[..., -1, 1] = -x[..., -1, 1]
            '''
            # u component
            x[..., 0, 0] = -x[..., 0, 0]
            x[..., -1, 0] = -x[..., -1, 0]
            # v component
            x[0, ..., 1] = -x[0, ..., 1]
            x[-1, ..., 1] = -x[-1, ..., 1]
            '''
        # Lastly, special handling for the four corners
        x[0, 0] = 0.5 * (x[0, 1] + x[1, 0])
        x[-1, 0] = 0.5 * (x[-1, 1] + x[-2, 0])
        x[0, -1] = 0.5 * (x[0, -2] + x[1, -1])
        x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])
        return x

    def set_density_bound(self):
        self.dens = self._set_bound(self.dens)

    def update_density(self, source):
        self.add_density_source(source)
        self.diffuse_density()
        self.set_density_bound()
        self.advect_density()
        self.set_density_bound()

    def get_density(self):
        return self.dens

    def set_velocity_bound(self):
        self.v = self._set_bound(self.v)

    def add_velocity_source(self, source):
        self.v += source[..., 1:] * self.dt

    def diffuse_velocity(self):
        if self.solver == 'fallback':
            a = self.dt * self.visc * self.N * self.N
            c = 1 + 4 * a
            self.v = self._itsolve2(self.v.copy(), self.v, self.N, a, c)
        else:
            self.v = self._solve(self.v, self.N, self.coeff_velocity)

    def advect_velocity(self):
        # Self advection
        self.v = self._advect(self.v)

    def project_velocity(self):
        v_dim = self.v.shape[-1]
        gradient_components = np.gradient(self.v, axis=np.arange(v_dim))
        # divergence is scalar
        # No need to divide 0.5 again as compared with the original implementation
        # Because np.gradient already divide by half
        divergence = np.add.reduce([m[..., i] for i, m in enumerate(gradient_components)])
        divergence = -divergence[..., np.newaxis] / self.N
        divergence = self._set_bound(divergence)
        #print('before')
        #print(divergence[:, :, 0])
        if self.solver == 'direct':
            a = 1.
            c = 4.
            divergence = self._itsolve2(np.zeros(divergence.shape), divergence, self.N, a, c)
        else:
            divergence = self._solve(divergence, self.N, self.coeff_project)
        #print('after')
        #print(divergence[:, :, 0])
        p = np.gradient(divergence, axis=np.arange(v_dim))
        p = np.stack(p, axis=v_dim).squeeze()
        self.v = self.v - (self.N * p)
        self.v = self._set_bound(self.v)

    def update_velocity(self, source):
        self.add_velocity_source(source)
        self.diffuse_velocity()
        self.project_velocity()
        self.set_velocity_bound()
        self.advect_velocity()
        self.project_velocity()
        self.set_velocity_bound()

    def get_velocity(self):
        return self.v

    def reset(self):
        self.dens = np.zeros(self.dens.shape)
        self.v = np.zeros(self.v.shape)

