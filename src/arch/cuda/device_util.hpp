#ifndef DEVICE_UTIL_HPP_
#define DEVICE_UTIL_HPP_

namespace PVCuda{

#define kxPos(k,nx,ny,nf) ((k/nf)%nx)
#define kyPos(k,nx,ny,nf) (k/(nx*nf))
#define featureIndex(k,nx,ny,nf) (k%nf)

__device__
inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
{
   return kf + (kx + ky * nx) * nf;
}

__device__
inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   const int kx_ex = nb + kxPos(k, nx, ny, nf);
   const int ky_ex = nb + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
}


}

#endif
