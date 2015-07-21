/**
 * A file that includes device conversion functions
 * TODO see if we can include conversions.h instead of having this file
 */

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
inline int kIndexExtended(int k, int nx, int ny, int nf, int lt, int rt, int dn, int up)
{
   const int kx_ex = lt + kxPos(k, nx, ny, nf);
   const int ky_ex = up + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + lt + rt, ny + dn + up, nf);
}

}

#endif
