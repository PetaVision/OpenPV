//! RETURNS X INDEX FROM LINEAR INDEX
/*!
 * Return the position kx for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *   - since k = ky * (nf*nx) + kx * nf + kf, we easily see first that
 *    a = k/nf = ky*nx + kx, and then that
 *    kx = mod(a,nx), i.e. kx is the reminder of the division of a by nx,
 *    since kx <= nx-1.
 *    .
 */
static inline int kxPos(int k, int nx, int ny, int nf)
{
   return (k/nf) % nx;
}

//! RETURNS Y INDEX FROM LINEAR INDEX
/*!
 * Return the position ky for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *   - since k = ky * (nf*nx) + kx * nf + kf, we easily see first that
 *    kx = floor(k/(nx*nf)) since kx*nf + kf < nx*nf
 *    (note that kx <= nx-1 and kf <= nf-1).
 *   .
 */
static inline int kyPos(int k, int nx, int ny, int nf)
{
   return k / (nx*nf);
}

 //! RETURNS FEATURE INDEX FROM LINEAR INDEX
/**
 * Return the feature index for the given k index
 * @k the k index (can be either global or local depending on if nx,ny are global or local)
 * @nx the number of neurons in the x direction
 * @ny the number of neurons in the y direction
 * @nf the number of neurons in the feature direction
 * REMARKS:
 *      - since k = ky * (nf*nx) + kx * nf + kf, we easily see that
 *      kf = mod(k,nf), i.e. kf it is the reminder of the division of k by nf,
 *      since kf <= nf-1.
 *      .
 */
static inline int featureIndex(int k, int nx, int ny, int nf)
{
   return k % nf;
}

//! RETURNS LINEAR INDEX FROM X,Y, AND FEATURE INDEXES
/*!
 * @kx
 * @ky
 * @kf
 * @nx
 * @ny
 * @nf
 * REMARKS:
 *      - This simply says that:
 *      k = ky * (nf*nx) + kx * nf + kf
 *      .
 */
static inline int kIndex(int kx, int ky, int kf, int nx, int ny, int nf)
{
   return kf + (kx + ky * nx) * nf;
}

//! RETURNS LINEAR INDEX IN THE EXTENDED SPACE FROM INDICES IN RESTRICTED SPACE
/*!
 * @k the k index in restricted space
 * @nx the size in x of restricted space
 * @ny the size in y of restricted space
 * @nf the size in f of restricted space
 * @nb the width of the margin
 *
 * REMARKS:
 *   - the linear indexing of neurons is done by varying first along these directions:
 *   feature direction, X direction, Y direction.
 *   - for given indices kf,kx,ky, the linear index k restricted is given by:
 *     k = ky*(nf*nx) + kx*nf + kf
 *   - kx is the X direction index in restricted space
 *   - ky is the Y direction index in restricted space
 *   .
 */
static inline int kIndexExtended(int k, int nx, int ny, int nf, int nb)
{
   const int kx_ex = nb + kxPos(k, nx, ny, nf);
   const int ky_ex = nb + kyPos(k, nx, ny, nf);
   const int kf = featureIndex(k, nx, ny, nf);
   return kIndex(kx_ex, ky_ex, kf, nx + 2*nb, ny + 2*nb, nf);
}
