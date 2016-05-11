module HyPerLayer
implicit none

real, parameter :: FP_INT_CORR = 0.5

type, BIND(C) :: PVLayerLoc
   real nx, ny
   real nxGlobal, nyGlobal ! total number of neurons in (x,y) across all hypercolumns
   real kx0, ky0  ! origin of the layer in index space
   real dx, dy;   ! maybe not needed but can use for padding anyway
end type PVLayerLoc

contains

!$FTT vectorize
real function featureIndex(k, nx, ny, nf) result(kf) BIND(C, name="featureIndex")
    integer, value, intent(in) :: k
    real,    value, intent(in) :: nx, ny, nf

    kf = MOD(REAL(k), nf)

end function featureIndex

!$FTT vectorize
real function kxPos(k, nx, ny, nf) result(kx) BIND(C, name="kxPos")
    integer, value, intent(in) :: k
    real,    value, intent(in) :: nx, ny, nf

   kx = FLOOR( MOD( REAL(FLOOR(REAL(k) / nf)), nx ) )

end function kxPos

!$FTT vectorize
real function kyPos(k, nx, ny, nf) result(ky) BIND(C, name="kyPos")
    integer, value, intent(in) :: k
    real,    value,  intent(in) :: nx, ny, nf

   ky = FLOOR( (REAL(k) / (nx*nf)) );

end function kyPos

!$FTT vectorize
integer function kIndexLocal(kx, ky, kf, nx, ny, nf) result(k) BIND(C, name="kIndexLocal")
    real, value, intent(in) :: kx, ky, kf, nx, ny, nf

    k = kf + (kx + ky * nx) * nf

end function kIndex

!$FTT vectorize
integer function kIndex(kx, ky, kf, nx, ny, nf) result(k) BIND(C, name="kIndex")
    real, value, intent(in) :: kx, ky, kf, nx, ny, nf
    integer :: ikx, iky, ikf, inx, iny, inf

    ikx = kf; iky = ky; ikf = kf
    inx = nx; iny = ny; inf = nf

    k = ikf + (ikx + iky * inx) * inf

end function kIndex

integer function gkIndex(loc) result(k) BIND(C)
    type(PVLayerLoc) :: loc

    k = loc%nx

end function gkIndex

!$FTT vectorize
integer function globalIndexFromLocal(kl, loc, nf) result(k) BIND(C, name="globalIndexFromLocal")
    integer, value, intent(in) :: kl
    real,    value, intent(in) :: nf
    type(PVLayerLoc), intent(in) :: loc

    real :: kxg, kyg, kf

   kxg = loc%kx0 + kxPos(kl, loc%nx, loc%ny, nf)
   kyg = loc%ky0 + kyPos(kl, loc%nx, loc%ny, nf)
   kf = featureIndex(kl, loc%nx, loc%ny, nf)

   k = kIndex(kxg, kyg, kf, loc%nxGlobal, loc%nyGlobal, nf)

end function globalIndexFromLocal



!
! OLD functions
!

!$FTT vectorize
elemental real function xPos(idx, nx, x0, dx, numFeatures) result(x)
    integer, intent(in) :: idx, nx, numFeatures
    real,    intent(in) :: dx, x0

    x = x0 + dx*(0.5 + MOD((idx/numFeatures),nx));

end function xPos

!$FTT vectorize
elemental real function yPos(idx, nx, y0, dy, numFeatures) result(y)
    integer, intent(in) :: idx, nx, numFeatures
    real,    intent(in) :: dy, y0
    integer :: ky

    ky = idx / INT(REAL(nx) * REAL(numFeatures) + FP_INT_CORR)
    y  = y0 + dy*(0.5 + ky)

end function yPos

!$FTT vectorize
elemental integer function globalIndex(kf, x, y, x0, y0, dx, dy, nx, numFeatures) result(idx)
    real, intent(in)    :: x, y, x0, y0, dx, dy
    integer, intent(in) :: kf, nx, numFeatures
    integer :: kx, ky

    kx = INT( (x - x0)/dx - 0.5 + FP_INT_CORR )
    ky = INT( (y - y0)/dy - 0.5 + FP_INT_CORR )

    idx = (kx + nx*ky)*numFeatures + kf

end function globalIndex

end module HyperLayer
