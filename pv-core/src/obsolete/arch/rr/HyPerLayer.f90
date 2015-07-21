module HyPerLayer
implicit none

real, parameter :: FP_INT_CORR = 0.5

contains

!$FTT vectorize
elemental real function xPos(idx, nx, x0, dx, numFeatures) result(x)
    integer, intent(in) :: idx, nx, numFeatures
    real,    intent(in) :: dx, x0
        
    x = x0 + dx*(0.5 + mod(idx/numFeatures, nx))
    
end function xPos

!$FTT vectorize
elemental real function yPos(idx, nx, y0, dy, numFeatures) result(y)
    integer, intent(in) :: idx, nx, numFeatures
    real,    intent(in) :: dy, y0
        
    y = y0 + dy*(0.5 + (idx/(nx*numFeatures)))

end function yPos

!$FTT vectorize
elemental integer function featureIndex(idx, numFeatures) result(kf)
    integer, intent(in) :: idx, numFeatures
        
    kf = mod(idx, numFeatures)

end function featureIndex

!$FTT vectorize
elemental integer function index(kf, x, y, x0, y0, dx, dy, nx, numFeatures) result(idx)
    real, intent(in)    :: x, y, x0, y0, dx, dy
    integer, intent(in) :: kf, nx, numFeatures
    integer :: kx, ky    

    kx = INT( (x - x0)/dx - 0.5 + FP_INT_CORR )
    ky = INT( (y - y0)/dy - 0.5 + FP_INT_CORR )
    idx = (kx + nx*ky)*numFeatures + kf
    
end function index

end module HyperLayer
