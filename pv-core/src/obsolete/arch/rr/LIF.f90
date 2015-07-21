module LIF

implicit none

contains

!$ FTT vectorize
elemental real function updateF(v, vth) result(f)
    real, intent(in) :: v, vth

    if ( (v - vth) > 0.0 ) then
        f = 1.0
    else
        f = 0.0
    end if

end function updateF

!$ FTT vectorize
elemental real function updateV(v, f) result(vnew)
    real, intent(in) :: v, f

    vnew = v - f*v

end function updateV

!$ FTT vectorize
pure subroutine update(V, F, vth)
    real, intent(inout) :: V(:), F(:)
    real, intent(in) :: vth

    F = updateF(V, vth)
    V = updateV(V, F)

end subroutine update

end module LIF




