!$ FTT vectorize
elemental real function accumulate(v, a, w)
   real, intent(in) :: v, a, w
   accumulate = v + a*w
end function accumulate


!$ FTT vectorize
pure subroutine hyperpatch_accumulate(V, a, W, kx0, ky0, kf)
   interface
      elemental real function accumulate(v, a, w)
         real, intent(in) :: v, a, w
      end function accumulate
   end interface
   real, intent(inout) :: V(:,:,:)
   real, intent(in)    :: W(:,:)
   real, intent(in)    :: a
   integer, intent(in) :: kx0, ky0, kf

   V(kx0:kx0+8,ky0:ky0+8,kf) = accumulate(V(kx0:kx0+8,ky0:ky0+8,kf), a, W)

end subroutine hyperpatch_accumulate

