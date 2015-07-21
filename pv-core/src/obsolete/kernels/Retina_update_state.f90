!$OFP CL_KERNEL :: update_state
subroutine update_state(time, dt, probStim, probBase,     &
                        nf, nx, ny, nb,                   &
                        phiExc, phiInh, activity, prevTime)
   implicit none
   real, intent(in) :: time, dt, probStim, probBase
   integer, intent(in) :: nf, nx, ny, nb
   real, intent(inout), dimension(nf,nx,ny) :: phiExc, phiInh
   real, intent(inout), dimension(nf,nx+2*nb,ny+2*nb) :: activity, prevTime

   ! local variables
   real, pointer, dimension(:,:,:) :: l_phiExc, l_phiInh, l_activ, l_prev
   !$OFP LOCAL :: l_phiExc, l_phiInh, l_activ, l_prev

   l_phiExc => local(phiExc)
   l_phiInh => local(phiInh)

   l_prev  => local(prevTime, [0,0,nb,nb,nb,nb])
   l_activ => local(activity, [0,0,nb,nb,nb,nb])

   l_activ = spike(time, dt, l_prev, probBase, (phiExc - phiInh)*probStim)

   where (l_activ > 0.0) l_prev = time

   phiExc = 0.0
   phiInh = 0.0

end subroutine update_state
