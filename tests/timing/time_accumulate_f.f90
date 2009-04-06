elemental real function accumulate_elemental(v, a, w)
   real, intent(in) :: v, a, w
   accumulate_elemental = v + a*w
end function accumulate_elemental

subroutine accumulate(n, v, a, w)
   real :: v(n), a, w(n)
   integer :: i
   do k = 1, n
      v(k) = v(k) + a*w(k)
   end do
end subroutine accumulate


program time_accumulate
   implicit none

   interface
      elemental real function accumulate_elemental(v, a, w)
         real, intent(in) :: v, a, w
      end function accumulate_elemental

      subroutine start_clock() BIND(C)
      end subroutine start_clock

      subroutine stop_clock() BIND(C)
      end subroutine stop_clock

      double precision function elapsed_time() BIND(C)
      end function elapsed_time
   end interface

   integer, parameter :: nx = 64
   integer, parameter :: ny = 64
   integer, parameter :: nb = 8
   integer, parameter :: nf = 4
   integer, parameter :: nxw = 4
   integer, parameter :: nyw = 4

   integer, parameter :: nloops = 1

   real, dimension(nf,nx,ny)   :: phi(nf,nx,ny)
   real, dimension(nf,nxw+nb,nyw+nb) :: w
   real :: a = 2

   integer :: i, j, k, t
   integer :: count

   print *, "starting init w ...."

!  ifort does not do array syntax very well without -O0
   w = 0.5

   print *, "starting init a ...."

   print *, "starting loop...."

   call start_clock()

   do t = 1, nloops

   do j = 1, ny
      do i = 1, nx
         phi(:,i:i+nxw-1,j:j+nyw-1) &
            = accumulate_elemental(phi(:,i:i+nxw-1,j:j+nyw-1), a, w)
      end do
   end do

!   do j = 1, ny
!      do i = 1, nx
!         do k = 1, nf
!            phi(k,i,j) = accumulate_elemental(phi(k,i,j), a, w(k,1,1))
!         end do
!      end do
!   end do

   end do

   call stop_clock()

   print *, "elapsed time is ", real(elapsed_time())

   print *
   print *, "junk value is", phi(1,32,32), " count is ", count

end program
