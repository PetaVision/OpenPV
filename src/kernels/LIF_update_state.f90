module LIF

   use, intrinsic :: ISO_C_BINDING

   type, bind(C) :: LIF_params
      real(C_FLOAT) :: Vrest
      real(C_FLOAT) :: Vexc
      real(C_FLOAT) :: Vinh
      real(C_FLOAT) :: VinhB

      real(C_FLOAT) :: tau
      real(C_FLOAT) :: tauE
      real(C_FLOAT) :: tauI
      real(C_FLOAT) :: tauIB

      real(C_FLOAT) :: VthRest
      real(C_FLOAT) :: tauVth
      real(C_FLOAT) :: deltaVth

      real(C_FLOAT) :: noiseFreqE
      real(C_FLOAT) :: noiseAmpE
      real(C_FLOAT) :: noiseFreqI
      real(C_FLOAT) :: noiseAmpI
      real(C_FLOAT) :: noiseFreqIB
      real(C_FLOAT) :: noiseAmpIB
   end type LIF_params

contains

!$OFP CL_KERNEL :: update_state
subroutine update_state(time, dt, params,         &
                        nf, nx, ny, nb,           &
                        V, Vth, G_E, G_I, G_IB,   &
                        phiExc, phiInh, phiInhB,  &
                        activity)
   implicit none
   real, intent(in) :: time, dt
   type(LIF_Params), intent(in) :: params
   integer, intent(in) :: nf, nx, ny, nb
   real, intent(inout), dimension(nf,nx,ny) :: V, Vth, G_E, G_I, G_IB
   real, intent(inout), dimension(nf,nx,ny) :: phiExc, phiInh, phiInhB
   real, intent(inout), dimension(nf,nx+2*nb,ny+2*nb) :: activity

   ! params
   real :: tau, tauE, tauI, tauIB, Vrest, VthRest, Vexc, Vinh, VinhB, tauVth, deltaVth

   real, parameter :: GMAX = 10.0

   ! local param variables
   real, pointer, dimension(:,:,:) :: l_activ
   !$OFP LOCAL :: V, Vth, G_E, G_I, G_IB, phiExc, phiInh, phiInhB, l_activ

   ! temporary arrays
   real, dimension(nf,nx,ny) :: tauInf, VmemInf

   !
   ! start of LIF2_update_exact_linear
   !

   ! define local param variables
   !
   tau   = params%tau
   tauE  = params%tauE
   tauI  = params%tauI
   tauIB = params%tauIB

   Vrest = params%Vrest
   Vexc  = params%Vexc
   Vinh  = params%Vinh
   VinhB = params%VinhB

   tauVth   = params%tauVth
   VthRest  = params%VthRest
   deltaVth = params%deltaVth

!   call add_noise(l, dt)

   G_E  = phiExc  + G_E *exp(-dt/tauE)
   G_I  = phiInh  + G_I *exp(-dt/tauI )
   G_IB = phiInhB + G_IB*exp(-dt/tauIB)

   tauInf  = (dt/tau) * (1.0 + G_E + G_I + G_IB)
   VmemInf = (Vrest + G_E*Vexc + G_I*Vinh + G_IB*VinhB) / (1.0 + G_E + G_I + G_IB)

   V = VmemInf + (V - VmemInf)*exp(-tauInf)

   !
   ! start of LIF2_update_finish
   !

   phiExc  = 0.0
   phiInh  = 0.0
   phiInhB = 0.0

   Vth = VthRest + (Vth - VthRest)*exp(-dt/tauVth)

   !
   ! start of update_f
   !

   where (G_E  > GMAX) G_E  = GMAX
   where (G_I  > GMAX) G_I  = GMAX
   where (G_IB > GMAX) G_IB = GMAX

   l_activ => local(activity, [0,0,nb,nb,nb,nb])

   where (V > Vth)
      l_activ = 1.0
      V = Vrest             ! reset cells that fired
      Vth = Vth + deltaVth  !
      G_IB = G_IB + 1.0     ! add hyperpolarizing current
   elsewhere
      l_activ = 0.0
   end where

   !
   ! These actions must be done outside of kernel
   !    1. set activity to 0 in boundary (if needed)
   !    2. update active indices
   !

end subroutine update_state


end module LIF
