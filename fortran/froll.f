      subroutine linfit(x,y,n,slope)

      implicit none
      integer n, i
      real*8  x(n), y(n) 
      real*8  slope, xbar, ybar, s_xx, s_xy
Cf2py intent(in) x
Cf2py intent(in) y
Cf2py intent(in) n
Cf2py intent(out) slope
Cf2py depend(n) x
Cf2py depend(n) y

      xbar = 0.0
      ybar = 0.0

      do i=1,n
         xbar = xbar + x(i)
         ybar = ybar + y(i)
      enddo

      xbar = xbar / n
      ybar = ybar / n

      s_xx = 0.0
      s_xy = 0.0

      do i=1,n
         s_xx = s_xx + (x(i) - xbar)**2
         s_xy = s_xy + (x(i) - xbar)*(y(i) - ybar)
      enddo

      slope = s_xy / s_xx

      end subroutine


      subroutine rolling_linfit(x,y,window,min_per,n,slopes)

      implicit none

      integer n, window, min_per
      real*8  x(n), y(n), slopes(n)

Cf2py intent(in) x
Cf2py intent(in) y
Cf2py intent(in) window
Cf2py intent(in) min_per
Cf2py intent(in) n
Cf2py intent(inout) slopes
Cf2py depend(n) x
Cf2py depend(n) y
Cf2py depend(n) slopes

      integer i, j, hw, wstart, wend, nfinite
      logical w_even

      w_even = mod(window,2) .eq. 0
      if (w_even) then
         hw = window/2
      else
         hw = (window-1)/2
      endif
      

      do i=1,n
         if (w_even) then
c            Window is an even number, the lower bound of the
c            window gets the extra point
             wstart = max(i-hw,1)
             wend = min(i+hw-1,n)
         else
c            Window is a odd number, both sides are symmetric
             wstart = max(i-hw,1)
             wend = min(i+hw,n)
         endif

         nfinite = 0
         do j=wstart,wend
            if (.not. isnan(x(j)) .and. .not. isnan(y(j))) then
                nfinite = nfinite + 1
                if(nfinite .ge. min_per) exit
            endif
         enddo

         if (nfinite .ge. min_per) then
             call linfit(x(wstart:wend), y(wstart:wend), wend-wstart+1, 
     &                   slopes(i))
         endif
      enddo

      end subroutine
