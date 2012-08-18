# LS objective function
function obj_value = mises(theta,y,x)
  m = theta(1,:);	
  k = theta(2,:);
[zeroth,ierr]=besseli(0,k);
     


     if ierr ~= 0
        ierr_st = num2str(ierr);
        bessel_st = 'bessel call return';
        print(strcat(ierr_st,bessel_st));
     end %if

     errors = y - (exp(k*cos(x-m)))/(2*pi*zeroth);   
	obj_value = errors'*errors;
   

endfunction	
