# LS objective function
function obj_value = sinmodelnz(theta,y,x)
  a = theta(1,:);	
b = theta(2,:);
c = theta(3,:);
d = theta(4,:);
	errors = y - a*sin(b*x+c)-d;   
        errors = errors.*(y~=0);
	obj_value = errors'*errors;
   
endfunction	
