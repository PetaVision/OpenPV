% performs a svd of KL variance matrix

case_str = '78';

dir = ['/n/u3/anghel/YEHUDA/case' case_str '/'];

field = 'ux';

evecs_file = [dir,'proc_data/av_surf_',field,'_evecs_0.500000_',...
     case_str,'.dat'];
cum_file   = [dir,'proc_data/av_surf_',field,'_cum_evals_0.500000_',...
     case_str, '.dat'];
evals_file = [dir,'proc_data/av_surf_',field,'_evals_0.500000_',...
     case_str, '.dat'];

read_data   = 1;
comp_evals  = 1;
write_spec = 1;
write_evecs = 1;

N = 2048;           % var matrix has dimension NxN
NEVECS = 20;        %  # of evecs to be printed
A = zeros(N,N);     % variance matrix

if(read_data)
  
  fprintf('read variance matrix:\n')
     fname = [dir,'data/KL_av_surf_',field,'_var_0.500000_', ...
        case_str, '.dat'];
  fid = fopen(fname);
  A = fscanf(fid,'%f',[N N]);
  fclose(fid);
  
  [m,n] = size(A)

  fprintf('check symmetry:\n')	
  for i=1:10
    for j=1:10
      fprintf('%f %f\n',A(i,j),A(j,i));
    end
  end
  %fprintf('type a char to continue:\n')
  %pause
  
end

 
% comp evals
 

if(comp_evals)
  fprintf('compute evals\n')
  
  [V,D] = eig(A);      % note: KLvar = V D V^-1 
  
  % check decomposition
  
  if(0)
    fprintf('check A = VDV^-1\n')
    B = V*D*V^-1;
    for i=1:10
      for j=1:10
	fprintf('%f %f\n',A(i,j),B(i,j));
      end
    end
    fprintf('type a char to continue:\n')
    pause 
    clear B
  end
  
  if(0)
    fprintf('check AV = VD\n')
    B1 = A*V;
    B2 = V*D;
    for i=N:-1:N-9
      for j=N:-1:N-9
	fprintf('%f %f\n',B1(i,j),B2(i,j));
      end
    end
    fprintf('type a char to continue:\n')
    pause
    clear B1
    clear B2
  end
  
  % check dimensions 
  
  if(0)
    fprintf('\nV matrix dimension \n');
    [m,n] = size(V)
    fprintf('type a char to continue:\n')
    pause

    fprintf('\nD matrix dimension \n');
    [m,n] = size(D)
    fprintf('type a char to continue:\n')
    pause
  end

  % evecs stored in the columns of v
  % evals are on the diagonal of s


  evals = zeros(1,N);

  for i=1:N
    evals(N+1-i) = D(i,i);
    %fprintf('e(%d)= %f\n',i,evals(N+1-i))
  end
  
  
  plot(evals(1:N),'-o')
  axis([0 100 0 Inf])
  xlabel('i')
  ylabel('eval(i)')
  fprintf('type a char to continue:\n')
  pause


  % cummulative spectrum

  cum_evals = zeros(1,N);

  for i=1:N
    cum_evals(i) = sum(evals(1:i))/sum(evals) ;
  end

  plot(cum_evals(1:N),'-o')
  axis([0 100 0 1.1])
  xlabel('i')
  ylabel('cum_eval(i)')
  fprintf('type a char to continue:\n')
  pause

  % write evals and cum_evals

  if (write_spec)
    dlmwrite(cum_file,cum_evals',' ');
    dlmwrite(evals_file,evals',' ');
  end  

end % comp_evals


% plot evecs

nx = 64;
ny = 32;
evec=zeros(nx,ny);

if(1)
  for m=N:-1:N-NEVECS+1
    norm(V(:,m),2)
    k=1;
    for j=1:ny
      for i=1:nx
	evec(i,j)=V(k,m);
	k = k+1;
      end
    end
    surf(evec)
    fprintf('evec %d  (strike any key) \n',m);
    pause
  end
end  % end ploting evecs


% write evecs

if(write_evecs)
  
  fprintf('print evecs!\n')
  
  fid = fopen(evecs_file,'w');


  for j = N:-1:N-NEVECS+1 
    vec = V(:,j);
    [m,n] = size(vec);
    fprintf('vec %d m= %d n= %d\n',j,m,n)
    fprintf(fid,'%12.8f\n',vec);
  end

  fclose(fid);
end
