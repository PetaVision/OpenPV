

input_dir = '/Users/manghel/Documents/workspace/marian/output/';
read_data = 1;
print_data = 0;

% presynaptic neuron location

x=5;
y=10;
patch_size = 9;

filename = ['r',num2str(x),'-',num2str(y),'.probe'];
filename = [input_dir, filename];


if read_data
if exist(filename,'file')
    
    fid = fopen(filename, 'r');
 
    t = 0;
    while t <= 10000
        
        t=t+1;
        %fprintf('%d\n',t);
       
        C = textscan(fid,'%*s',1);

        %read M
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        M(t,:)=C{1}';
        if print_data
        for i=1:patch_size
           fprintf('%f ', M(t,i));
        end
        fprintf('\n');
        end
        
        % read P
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        P(t,:)=C{1}'; 
        if print_data
        for i=1:patch_size
           fprintf('%f ', P(t,i));
        end
        fprintf('\n');
        end
        
        % read W
        C = textscan(fid,'%*s',1);
        C = textscan(fid,'%f',patch_size);
        W(t,:)=C{1}';  
        if print_data
        for i=1:patch_size
           fprintf('%f ', W(t,i));
        end
        fprintf('\n');
        pause
        end
        
        %eofstat = feof(fid);
%       fprintf('eofstat = %d\n', eofstat);
        if (feof(fid))
            fprintf('feof reached: t = %d\n',t);
            break;
        end
         
    
    end
    fclose(fid);
%  
else
    disp(['probe file could not be open ', filename]);
    
end

end

cmap = colormap(hsv(128));

figure('Name','M values')
for i=1:patch_size
   plot(M(:,i),'-','Color',cmap(i*10,:));hold on
   pause
end

figure('Name','P values')
for i=1:patch_size
   plot(P(:,i),'-','Color',cmap(i*10,:));hold on
   pause
end

figure('Name','W values')
for i=1:patch_size
   plot(W(:,i),'-','Color',cmap(i*10,:));hold on
   pause
end


