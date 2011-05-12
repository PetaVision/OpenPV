function [ preIndexes, features ] = wmax_readFeatures( xScale, yScale, rate_file, pc_file , varargin)
%wmax_readFeatures() reads post conn probes information and features
%   For each post-synaptic neuron we read its pre-synaptic input neurons
% and based on this information extracts the features that should be
% learned using STDP from rate information computed from the firing
% activity in the pre-synaptic layer. This rate information should be 
% computed only when the Image is kep fixed so that the features do not
% change in time. We assume that once the feature set is extracted from
% a fixed image, the feature set remains the same when the image is moved.
% This assumes that we remain in the same statistical space  describing the
% features in the input images. Ergodic hypothesis - spatial average is the
% same to temporal average.

global input_dir  output_dir conn_dir NX NY 

NXpost = NX * xScale; % L1 size
NYpost = NY * yScale;

nxpPost = 5; % post-synaptic neuron patch size (receptive field)
nypPost = 5;

preIndexes = {};
features = {};

debug = varargin{1};

% read activity
% filename = [output_dir 'a1.pvp.rate'];
% fid = fopen(filename);
% aOn = fscanf(fid, '%f', [NX NX]);
% aOn = aOn'  % transpose since it reads in column order
% % and we write in row order
% 
% filename = [output_dir 'a2.pvp.rate'];
% fid = fopen(filename);
% aOff = fscanf(fid, '%f', [NX NX]);
% aOff = aOff'  % transpose since it reads in column order
% % and we write in row order
fprintf('reading activity from %s and syn indexes from %s\n',...
    rate_file,pc_file);
filename = [output_dir rate_file];
fid = fopen(filename);
activity = fscanf(fid, '%f', [NX NX]);
activity = activity';  % transpose since it reads in column order
            % and we write in row order

if debug
    [m,n] = size(activity);
    for i=1:m
        for j=1:n
            fprintf('%d ',activity(j,i));
        end
        fprintf('\n');
    end
    pause
end

% read pre-synaptic neurons afferent to each post-syn neuron
for kyPost=0:(NYpost-1)
    for kxPost=0:(NXpost-1)
        kPost = kyPost * NXpost + kxPost + 1; % shift by 1
        
        filename = [conn_dir pc_file '_' num2str(kxPost) '_' num2str(kyPost) '_0.dat'];
        if debug
            %fprintf('reading %s\n',filename);
            fprintf('\nky = %d kx = %d kPost:\n\n',kyPost+1, kxPost+1, kPost);
        end
        ind =[];
        
        if exist(filename,'file')
            fid = fopen(filename, 'r', 'native');
            s=fgetl(fid);             % read first line - empty
            s=fgetl(fid);             % read second line
            %w=fscanf(fid,'%f',[4,4])  % read weights
            %pause
            % read patch indices
            for jp=1:nypPost
                for ip=1:nxpPost
                    k  = fscanf(fid, '%d',1); % global linear index
                    kx = fscanf(fid, '%d',1);
                    ky = fscanf(fid, '%d',1);
                    kf = fscanf(fid, '%d',1);
                    %fprintf('%d %d %d %d ',k,kx,ky,kf);
                    % remove boundary pre-syn neurons
                    if kx >= 0 & ky >= 0 & kx < NX & ky < NY
                        if debug
                            if k + 1 < 10
                                fprintf(' %d ',k + 1);
                            else
                                fprintf('%d ',k + 1);
                            end
                        end
                        ind=[ind;k];
                    end
                end % ip loop
                if debug,fprintf('\n');end
            end % jp loop
            fclose(fid);
            ind = ind' + 1; % shift by one
            % keep only indexes for neurons that have no margin
            % effects
            if length(ind) == nxpPost * nypPost
                preIndexes{kPost} = ind;
                %preIndexes{kPost}
                if debug
                    for jp=1:nypPost
                        for ip=1:nxpPost
                            fprintf('%d ', preIndexes{kPost}(jp,ip));
                        end
                        fprintf('\n');
                    end
                    pause
                end
                % get feature vector; 1 x n array, row order
                features{kPost} = activity(ind);
                % normalize features
                features{kPost} = features{kPost} ./ norm(features{kPost});
                F = reshape(features{kPost},[nxpPost nxpPost])';
                if debug
                    for jp=1:nypPost
                        for ip=1:nxpPost
                            fprintf('%d ', F(jp,ip));
                        end
                        fprintf('\n');
                    end
                    pause
                end
            else
                preIndexes{kPost} = [];
                % get feature vector
                features{kPost} = [];
            end
            
        else
            disp(['Skipping, could not open ', filename]);
            return
        end
    end % kxPost loop
    if debug
        pause
    end
end % kyPost loop

disp('done!');

end % end of function

