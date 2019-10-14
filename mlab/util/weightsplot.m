% Make basis vector images
%  output_dir: where to place a directory "weights_movie". Plots will be located here.
%  checkpoint_dir: where to find the weight files
%  weight_files: string of weightfile name. 
%       If more than one weight (e.g. stereo) input is a cell of strings.
%       Cell arrangement also specifies sub arangement of kernels (e.g. 4x1, 2x2)
%  ind: vector of indices to plot kernels in this order. "Used by sortweightsplot.m"
%  cols: number of columns (for overriding standard aspect ratio)
%  prefix: file name prefix (e.g. sorted)
function weightsplot(output_dir,checkpoint_dir,weights_files,ind,cols,prefix)

if ~exist('ind','var')
        ind=[];
end
if ~exist('cols','var')
        cols=[];
end
if ~exist('prefix') || isempty(prefix)
    prefix='';
else
    prefix=[prefix,'_'];
end

wfolder=dir(checkpoint_dir);
if isempty(dir(fullfile(output_dir,'weights_movie')))
    mkdir(fullfile(output_dir,'weights_movie'));
end

outdir=fullfile(output_dir,'weights_movie');
if isempty(wfolder)
    error('Error: checkpoint folder does not exist.')
end

if ~iscell(weights_files) && ischar(weights_files)
    weights_files_tmp=weights_files;
    weights_files=cell(1,1);
    weights_files{1}=weights_files_tmp;
end


for checkpoint_i=3:length(wfolder)
    %assert(iscell(weights_files),'Error: weights_files has to be a cell. Note: Order of elements will be reflect in the plot (e.g. for stereo kernels).')
    w=cell(size(weights_files));
    header=cell(size(weights_files));
    for weights_i = 1 : numel(w)
            [w_tmp,header(weights_i)]=readpvpfile(fullfile(checkpoint_dir,wfolder(checkpoint_i).name,weights_files{weights_i}));
            w{weights_i}=squeeze(w_tmp{1,1}.values{1,1});
        end
        niceImageToSave=createNiceProportionsToPlot(w,ind,cols,header{1}.nf);
    imwrite(uint8(niceImageToSave),fullfile(outdir,[prefix,wfolder(checkpoint_i).name,'.png']));
    
    fprintf('%d of %d done\n',checkpoint_i-2,length(wfolder)-2)
end %checkpoint_i


end %weightsplot







function out=createNiceProportionsToPlot(weights,ind,cols,nSubplots)


    % make an approximate 16:9 image of the weights
    if ~exist('cols','var') || isempty(cols)       
        num_cols=1;num_rows=1;
        while num_cols*num_rows < nSubplots
                    if (num_cols+1)*num_rows >= nSubplots
                        num_cols=num_cols+1;
                    elseif num_cols*(num_rows+1) >= nSubplots
                        num_rows=num_rows+1;
                    elseif num_cols/num_rows > (16/size(weights,2))/(9/size(weights,1)) %16:9, but weights might have a sub ratio (e.g. stereo)
            num_rows=num_rows+1;
        else
                        num_cols=num_cols+1;
        end
        end
        if (num_cols-1)*num_rows >= nSubplots
                    num_cols=num_cols-1;
                elseif num_cols*(num_rows-1) >= nSubplots
                    num_rows=num_rows-1;
                end
        else
                num_rows = ceil(nSubplots/num_cols);
        end
            
    % put the kernels to their right places		
    container=cell(num_rows,num_cols);
    subcontainer_tmp=cell(size(weights));
    for i_kernel = 1:nSubplots
            % put subkernels to their place 
            minMaxWeights_tmp=0;
            for i_nPerKernel = 1:numel(weights)
                    if ndims(weights{i_nPerKernel})==4; % 3dm color kernels
                        subcontainer_tmp{i_nPerKernel}=weights{i_nPerKernel}(:,:,:,i_kernel);
                    elseif ndims(weights{i_nPerKernel})==3; % 2dim greylevel kernels
                        subcontainer_tmp{i_nPerKernel}=weights{i_nPerKernel}(:,:,i_kernel);
                    elseif ndims(weights{i_nPerKernel})==2 % reshapes vector to square (so far only pca data)
                    w_tmp=weights{i_nPerKernel}(:,i_kernel);
                        w_reshapeBase_tmp=nan(ceil(sqrt(length(w_tmp))));
                        w_reshapeBase_tmp(1:length(w_tmp))=w_tmp;
                        subcontainer_tmp{i_nPerKernel}=w_reshapeBase_tmp;
                    end
                    
                    minMaxWeights_tmp=max(minMaxWeights_tmp,max(abs(subcontainer_tmp{i_nPerKernel}(:))));
            end %i_nPerKernel
            
            % normalize (for max value)
            for i_nPerKernel = 1:numel(weights)
                subcontainer_tmp{i_nPerKernel}=subcontainer_tmp{i_nPerKernel}/minMaxWeights_tmp;
            end
            
            % add dotted borders (for subkernels, e.g. stereo)
            % add vertical dotted borders 
            dottedVertLine=[zeros(1,1);-ones(2,1);zeros(1,1)];
            dottedVertLine=repmat(dottedVertLine,ceil((1+size(subcontainer_tmp{1},1))/length(dottedVertLine)),1);
            
            if ndims(weights{i_nPerKernel})==4
                dottedVertLine=repmat(dottedVertLine,1,1,3);
            end
            
            for i_horz = 1 : size(subcontainer_tmp,1)
                for i_dottedVertLines = 1 : size(subcontainer_tmp,2)-1
                    subcontainer_tmp{i_horz,i_dottedVertLines}=[subcontainer_tmp{i_horz,i_dottedVertLines},...
                        dottedVertLine(1:size(subcontainer_tmp{i_horz,i_dottedVertLines},1),:,:)];
                end
            end
            
            % add horizontal dotted borders 
            dottedHorzLine=[zeros(1,1),-ones(1,2),zeros(1,1)];
            dottedHorzLine=repmat(dottedHorzLine,1,ceil((1+size(subcontainer_tmp{1},2))/length(dottedHorzLine)));
            if ndims(weights{i_nPerKernel})==4
                dottedHorzLine=repmat(dottedHorzLine,1,1,3);
            end
            for i_vert = 1 : size(subcontainer_tmp,2)
                for i_dottedHorzLines = 1 : size(subcontainer_tmp,1)-1
                    subcontainer_tmp{i_dottedHorzLines,i_vert}=[subcontainer_tmp{i_dottedHorzLines,i_vert};...
                        dottedHorzLine(:,1:size(subcontainer_tmp{i_dottedHorzLines,i_vert},2),:)];
                end
            end
            
            for i_row = 1 : size(subcontainer_tmp,1)
                horz_tmp=[];
                for i_col = 1 : size(subcontainer_tmp,2)
                    horz_tmp=[horz_tmp,subcontainer_tmp{i_row,i_col}];
                end
                container{i_kernel}=[container{i_kernel};horz_tmp];
            end
            
    end %i_kernel
        
    emptycell=find(cellfun(@isempty,container)==1);
    
    for i_emptycell=1:length(emptycell)
        container{emptycell(i_emptycell)}=zeros(size(container{1,1}));
    end
        
    % apply sorting
    if (exist('ind') && ~isempty(ind))
            container(1:max(size(ind)))=container(ind);
        end

        %add borders
        for i_row = 1 : size(container,1)
            for i_col = 1 : size(container,2)
                
                    %top and left
                    container{i_row,i_col}=[...
                    -ones(size(container{i_row,i_col},1),...
                        1,
                        size(container{i_row,i_col},3)),...
                    container{i_row,i_col}];
                    container{i_row,i_col}=[...
                    -ones(1,...
                        size(container{i_row,i_col},2),...
                        size(container{i_row,i_col},3)
                    );container{i_row,i_col}];
                    %bottom and right
                    if i_row == size(container,1)
                        container{i_row,i_col}=[container{i_row,i_col};...
                            -ones(1,...
                                size(container{i_row,i_col},2),...
                                size(container{i_row,i_col},3)
                        )];
                    end
                    if i_col == size(container,2)
                        container{i_row,i_col}=[container{i_row,i_col},...
                            -ones(size(container{i_row,i_col},1),...
                                1,...
                                size(container{i_row,i_col},3)
                            )];
                    end
            end
        end

    concate_weight=cell2mat(container);
    minVal=min(concate_weight(:));
    maxVal=max(concate_weight(:));
    minmaxVal=max(abs(maxVal),abs(minVal));
    out=concate_weight*(1.2*127/minmaxVal)+128;
end %createNiceProportionsToPlot


