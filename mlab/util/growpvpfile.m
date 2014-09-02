%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Grow PVP activity file to larger dimensions
%% 
%%    Sheng Lundquist
%%    Aug 6, 2014
%%
%% Input: PVP file, offsetX, offsetY, nX, nY, (progressPeriod)
%%    PVP file - File to be cropped
%%    offsetX  - How much to stride in the X direction
%%    offsetY  - How much to stride in the Y direction
%%    nX       - X size of the grown portion in number of neurons
%%    nY       - Y size of the grown portion in number of neurons
%%
%%    (progressPeriod) - Optional, prints message to screen every progressPeriod frames 
%%
%% Output: Cropped PVP activity file
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function growpvpfile(filename, offsetX, offsetY, nX, nY, progressPeriod)

   if nargin < 5
      error('croppvpfile requires inputs: filename, offsetX, offsetY, nX, nY');
   end%if

   if nargin < 6 || ~exist('progressPeriod','var') || isempty(progressPeriod)
      progressPeriod = -1;
   end%if

   fid = fopen(filename);

   hdr = readpvpheader(fid);

   assert(hdr.nx + offsetX <= nX);
   assert(hdr.ny + offsetY <= nY);

   %Sparse file
   if(hdr.filetype == 6)
      keyboard
   
   else if(hdr.filetype == 4)
      newData = cell(hdr.nbands,1);

      for writeTime = 1:1:hdr.nbands
         if exist('progressPeriod','var')
             if ~mod(writeTime,progressPeriod)
                 fprintf(1,'File %s: frame %d of %d\n',filename, writeTime, hdr.nbands);
                 if exist('fflush')
                    fflush(1);
                 end%if
             end%if
         end%if

         [oldData, hdr] = readpvpfile(filename,0,writeTime,writeTime);
         newData{writeTime}.time = oldData{writeTime}.time;
         oldVal = oldData{writeTime}.values;
         newData{writeTime}.values = zeros(nX, nY, hdr.nf);
         newData{writeTime}.values(offsetX+1:hdr.nx+offsetX, offsetY+1:hdr.ny+offsetY, :) = oldVal;
         %Mirroring for now. TODO: allow for initialization values for this as well

         %Left border
         border = newData{writeTime}.values(offsetX+1:2*offsetX, :, :);
         border = border(end:-1:1, :, :);
         if(~isempty(border))
            newData{writeTime}.values(1:offsetX, :, :) = border;
         end
         %Right border
         rightSize = nX - (hdr.nx + offsetX);
         border = newData{writeTime}.values(hdr.nx+offsetX-rightSize+1:hdr.nx+offsetX, :, :);
         border = border(end:-1:1, :, :);
         if(~isempty(border))
            newData{writeTime}.values(end-rightSize+1:end, :, :) = border;
         end

         %Top border
         border = newData{writeTime}.values(:, offsetY+1:2*offsetY, :);
         border = border(:, end:-1:1, :);
         if(~isempty(border))
            newData{writeTime}.values(:, 1:offsetY, :) = border;
         end

         botSize = nY - (hdr.ny + offsetY);
         %Bottom border
         border = newData{writeTime}.values(:, hdr.ny+offsetY-botSize+1:hdr.ny+offsetY, :);
         border = border(:, end:-1:1, :);
         if(~isempty(border))
            newData{writeTime}.values(:, end-botSize+1:end, :) = border;
         end

      end%for
   end

   filename_split = strsplit(filename,'.');

   newFilename = [filename_split{1},'_grow_',num2str(nX),'x',num2str(nY),'y.',filename_split{2}];

   writepvpactivityfile(newFilename,newData);

%end%function
end
