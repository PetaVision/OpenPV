function writepvpsparseactivityfile(filename, data, nx, ny, nf)
   %  writepvpsparseactivityfile.m
   %
   % Note: the sparse-binary format is no longer supported (as of Mar 14, 2017)
   % Instead, use writepvpsparsevaluesfile to write files in the sparse-values
   % format (file-type = 6).
   
   error('writepvpsparseactivityfile:obsolete',
         'writepvpsparseactivityfile.m is obsolete. Use writepvpsparsevaluesfile.m')
end%function 
