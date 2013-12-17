%%Bresenham's algorithm
%% 
%%http://en.wikipedia.org/wiki/Bresenham's_line_algorithm
%%
%% inputs:
%%   if pt1 = [y,x] and pt2 = [y,x], then coords = [pt1;pt2]
%%      

function line_mask = bresenham(matrix,coords)

    [height width] = size(matrix);

    line_mask = matrix;

    x = round(coords(:,2));
    y = round(coords(:,1));

    steep = abs(y(2)-y(1)) > abs(x(2)-x(1));

    swap = @(x,y) [y;x];

    if steep
        out = swap(x,y);
        x = [out(1);out(2)];
        y = [out(3);out(4)];
    endif

    if x(1) > x(2)
        x = swap(x(1),x(2));
        y = swap(y(1),y(2));
    endif

    if y(1) < y(2)
        ystep = 1;
    else
        ystep = -1;
    endif

    delx = x(2)-x(1);
    dely = abs(y(2)-y(1));
    err  = delx/2;
    y_n  = y(1);

    for x_n=x(1):x(2)
        if lt(x_n,1)
            x_n = 1;
        endif
        if lt(y_n,1)
            y_n=1;
        endif
        if steep
            if gt(x_n,height) || gt(y_n,width)
                continue
            endif
            line_mask(x_n,y_n) = 1;
        else
            if gt(y_n,height) || gt(x_n,width)
                continue
            endif
            line_mask(y_n,x_n) = 1;
        endif
        err -= dely;
        if lt(err,0)
            y_n += ystep;
            if lt(y_n,1)
                y_n=1;
            endif
            err += delx;
            if steep
                if gt(x_n,height) || gt(y_n,width)
                    continue
                endif
                line_mask(x_n,y_n) = 1;
            else
                if gt(y_n,height) || gt(x_n,width)
                    continue
                endif
                line_mask(y_n,x_n) = 1;
            endif
        endif
    endfor

    if ~eq(size(line_mask),size(matrix))
        error('bresenham: ERROR: The output matrix is not the same size as the input matrix')
    endif
endfunction
