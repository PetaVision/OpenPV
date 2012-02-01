function annotateimage(inAct, outAct, rotation, rotinc, showframe, dnoff, mt0ff)

xgridsize=30;
ygridsize=40;


[tsize, ysize, xsize]=size(inAct);
picsize=max(ysize,xsize);


figure; hold on;
imagesc(squeeze(inAct(showframe+dnoff,:,:)));
xlim([0 picsize])
ylim([0 picsize])
colormap(gray);
axisHandle=get(gca);
loc=axisHandle.Position;
x0=loc(1);xpiclen=loc(3)-x0;picadjx=loc(3)/picsize;
y0=loc(2);ypiclen=loc(4)-y0;picadjy=loc(4)/picsize;
arrowlen=100*(xgridsize+ygridsize)/4;
%rotation=pi/4; %pi/4;

maxMT=max(max(max(max(outAct))))


for(xx=[0:xsize/xgridsize-1])
    stx=xx*xgridsize+1;
    for(yy=[0:ysize/ygridsize-1])
        sty=yy*ygridsize+1;
        aveMotion1=sum(sum(outAct(showframe+mt0ff,[sty:sty+ygridsize-1],[stx:stx+xgridsize-1],1)))/(xgridsize*ygridsize);
        aveMotion2=sum(sum(outAct(showframe+mt0ff,[sty:sty+ygridsize-1],[stx:stx+xgridsize-1],2)))/(xgridsize*ygridsize);
        aveMotion3=sum(sum(outAct(showframe+mt0ff,[sty:sty+ygridsize-1],[stx:stx+xgridsize-1],3)))/(xgridsize*ygridsize);
        aveMotion4=sum(sum(outAct(showframe+mt0ff,[sty:sty+ygridsize-1],[stx:stx+xgridsize-1],4)))/(xgridsize*ygridsize);
        arrowstx=(stx+xgridsize/2)*picadjx;
        arrowsty=(sty+ygridsize/2)*picadjy;
        
        %if(rot==1)
            radius=(aveMotion1/maxMT)*(arrowlen)*picadjx;
            arrowfinish1x=arrowstx+radius*cos(rotation+3*rotinc); %7*pi/4);
            arrowfinish1y=arrowsty+radius*sin(rotation+3*rotinc); %7*pi/4);
            
            radius=(aveMotion2/maxMT)*(arrowlen)*picadjx;
            arrowfinish2x=arrowstx+radius*cos(rotation+2*rotinc); %5*pi/4);
            arrowfinish2y=arrowsty+radius*sin(rotation+2*rotinc); %5*pi/4);
            
            radius=(aveMotion3/maxMT)*(arrowlen)*picadjx;
            arrowfinish3x=arrowstx+radius*cos(rotation+rotinc); %3*pi/4);
            arrowfinish3y=arrowsty+radius*sin(rotation+rotinc); %3*pi/4);
            
            radius=(aveMotion4/maxMT)*(arrowlen)*picadjx;
            arrowfinish4x=arrowstx+radius*cos(rotation+0); %pi/4);
            arrowfinish4y=arrowsty+radius*sin(rotation+0); %pi/4);;
%         else
%             arrowfinish1x=arrowstx+(aveMotion1/maxMT)*(arrowlen)*picadjx;
%             arrowfinish1y=arrowsty;
%             
%             arrowfinish2x=arrowstx;
%             arrowfinish2y=arrowsty-(aveMotion2/maxMT)*(arrowlen)*picadjy;
%             
%             arrowfinish3x=arrowstx-(aveMotion3/maxMT)*(arrowlen)*picadjx;
%             arrowfinish3y=arrowsty;
%             
%             arrowfinish4x=arrowstx;
%             arrowfinish4y=arrowsty+(aveMotion4/maxMT)*(arrowlen)*picadjy;
%         end

        if((arrowfinish1x~=arrowstx)||(arrowfinish1y~=arrowsty))
            annotation('arrow',[x0+arrowstx, x0+arrowfinish1x], [y0+arrowsty, y0+arrowfinish1y], 'Color', 'r');
        end
        if((arrowfinish2x~=arrowstx)||(arrowfinish2y~=arrowsty))
            annotation('arrow',[x0+arrowstx, x0+arrowfinish2x], [y0+arrowsty, y0+arrowfinish2y], 'Color', 'b');
        end
        if((arrowfinish3x~=arrowstx)||(arrowfinish3y~=arrowsty))
            annotation('arrow',[x0+arrowstx, x0+arrowfinish3x], [y0+arrowsty, y0+arrowfinish3y], 'Color', 'g');
        end
        if((arrowfinish4x~=arrowstx)||(arrowfinish4y~=arrowsty))
            annotation('arrow',[x0+arrowstx, x0+arrowfinish4x], [y0+arrowsty, y0+arrowfinish4y], 'Color', 'c');
        end
        
        arrowSumFinishx=arrowstx+(aveMotion1/maxMT)*(arrowlen)*picadjx-(aveMotion3/maxMT)*(arrowlen)*picadjx;
        arrowSumFinishy=arrowsty+(aveMotion4/maxMT)*(arrowlen)*picadjy-(aveMotion2/maxMT)*(arrowlen)*picadjy;
%         if((arrowSumFinishx~=arrowstx)||(arrowSumFinishy~=arrowsty))
%             annotation('arrow',[x0+arrowstx, x0+arrowSumFinishx], [y0+arrowsty, y0+arrowSumFinishy], 'Color', 'y');
%         end
    end
end