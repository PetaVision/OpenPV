% Takes in the output of a classifier and the ground truth and scores it.
% Expects ground truth to be a one-hot 1x1xF vector.

function score = calc_score(estPvp, gtPvp)
   est   = readpvpfile(estPvp);
   gt    = readpvpfile(gtPvp);
   score = 0;
   total = size(gt)(1);
   for i=1:total
      [estVal, estInd] = max(est{i}.values);
      [gtVal, gtInd]   = max(gt{i}.values);
      if estInd == gtInd
         score += 1;
      end
   end
   score = (score / total * 100.0);
end
