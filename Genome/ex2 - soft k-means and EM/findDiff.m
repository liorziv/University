function [eucDis, miny] = findDiff(vec1,vec2)
 diff = (bsxfun(@minus, vec1, vec2));
       tmp = diff.*diff;
       eucDis = sqrt(sum(tmp,2));
       %[minVal,minIdx] = min(tmps);
       %idx = centerIdx(minIdx);
       miny = zeros(length(vec2),1);
       for i = 1: length(vec2)
           miny(i) = norm(vec1 - vec2(i,:));
           if(miny(i) ~= eucDis(i))
               disp(miny(i));
               disp(eucDis(i));
           end
       end
end