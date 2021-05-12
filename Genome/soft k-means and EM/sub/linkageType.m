function [dist] = linkageType(a,b,type)
dist = 0;
if(iscell(a))
     a = a{1};
end
if(iscell(b))
   b= b{1};
end

 aSize = size(a);
 bSize = size(b);

diffArr = zeros(aSize(1)*bSize(1),1);
k = 1;
 for i = 1:aSize(1)
      for j = 1:bSize(1)
        diff = a(i,:) - b(j,:);
        diffArr(k) = sqrt(diff*diff');
        k = k + 1;
        
      end      
 end
 
 %for Average linkage
 if(type == 1)

 sizeOfMut = aSize(1)*bSize(1);
 dist = sum(diffArr)/sizeOfMut;
 end
 
  %for single linkage
 if(type == 2)
     dist = min(diffArr);
 end
end

