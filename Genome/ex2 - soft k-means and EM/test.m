li = 0;
flag = 1;
for i = 1:length(T1)
    for j = 1:length(Z)
        if((T1(i,1) == Z(j,1)& T1(i,2) ~= Z(j,2)) | (T1(i,2) == Z(j,2)&T1(i,1) ~= Z(j,1)))
          
              
                li = li +1;
                display(T1(i,:));
                display(Z(j,:));
                 display(j);
                display(i);
                return;
          end
              
                
                
            
      end
    end

