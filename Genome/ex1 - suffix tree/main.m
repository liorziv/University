B= findRead(a,G(8));
%Q1.a
ST = initialize('mississippi');

%Q1.b
Seqs = load('yeastGenomeSeq.mat');
G = initializeAll(Seqs);


%Q1.3
rangeArr = [10 50 100 200 500, 1000:1000:10000];
t = zeros(1,length(rangeArr));
timeCtr = 1;
for i = rangeArr
    Seq = randseq(i);
    tic;
    initialize(Seq);
    t(timeCtr) = toc;
    timeCtr = timeCtr +1;
end
%O1.3.1
%figure of time Vs size of sequence
figure; plot(rangeArr,t);
ylabel('Suffix Tree initialize running time');
xlabel('Sequence size')

%Q1.3.2
p = polyfit(rangeArr,t,2); 
f1 = polyval(p,rangeArr);

figure;
plot(rangeArr,t,'o')
hold on
plot(rangeArr,t)
plot(rangeArr,f1,'r--')
legend('Actual Time','polyfit')
ylabel('Suffix Tree initialize running time');
xlabel('Sequence size')


%Q1.3.3
tPred = polyval(p,15000);
SeqPred = randseq(15000);
tic;
initialize(SeqPred);
tActual = toc;

%Q2
Seqs2 = load('yeastDNASeqSamples.mat');
readSize = size(Seqs2.reads);
readIdx = cell(1,readSize(2));
count = 1;
%saves the search results for each read
for r = Seqs2.reads
    tmp = findRead(r,G);
    if(isempty(tmp))
        count = count +1;
    else
        readIdx{count} = tmp;
        count = count +1;
    end
    
    
end

%write to output.txt
fileID = fopen('output.txt','w');
for i = 1:readSize(2)
    fwrite(fileID,readIdx{i});
    if(i ~= 100)
         fprintf(fileID,'\r\n');
    end
   
end

fclose(fileID);

