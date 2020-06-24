[Ypred,probs] = classify(net,imdsTest);
idx = randperm(numel(imdsTest.Files),6);
figure;
for i =1:6
   subplot(2,3,i)
  I = readimage(imdsTest,idx(i));
  imshow(I)
  label = YPred(idx(i));
  title(string(label) +','+ num2str(100*max(probs(idx(i),:)),3) + "%");
end