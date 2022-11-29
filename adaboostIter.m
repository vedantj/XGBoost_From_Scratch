function [hfunctionWeights alphaValue newWeights]= adaboostIter(data, signOn, weights)

thresholds = round(min(data)):round(max(data));
errors = [];
newWeights = [];
for i = 1:length(thresholds)

    
    onesGreater = data > thresholds(i);
    createdSigns = [];
    createdSigns(onesGreater) = 1;
    createdSigns(~onesGreater) = -1;
    erroriter = 0;
    
    for j = 1:length(data)
       xi =weights(j)*.5*(1-createdSigns(j)*signOn(j));
       erroriter = erroriter + xi;
    end
    errors = [errors erroriter];
end

[val loc] = min(errors);

hfunctionWeights = thresholds(loc);
alphaValue = .5 * log((1-val)/(val));

    for j = 1:length(data) 
     if (data(j) >  hfunctionWeights)
         g = 1;
     else
         g = -1;
     end    
     %xi =weights(j)*.5*(1-createdSigns(j)*signs(j));
     %save new weights
     newWeights(j) = weights(j)*exp(-1*alphaValue*signOn(j)*g);
    end
    
    newWeights = newWeights/sum(newWeights);
    newWeights = {newWeights};

end