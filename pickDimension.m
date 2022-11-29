function [x] = pickDimension(data, signOn, weights)

minEach = [];

for i = 1:10
    dataTo = data(i,:);
    thresholds = round(min(dataTo)):round(max(dataTo));
    errors = [];

for i = 1:length(thresholds)
    onesGreater = dataTo > thresholds(i);
    createdSigns = [];
    createdSigns(onesGreater) = 1;
    createdSigns(~onesGreater) = -1;
    erroriter = 0;
    
    for j = 1:length(dataTo)
       xi =weights(j)*.5*(1-createdSigns(j)*signOn(j));
       erroriter = erroriter + xi;
    end
    errors = [errors erroriter];
end

[val loc] = min(errors);
minEach = [minEach val];
    
    
    
    
end
[val x] = min(minEach);



end