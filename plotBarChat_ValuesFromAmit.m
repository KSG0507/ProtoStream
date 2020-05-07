experimentType = 'RetailPerformance';
logYAxis = false;
writeNA = false;
%algos = {'ProtoDash','L2C','RandomW','K-Medoids','P-Lasso','ProtoGreedy'};
algos = {'PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGrdy'};
axesFontSize = 30;
otherFontSize = 30;
axisTight = false;
switch experimentType
    case 'CDCPerformance'
        accValues = [80,60,20,50,0,80];
        titleString = 'a) CDC performance';
        yLabel = '% overlap';
        writeNA = true;
    case 'CDCTime'
        accValues = [20,16,9,213,0,2532];
        titleString = 'b) CDC time';
        yLabel = 'Computation time (secs)';
        logYAxis = true;
        writeNA = true;
    case 'RetailPerformance'
        accValues = [101.23,243.96,201.43, 462.37,396.21,329.99,100.79,418.58];
        titleString = 'a) Retail performance';
        yLabel = 'RMSE';
        algos = {'PrDash','L2C','L2C-A','RndW','K-Med','P-Las','PrGrdy','PrAll'};
        axesFontSize = 30;
        axisTight = true;
    case 'RetailTime'
        accValues = [18003,14479,36719,8997,75600,140400,57600];
        titleString = 'b) Retail time';
        yLabel = 'Computation time (secs)';
end
p = axes;
han = bar(p,accValues,0.4);
x_loc = get(han, 'XData');
y_height = get(han, 'YData');
if(writeNA)
    text(x_loc(5), y_height(5)+1.2,'N/A', 'Color', 'k','HorizontalAlignment','center',...
    'VerticalAlignment','bottom','fontsize',otherFontSize,'fontweight','bold');
end
title(titleString,'fontsize',otherFontSize,'fontweight','bold');
ylabel(yLabel,'fontsize',otherFontSize,'fontweight','bold');
set(gca,'fontsize',axesFontSize,'fontweight','bold');
set(gca,'XTickLabels',algos);
if(logYAxis)
    set(gca,'YScale','log');
end
if(axisTight)
    axis tight;
end