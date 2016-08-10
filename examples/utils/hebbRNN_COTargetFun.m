function [z, targetFeedforward] = hebbRNN_centerOutTargetFunction(t, r, targetFunPassthrough, targetFeedforward)

L = targetFunPassthrough.L;
kinTimes = targetFunPassthrough.kinTimes;
if ismember(t,kinTimes)    
    if isempty(targetFeedforward)
        % initial angles
        ang1 = [pi/10*8 (2*(pi - pi/10*8))];
        initvals(1) = -(L(1)*cos(ang1(1)+pi) + L(2)*cos(ang1(2)+ang1(1)));
        initvals(2) = -(L(1)*sin(ang1(1)) + L(2)*sin(ang1(2)+ang1(1)-pi));

        targetFeedforward.initvals = initvals;
        targetFeedforward.t = [];
        targetFeedforward.posL1 = [];
        targetFeedforward.ang = [];
    else
        ang1 = targetFeedforward.ang(end,:);
        initvals = targetFeedforward.initvals;
    end
    
    ang = zeros(1,length(r));
    for d = 1:length(r)
        ang(d) = ang1(d) + r(d)/150;
    end
    if ang(1) > pi/2*3
        ang(1) = pi/2*3;
    elseif ang(1) < 0
        ang(1) = 0;
    end
    if ang(2) > (pi - pi/20)
        ang(2) = (pi - pi/20);
    elseif ang(2) < pi/20
        ang(2) = pi/20;
    end
    
    pos(1) = initvals(1) + L(1)*cos(ang(1)+pi) + L(2)*cos(ang(2)+ang(1));
    pos(2) = initvals(2) + L(1)*sin(ang(1)) + L(2)*sin(ang(2)+ang(1)-pi);
    
    posL1(1) = initvals(1) + L(1)*cos(ang(1)+pi);
    posL1(2) = initvals(2) + L(1)*sin(ang(1));
    
    z = pos';
    targetFeedforward.ang(end+1,:) = ang;
    targetFeedforward.t(end+1) = t;
    targetFeedforward.posL1(end+1,:) = posL1;
    
else
    z = [0; 0];
end
end