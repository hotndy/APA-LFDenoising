function [dispNet] = CreateNet()

%% network for derain
dispNet.layers = {} ;
dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(11,11,2,100),  zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);

dispNet.layers{end+1} = struct('type', 'relu');
dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(5,5,100,100), zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);

dispNet.layers{end+1} = struct('type', 'relu');
dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(3,3,100,50),  zeros(1,50,'single')}},  'stride', 1, 'pad', 0);

dispNet.layers{end+1} = struct('type', 'relu');
dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(1,1,50,1),    zeros(1,1,'single')}},   'stride', 1, 'pad', 0);

dispNet = InitLayers(dispNet); % creating spaces for corresponding parameters x dzdx etc.
%% network for derain

% %% network for derain
% dispNet.layers = {} ;
% dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(11,11,15,100),  zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% dispNet.layers{end+1} = struct('type', 'relu');
% dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(5,5,100,100), zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% dispNet.layers{end+1} = struct('type', 'relu');
% dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(3,3,100,50),  zeros(1,50,'single')}},  'stride', 1, 'pad', 0);
% 
% dispNet.layers{end+1} = struct('type', 'relu');
% dispNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(1,1,50,1),    zeros(1,1,'single')}},   'stride', 1, 'pad', 0);
% 
% dispNet = InitLayers(dispNet); % creating spaces for corresponding parameters x dzdx etc.
% %% network for derain

% depthNet.layers = {} ;
% depthNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(7,7,200,100), zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% depthNet.layers{end+1} = struct('type', 'relu');
% depthNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(5,5,100,100), zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% depthNet.layers{end+1} = struct('type', 'relu');
% depthNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(3,3,100,50),  zeros(1,50,'single')}},  'stride', 1, 'pad', 0);
% 
% depthNet.layers{end+1} = struct('type', 'relu');
% depthNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(1,1,50,1),    zeros(1,1,'single')}},   'stride', 1, 'pad', 0);

% colorNet.layers = {} ;
% colorNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(7,7,15,100),  zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% colorNet.layers{end+1} = struct('type', 'relu');
% colorNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(5,5,100,100), zeros(1,100,'single')}}, 'stride', 1, 'pad', 0);
% 
% colorNet.layers{end+1} = struct('type', 'relu');
% colorNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(3,3,100,50),  zeros(1,50,'single')}},  'stride', 1, 'pad', 0);
% 
% colorNet.layers{end+1} = struct('type', 'relu');
% colorNet.layers{end+1} = struct('type', 'conv', 'weights', {{InitWeight(1,1,50,3),    zeros(1,3,'single')}},   'stride', 1, 'pad', 0);

% depthNet = InitLayers(depthNet); colorNet = InitLayers(colorNet);