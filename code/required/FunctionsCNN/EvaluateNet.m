function res = EvaluateNet(net, features, res, isForward)

global param;
numLayers = numel(net.layers);

if (isForward)
    res = struct('x', cell(1, numLayers+1), 'dzdx', cell(1, numLayers+1));
    res(1).x = features;
end
    

if isForward
    %% Forward pass
    for i = 1 : numLayers
        l = net.layers{i};
        switch l.type
            case 'conv'
                res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, 'pad', l.pad, 'stride', l.stride, param.gpuMethod);

            case 'relu'
                res(i+1).x = vl_nnrelu(res(i).x,[]) ;

            case 'sigmoid'
                res(i+1).x = vl_nnsigmoid(res(i).x);
                
            case 'softmax'
                res(i+1).x = vl_nnsoftmax(res(i).x);
                
            case 'softmaxloss'
                res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
            otherwise
                error('Unknown layer type ''%s''.', l.type) ;
        end
    end

else
    %% Backward pass    
    for i = numLayers : -1 : 1
        l = net.layers{i};
        switch l.type

            case 'conv'
                [res(i).dzdx, dzdw{1}, dzdw{2}] = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, res(i+1).dzdx, 'pad', l.pad, 'stride', l.stride, param.gpuMethod);

            case 'relu'
                res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx);

            case 'sigmoid'
                res(i).dzdx = vl_nnsigmoid(res(i).x, res(i+1).dzdx);
                
            case 'softmax'
                res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx);
                
            case 'softmaxloss'
                res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        end

        if (strcmp(l.type, 'conv'))
            res(i).dzdw = dzdw;            
            dzdw = [];
        end

    end
end

