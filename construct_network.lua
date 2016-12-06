---------------------------------
--Code to construct the network
--------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------------------------


-- Construct the network



function construct_network(alexnet_prototxt_path, alexnet_model_path,ini_scale,mul_factor,modeltype,localization_network_path,atr_name)
	-- Localization Network
	local spanet=nn.Sequential()

	-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
	local tranet=nn.Sequential()
	tranet:add(nn.Identity())
	tranet:add(nn.Transpose({2,3},{3,4}))

	-- initialize with first 5 conv layers of Alexnet
	local locnet =loadcaffe.load(alexnet_prototxt_path,alexnet_model_path,'cudnn')
	for i=24,16,-1 do
		locnet:remove(i)
	end

	-- add new conv layer and fully connected layer
	locnet:add(cudnn.SpatialConvolution(256,128,1,1))
	locnet:add(nn.View(4608))
	locnet:add(cudnn.ReLU(true))
	locnet:add(nn.Linear(4608,128))
	locnet:add(cudnn.ReLU(true))

	-- add final fully connected layer which predicts scale and translations
	-- randomly initialize translation and scale is initialized with ini_scale
	local outLayer = nn.Linear(128,3)
	outLayer.weight[1]:fill(0)
	local bias = torch.FloatTensor(3):fill(0)
	bias[1]=(ini_scale)/mul_factor
	outLayer.bias:copy(bias)
	locnet:add(outLayer)

	-- add multiplication layer to change scale slower compared to translations
	local mulLayer = nn.CMul(3)
	mulLayer.weight:fill(1)
	mulLayer.weight[1]=mul_factor
	locnet:add(mulLayer)


	-- add matrix and grid generator

	local locnetnew=nn.Sequential()
	locnetnew:add(locnet)
	locnetnew:add(nn.AffineTransformMatrixGenerator(false,true,true))
	locnetnew:add(nn.AffineGridGeneratorBHWD(227,227))

	-- add bilinear sampler which takes grid and image as input
	local concat=nn.ConcatTable()
	concat:add(tranet)
	concat:add(locnetnew)
	spanet:add(concat)
	spanet:add(nn.BilinearSamplerBHWD())

	-- and we transpose back to standard BDHW format for subsequent processing by nn modules
	spanet:add(nn.Transpose({3,4},{2,3}))



	-- Ranker Network

	-- initialize with ranknet
	local ranknet =loadcaffe.load(alexnet_prototxt_path,alexnet_model_path,'cudnn')
	ranknet:remove(24)
	ranknet:remove(23)

	-- construct one branch of siamese network
	local siamese_1=nn.Sequential()
	siamese_1:add(spanet)
	siamese_1:add(ranknet)

	-- if it is combined model, then add extra branch for global image
	if(modeltype==2) then
		siamese_1_cat=nn.Concat(2)
		siamese_1_cat:add(siamese_1)
		ranknetp=ranknet:clone('weight','bias','gradWeight','gradBias')
		siamese_1_cat:add(ranknetp)
	end

	-- add final layer which gives the final ranking score
	if(modeltype==2) then
		scoreLayer = nn.Linear(8192,1)
	else
		scoreLayer = nn.Linear(4096,1)
	end
	local method = 'xavier'
	local scoreLayer_new = require('weight-init')(scoreLayer, method)
	scoreLayer_new.bias:fill(0)

	-- ind_list represents layers to be initialized in the localization layer
	ind_list={1, 5, 9, 11, 13, 16, 19, 21}
	if(modeltype==2) then
		siamese_1_combine=nn.Sequential()
		siamese_1_combine:add(siamese_1_cat)
		siamese_1_combine:add(scoreLayer_new)
		-- Initialize Localization Network
		initialize_localization_network(localization_network_path,atr_name,ind_list,siamese_1_combine,mul_factor)
	else
		siamese_1:add(scoreLayer_new)
	end
	


	-- Construct Siamese Network
	local siamese_net = nn.ParallelTable()
	if(modeltype==2) then
		siamese_1_combine=siamese_1_combine:cuda()
		parameters,gradParameters = siamese_1_combine:getParameters()
		siamese_2_combine=siamese_1_combine:clone('weight','bias','gradWeight','gradBias')
		siamese_net:add(siamese_1_combine)
		siamese_net:add(siamese_2_combine)
	else
		siamese_1=siamese_1:cuda()
		parameters,gradParameters = siamese_1:getParameters()
		siamese_2=siamese_1:clone('weight','bias','gradWeight','gradBias')
		siamese_net:add(siamese_1)
		siamese_net:add(siamese_2)
	end
	siamese_net=siamese_net:cuda()
	siamese_net:training()
	return siamese_net

end
