---------------------------------
--Code to train the network
--------------------------------

require 'nn'
require 'loadcaffe'
require 'hdf5'
require 'optim'
require 'cutorch'
require 'cudnn'
require 'stn'
require 'image'
require 'gnuplot'
require 'utils'
require 'construct_network'

---------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Set up the parmaeters

cmd = torch.CmdLine()
cmd:text('Training Relative attribute ranker and localizer')
cmd:text('Options')
cmd:option('-learning_rate',0.001,'Global learning rate of teh network')
cmd:option('-localization_learning_rate',0.1,'Learning rate of localization layer relative to global learning rate')
cmd:option('-batch_size',25,'batch size')
cmd:option('-attribute_num',7,'1:baldhead, 2:darkhair, 3:eyesopen, 4:goodlooking, 5:masculinelooking, 6:mouthopen, 7:smile, 8:vforehead, 9:v_teeth, 10:young')
cmd:option('-scale_ratio',0.05,'Realtive change rate of scale compared to translations')
cmd:option('-gpu_no',1,'GPU to be used')
cmd:option('-scale',0.33,'Initial Scale')
cmd:option('-modeltype',2,'1:only STN, 2:Combined')

params = cmd:parse(arg)

local lr=tonumber(params.learning_rate)
local loc_lr=tonumber(params.localization_learning_rate)
local batchSize=tonumber(params.batch_size)
local atr_num=tonumber(params.attribute_num)
local mul_factor=tonumber(params.scale_ratio)
local ini_scale = tonumber(params.scale)
local atr_name_all={'baldhead','darkhair','eyesopen','goodlooking','masculinelooking','mouthopen','smile','vforehead','v_teeth','young'}
local atr_name=atr_name_all[atr_num]
local modeltype = tonumber(params.modeltype)
local num_itr=400
cutorch.setDevice(tonumber(params.gpu_no))

-- visualization parameters
local img_gap=40 -- gap between training images while displaying on the webpage
local epoch_gap=5 -- number of epochs after which images will be displayed on the webpage
local save_interval = 100 -- number of epochs after which models will be saved
local plot_interval = 5 -- number of epochs after which plot will be updated

-- Pretrained Alexnet location

local alexnet_model_path=''
local alexnet_prototxt_path=''

-- Dataset directory path

local dataset_dir_path= 'train_test_data/faces_train/'

-- Output directory path

local output_dir_path=''

-- Localization network path

local localization_network_path='learned_model/models_localization/'

--------------------------------------------------------------------------------------------------------------------------------------------------------------------


-- Construct the network
local siamese_net = construct_network(alexnet_prototxt_path, alexnet_model_path,ini_scale,mul_factor,modeltype,localization_network_path,atr_name)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Set up the learning rate
local params_lr=setup_learningrate(siamese_net,ind_list,loc_lr,modeltype)


-- load data
local f_name=dataset_dir_path..'/new_faces_full_'..atr_name..'/file_1.h5'
local mydata, mydatap, mylabel = load_data(f_name)
local num_samples = mylabel:size()[1]

-- Construct directory to store output
local base_dir, im_folder = construct_directory(output_dir_path,atr_name,num_samples,modeltype)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




local total_cnt=1
local my_loss = torch.Tensor(num_itr,mylabel:size()[1]):fill(0)
local my_accuracy = torch.Tensor(num_itr,1):fill(0)
local my_cord = torch.Tensor(num_itr,mylabel:size()[1],6):fill(0)   
local config = {learningRate = lr, learningRates = params_lr, weightDecay = 0.0005,
                momentum = 0.9,
                learningRateDecay = 0}

local mean_value = torch.Tensor(3,227,227):fill(0)   
mean_value[1]=104; mean_value[2]=117; mean_value[3]=122; mean_value=mean_value:cuda()

-- training starts

for g =1,num_itr do
	

        local  cnt_correct = 0
        local  cnt_all = 0
        local  rnd_order=torch.randperm(mylabel:size()[1])

	for t = 1,num_samples,batchSize do


	 -- create mini batch
        	local mini_batchsize= math.min(t+batchSize-1,num_samples) - t +1
		local inputs, inputsp, targets = create_batch(mydata,mydatap,mylabel,rnd_order,mini_batchsize,t)
            			

      		-- create closure to evaluate f(X) and df/dX
      		local feval = function(x)
         		-- get new parameters
         		if x ~= parameters then
            			parameters:copy(x)
         		end

         		-- reset gradients
         		gradParameters:zero()

        		 -- f is the loss
         		local f = 0
        		new_labels=torch.ceil(targets:clone()/2)
			new_labels:fill(1)
        		 -- evaluate function for complete mini batch
            		local all_output = siamese_net:forward({inputs,inputsp})
	    		local gd1_all=torch.Tensor(inputs:size()[1],1):zero()
	    		local gd2_all=torch.Tensor(inputs:size()[1],1):zero()	
         		for i = 1,inputs:size()[1] do
            			local my_output = torch.Tensor(2)
	    			my_output=my_output:cuda()
            			my_output[1] = all_output[1][i][new_labels[i][1]]
            			my_output[2] = all_output[2][i][new_labels[i][1]]
            			total_cnt=t+i-1
           			
				-- save intermedaite results for the visualization 
				save_visualization(g,total_cnt,my_cord,siamese_net,rnd_order,mean_value,i,inputs[i],inputsp[i],img_gap,epoch_gap,modeltype,base_dir,im_folder)
	   	
	    			local my_prob = 1/(1+torch.exp(-1*(my_output[1]-my_output[2])))

	    			local err=0	
	    			local gd1=torch.Tensor(1):zero()	
	    			local gd2=torch.Tensor(1):zero()	
            			if(targets[i][1]%2==0) then
					err= (-0.5* torch.log(my_prob)) + (-0.5*torch.log(1-my_prob)) 
                			gd1[1]=my_prob-0.5
                			gd2[1]=0.5-my_prob
		
	    			elseif(targets[i][1]%2==1) then
					err = (-1*torch.log(my_prob))
                			gd1[1]=my_prob-1
                			gd2[1]=1-my_prob
            			end
	    			my_loss[g][total_cnt]=err

	    			if(targets[i][1]%2==1) then
					cnt_all=cnt_all+1
					if(my_output[1]>my_output[2]) then
						cnt_correct=cnt_correct+1
					end	
	    			end	
            			f = f + err
	    			gd1_all[i][new_labels[i][1]]=gd1[1]	
	    			gd2_all[i][new_labels[i][1]]=gd2[1]	
            
         		end
            		gd1_all=gd1_all:cuda()
            		gd2_all=gd2_all:cuda()
            		siamese_net:backward({inputs,inputsp}, {gd1_all,gd2_all})
         		-- normalize gradients and f(X)
	 		if(inputs:size()[1] >0) then
         			gradParameters:div(inputs:size()[1])
         			f = f/inputs:size()[1]
			end

         		-- return f and df/dX
         		return f,gradParameters
      		end

         	optim.sgd(feval, parameters, config)
	end

	my_accuracy[g]=cnt_correct/cnt_all
        print('epoch:'..g..' accuracy:'..cnt_correct/cnt_all..' loss: '..torch.mean(my_loss[g]))

------------------------------------------------------------------------------------------------------------------------------------------------------

	-- visualize results
	create_webpage(base_dir,im_folder,g,epoch_gap,img_gap,my_cord,mylabel,num_samples)

	-- Plot graphs
	plot_graph(g,my_loss,my_accuracy,base_dir,im_folder,plot_interval)

	-- save model
	save_model(g,base_dir,siamese_net,my_loss,my_accuracy,save_interval)

end
