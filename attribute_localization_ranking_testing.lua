---------------------------
-- code to evaluate model
---------------------------
require 'nn'
require 'loadcaffe'
require 'hdf5'
require 'optim'
require 'cutorch'
require 'cudnn'
require 'stn'
require 'image'
require 'utils'

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Set parameters

cmd = torch.CmdLine()
cmd:text('Testing Relative attribute ranker')
cmd:text('Options')
cmd:option('-batch_size',25,'batch size')
cmd:option('-attribute_num',2,'1:baldhead, 2:darkhair, 3:eyesopen, 4:goodlooking, 5:masculinelooking, 6:mouthopen, 7:smile, 8:vforehead, 9:v_teeth, 10:young')
cmd:option('-gpu_no',1,'GPU to be used')
cmd:option('-modeltype',2,'1:only STN, 2:Combined')
params = cmd:parse(arg)
local batchSize=tonumber(params.batch_size)
local atr_num=tonumber(params.attribute_num)
local atr_name_all={'baldhead','darkhair','eyesopen','goodlooking','masculinelooking','mouthopen','smile','vforehead','v_teeth','young'}
local atr_name=atr_name_all[atr_num]
cutorch.setDevice(tonumber(params.gpu_no))
local modeltype = tonumber(params.modeltype)
	

--dataset directory path

local dataset_dir_path= 'train_test_data/faces_test/'

-- model dir path
local model_dir_path=''
if(modeltype==1) then
	model_dir_path = 'learned_model/models_localization/'
else
	model_dir_path = 'learned_model/models_combined/'
end
-----------------------------------------------------------------------------------------------------------------------------------


-- load data
local f_name=dataset_dir_path..'/new_faces_full_'..atr_name..'/file_1.h5'
local mydataval, mydatapval, mylabelval = load_data(f_name)
local total_samples= mylabelval:size()[1]

-- load model
local siamese_network= torch.load(model_dir_path..'/attribute='..atr_name..'/learned_model_400.dat')
siamese_network = siamese_network:cuda()
----------------------------------------------------------------------------------------------------------------------------------

-- evaluate
-- for evaluation average the score of 10 crops
tvalx={0,0,30,30,15}
tvaly={0,30,0,30,15}
siamese_network:evaluate()
final_output=torch.Tensor(2,500,1):fill(0)

for z=1,10 do
        val_corr=torch.Tensor(1):fill(0)
       	val_all=torch.Tensor(1):fill(0)
       	for t = 1,total_samples,batchSize do
        	mini_batchsize= math.min(t+batchSize-1,total_samples) - t +1
        	inputs = torch.Tensor(tonumber(mini_batchsize),3,227,227)
        	inputsp = torch.Tensor(tonumber(mini_batchsize),3,227,227)
        	targets = torch.Tensor(tonumber(mini_batchsize),1)
        	inputs=inputs:cuda()
        	inputsp=inputsp:cuda()
        	targets=targets:cuda()
        	for i = t,math.min(t+batchSize-1,total_samples) do

        		x_tran=tvalx[((z-1)%5)+1]
        		y_tran=tvaly[((z-1)%5)+1]
        		if(z<=5) then
         			inputs[i-t+1] = image.scale(image.crop(mydataval[i],x_tran,y_tran,x_tran+226,y_tran+226),227,227)
          			inputsp[i-t+1] = image.scale(image.crop(mydatapval[i],x_tran,y_tran,x_tran+226,y_tran+226),227,227)
        		else
         			inputs[i-t+1] = image.hflip(image.scale(image.crop(mydataval[i],x_tran,y_tran,x_tran+226,y_tran+226),227,227))
          			inputsp[i-t+1] = image.hflip(image.scale(image.crop(mydatapval[i],x_tran,y_tran,x_tran+226,y_tran+226),227,227))
        		end
          		targets[i-t+1] = (mylabelval[i])
        	end
            	local all_output_val = siamese_network:forward({inputs,inputsp})

            	final_output[{1,{t,t+mini_batchsize-1},{}}]=torch.add(final_output[{1,{t,t+mini_batchsize-1},{}}],all_output_val[1]:double())
            	final_output[{2,{t,t+mini_batchsize-1},{}}]=torch.add(final_output[{2,{t,t+mini_batchsize-1},{}}],all_output_val[2]:double())
        end
end

val_corr=0
val_all=0
for val_cnt=1,total_samples do
	if(mylabelval[val_cnt][1]%2==1) then
        	val_all=val_all+1
                if(final_output[1][val_cnt][1]>final_output[2][val_cnt][1]) then
                	val_corr=val_corr+1
                end
        end
end

accuracy = val_corr/val_all
print('accuracy:'..accuracy)


