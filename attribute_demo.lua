---------------------------
-- demo code
---------------------------
require 'nn'
require 'cutorch'
require 'cudnn'
require 'stn'
require 'image'
------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Set parameters

cmd = torch.CmdLine()
cmd:text('Demo Relative attribute ranker and localizer')
cmd:text('Options')
cmd:option('-attribute_num',6,'1:baldhead, 2:darkhair, 3:eyesopen, 4:goodlooking, 5:masculinelooking, 6:mouthopen, 7:smile, 8:vforehead, 9:v_teeth, 10:young');
cmd:option('-gpu_no',1,'GPU to be used')
params = cmd:parse(arg)
local atr_num=tonumber(params.attribute_num)
local atr_name_all={'baldhead','darkhair','eyesopen','goodlooking','masculinelooking','mouthopen','smile','vforehead','v_teeth','young'};
local atr_name=atr_name_all[atr_num]
cutorch.setDevice(tonumber(params.gpu_no))

model_dir_path_localization = 'learned_model/models_localization/' -- used for localization
model_dir_path_combined = 'learned_model/models_combined/' -- used for getting ranking score

im_inp_path={}
im_out_path={}
num_images=2
im_inp_path[1]='demo_data/input_images/inp_1.jpg'
im_inp_path[2]='demo_data/input_images/inp_2.jpg'
im_out_path[1]='demo_data/output_images/out_1.jpg'
im_out_path[2]='demo_data/output_images/out_2.jpg'

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- load models
	
local siamese_network_localization= torch.load(model_dir_path_localization..'/attribute='..atr_name..'/learned_model_400.dat')
siamese_network_localization= siamese_network_localization:get(1):get(1) -- take one branch of siamese
local siamese_network_combined= torch.load(model_dir_path_combined..'/attribute='..atr_name..'/learned_model_400.dat')
siamese_network_combined = siamese_network_combined:get(1) -- take one branch of siamese
siamese_network_localization= siamese_network_localization:cuda()
siamese_network_combined = siamese_network_combined:cuda()
-- load images
	
local img_batch = torch.Tensor(num_images,3,227,227)
for i=1,num_images do
	img_batch[i] = image.scale(image.load(im_inp_path[i],3,'byte'),227,227)	
end
-- image preprocessing	

local mean_value = torch.Tensor(3,227,227):fill(0)
mean_value[1]=104; mean_value[2]=117; mean_value[3]=122;
for i=1,num_images do
	img_batch[i]=img_batch[i]:index(1,torch.LongTensor{3,2,1})
	img_batch[i]=img_batch[i]-mean_value			
end
img_batch = img_batch:cuda()

-- rank and localize

localized_output=siamese_network_localization:forward(img_batch)
rank_score=siamese_network_combined:forward(img_batch)
	
	
for i=1,num_images do
	print('Score Image-'..i..': '..rank_score[i][1])
end
	
	
for i=1,num_images do
	tmp_output= localized_output[i]:double()+mean_value
	image.save(im_out_path[i],image.scale(tmp_output:index(1,torch.LongTensor{3,2,1})/255,100,100))	
end
