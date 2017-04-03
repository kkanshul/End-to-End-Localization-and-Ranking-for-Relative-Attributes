
-- all the utility functions

-- function to initialize the localization network
function initialize_localization_network(localization_network_path,atr_name,ind_list,siamese_1_combine,mul_factor)

        local tmp_loc_model=torch.load(localization_network_path..'/attribute='..atr_name..'/learned_model_400.dat'):double()
        for i=1,#ind_list do
                siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight:copy(tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight)
                siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias:copy(tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias)
        end
                local old_mul_factor= tmp_loc_model:get(1):get(1):get(1):get(2):get(1):get(22).weight[1]
                siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).weight[1]:mul(old_mul_factor/mul_factor)
                siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).bias[1] = siamese_1_combine:get(1):get(1):get(1):get(1):get(2):get(1):get(21).bias[1] * (old_mul_factor/mul_factor)

end


-- function to lad the data from hdf5 format
function load_data(f_name)

        print('loading: '..f_name)
        local myFile=hdf5.open(f_name,'r')
        local mydata=myFile:read('data'):all()
        local mydatap=myFile:read('datap'):all()
        local mylabel=myFile:read('label'):all()
        mylabel=mylabel:cuda()

        return mydata, mydatap, mylabel
end


-- function to setup the learning rate of individual layers
function setup_learningrate(siamese_net,ind_list,loc_lr,modeltype)

        local params_lr_m = siamese_net:clone()
        local params_lr = params_lr_m:getParameters()
        params_lr:fill(1)
        if(modeltype==2) then
                for i=1,#ind_list do
                        params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight:fill(loc_lr)
                        params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias:fill(loc_lr)
                end
                params_lr_m:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(22).weight:fill(0)
        else
                for i=1,#ind_list do
                        params_lr_m:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).weight:fill(loc_lr)
                        params_lr_m:get(1):get(1):get(1):get(2):get(1):get(ind_list[i]).bias:fill(loc_lr)
                end
                params_lr_m:get(1):get(1):get(1):get(2):get(1):get(22).weight:fill(0)
        end

        return params_lr

end

-- function to construct directories and sub-directories
function construct_directory(output_dir_path,atr_name,num_samples,modeltype)
	local base_dir='';
        if(modeltype==1) then
		base_dir=output_dir_path..'/models_localization/attribute='..atr_name..'/'
		paths.mkdir(output_dir_path..'/models_localization/')
	else
		base_dir=output_dir_path..'/models_combined/attribute='..atr_name..'/'
		paths.mkdir(output_dir_path..'/models_combined/')
	end
	paths.mkdir(base_dir)
        local im_folder='images/'
        paths.mkdir(base_dir..im_folder)
        for k=1,num_samples do
                paths.mkdir(base_dir..im_folder..k)
        end

        return base_dir, im_folder

end

-- function to create mini batch
function create_batch(mydata,mydatap,mylabel,rnd_order,mini_batchsize,t)

        local inputs = torch.Tensor(tonumber(mini_batchsize),3,227,227)
        local inputsp = torch.Tensor(tonumber(mini_batchsize),3,227,227)
        local targets = torch.Tensor(tonumber(mini_batchsize),1)
        inputs=inputs:cuda()
        inputsp=inputsp:cuda()
        targets=targets:cuda()

        for i = t,(mini_batchsize+t-1) do

                local x_tran= torch.random(1,30)
                local y_tran= torch.random(1,30)
                local x_tranp= torch.random(1,30)
                local y_tranp= torch.random(1,30)
                local flip_1=torch.random(1,2)
                local flip_2=torch.random(1,2)
                if(flip_1%2==0) then
                        inputs[i-t+1] = image.scale(image.crop(mydata[rnd_order[i]],x_tran,y_tran,x_tran+226,y_tran+226),227,227)
                else
                        inputs[i-t+1] = image.hflip(image.scale(image.crop(mydata[rnd_order[i]],x_tran,y_tran,x_tran+226,y_tran+226),227,227))
                end
                if(flip_2%2==0) then
                        inputsp[i-t+1] = image.scale(image.crop(mydatap[rnd_order[i]],x_tranp,y_tranp,x_tranp+226,y_tranp+226),227,227)
                else
                        inputsp[i-t+1] = image.hflip(image.scale(image.crop(mydatap[rnd_order[i]],x_tranp,y_tranp,x_tranp+226,y_tranp+226),227,227))
                end
                targets[i-t+1] = (mylabel[rnd_order[i]])
       end

       return inputs, inputsp, targets

end

-- function to save intermediate results to be used for the visualization
function save_visualization(itr_num,total_cnt,my_cord,siamese_net,rnd_order,mean_value,inp_ind,inp1,inp2,img_gap,epoch_gap,modeltype,base_dir,im_folder)
        if(modeltype==2) then
                my_cord[itr_num][rnd_order[total_cnt]][{{1,3}}]=siamese_net:get(1):get(1):get(1):get(1):get(1):get(2):get(1):get(22).output[inp_ind][{{1,3}}]:double()
                my_cord[itr_num][rnd_order[total_cnt]][{{4,6}}]=siamese_net:get(2):get(1):get(1):get(1):get(1):get(2):get(1):get(22).output[inp_ind][{{1,3}}]:double()
        else
                my_cord[itr_num][rnd_order[total_cnt]][{{1,3}}]=siamese_net:get(1):get(1):get(1):get(2):get(1):get(22).output[inp_ind][{{1,3}}]:double()
                my_cord[itr_num][rnd_order[total_cnt]][{{4,6}}]=siamese_net:get(2):get(1):get(1):get(2):get(1):get(22).output[inp_ind][{{1,3}}]:double()
        end

        if(rnd_order[total_cnt]%img_gap==0 and  (itr_num-1)%epoch_gap ==0) then
                im_name_crop1=base_dir..im_folder..rnd_order[total_cnt]..'/im_1_'..itr_num..'.jpg'
                im_name_crop2=base_dir..im_folder..rnd_order[total_cnt]..'/im_2_'..itr_num..'.jpg'
                im_name_org1=base_dir..im_folder..rnd_order[total_cnt]..'/imorg_1'..'.jpg'
                im_name_org2=base_dir..im_folder..rnd_order[total_cnt]..'/imorg_2'..'.jpg'

                if (modeltype==2) then
                        image.save(im_name_crop1,(image.scale((siamese_net:get(1):get(1):get(1):get(1):get(3).output[inp_ind]+mean_value):double(),64,64):index(1,torch.LongTensor{3,2,1}))/255)
                        image.save(im_name_crop2,(image.scale((siamese_net:get(2):get(1):get(1):get(1):get(3).output[inp_ind]+mean_value):double(),64,64):index(1,torch.LongTensor{3,2,1}))/255)
                else
                        image.save(im_name_crop1,(image.scale((siamese_net:get(1):get(1):get(3).output[inp_ind]+mean_value):double(),64,64):index(1,torch.LongTensor{3,2,1}))/255)
                        image.save(im_name_crop2,(image.scale((siamese_net:get(2):get(1):get(3).output[inp_ind]+mean_value):double(),64,64):index(1,torch.LongTensor{3,2,1}))/255)
                end
                image.save(im_name_org1,((inp1+mean_value):index(1,torch.LongTensor{3,2,1}))/255)
                image.save(im_name_org2,((inp2+mean_value):index(1,torch.LongTensor{3,2,1}))/255)
       end


end


-- function to create webapge
function create_webpage(base_dir,im_folder,itr_num,epoch_gap,img_gap,my_cord,mylabel,num_samples)

        local html_start='<html><head><table><tr><td><img src="'..im_folder..'/plot_loss.png'..'"></img></td><td><img src="'..im_folder..'/plot_accuracy.png'..'"></img></td><</tr></table></head><body><table>'
        local html_end='</table></body></html>'
        local file = torch.DiskFile(base_dir..'/index.html', 'w')
        file:writeString(html_start..'\n')
        for k=1,num_samples do

                if (k%img_gap==0) then
                        file:writeString('<tr>'..'\n')
                        im_name_org1='<td> <h2>'..k..'</h2><img src="'..im_folder..k..'/imorg_1'..'.jpg"  border="4" style="border-color: #ff0000" /img> <h5>label:'..mylabel[k][1]..'</h5></td>'
                        file:writeString(im_name_org1)
                        for l=1,itr_num,epoch_gap  do
                                im_name_crop1='<td><img src="'..im_folder..k..'/im_1_'..l..'.jpg"  border="4" style="border-color: #ff0000" /img>'..'<h5> epoch:'..l..' cord:'..string.format('%0.2f,%0.2f,%0.2f',my_cord[l][k][1],my_cord[l][k][2],my_cord[l][k][3])..'</h5></td>'
                                file:writeString(im_name_crop1..'\n')
                        end
                        file:writeString('</tr><tr>'..'\n')
                        im_name_org2='<td> <h2>'..k..'</h2><img src="'..im_folder..k..'/imorg_2'..'.jpg" border="4" style="border-color: #ff0000" /img> <h5>label:'..mylabel[k][1]..'</h5></td>'
                        file:writeString(im_name_org2..'\n')
                        for l=1,itr_num,epoch_gap  do
                                im_name_crop2='<td><img src="'..im_folder..k..'/im_2_'..l..'.jpg"  border="4" style="border-color: #ff0000" /img>'..'<h5> epoch:'..l..' cord:'..string.format('%0.2f,%0.2f,%0.2f',my_cord[l][k][4],my_cord[l][k][5],my_cord[l][k][6])..'</h5></td>'
                                file:writeString(im_name_crop2..'\n')
                        end
                        file:writeString('</tr>'..'\n')

                end

        end
        file:writeString(html_end)
        file:close()

end

-- function to plot loss and accuracy graph
function plot_graph(itr_num,my_loss,my_accuracy,base_dir,im_folder,plot_interval)

        if((itr_num-1)%plot_interval==0) then
              local epoch_loss = torch.Tensor(itr_num)
              local epoch_accuracy = torch.Tensor(itr_num)
                for k=1,itr_num do
                        epoch_loss[k]=torch.mean(my_loss[k])
                        epoch_accuracy[k]=my_accuracy[k]
                end
                local plot_name_loss=base_dir..im_folder..'/plot_loss.png'
                gnuplot.pngfigure(plot_name_loss)
                gnuplot.plot(epoch_loss)
                gnuplot.xlabel('epochs')
                gnuplot.ylabel('loss')
                gnuplot.plotflush()

                local plot_name_acc=base_dir..im_folder..'/plot_accuracy.png'
                gnuplot.pngfigure(plot_name_acc)
                gnuplot.plot(epoch_accuracy)
                gnuplot.xlabel('epochs')
                gnuplot.ylabel('accuracy')
                gnuplot.plotflush()
        end

end

-- function to save intermediate models
function save_model(itr_num,base_dir,siamese_net,my_loss,my_accuracy,save_interval)

        if((itr_num)%save_interval==0) then
                local loss_loc=base_dir..'/intermediate_params_my_loss.dat'
                local acc_loc=base_dir..'/intermediate_params_my_accuracy.dat'
                local model_loc=base_dir..'/learned_model_'..(itr_num)..'.dat'
                torch.save(loss_loc,my_loss)
                torch.save(acc_loc,my_accuracy)
		-- transfer to cpu and clear the states (to reduce size) before saving model
		siamese_net_copy= siamese_net:clone():double();
		siamese_net_copy:clearState();
                torch.save(model_loc,siamese_net_copy)
        end

end

