#!/usr/bin/env python

# inter-task affinity
# paper: Fifty, Chris, et al. "Efficiently identifying task groupings for multi-task learning." Advances in Neural Information Processing Systems 34 (2021): 27503-27516.
# original code: https://github.com/google-research/google-research/tree/master/tag


from tqdm import tqdm
import copy
from itertools import combinations,product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from famo import FAMO


# Create a new model for the selected tasks
class SubModel(nn.Module):
    def __init__(self, task_layers_list=None,tsk_name=None, shared_layers=None):
        super(SubModel, self).__init__()
        if shared_layers:
            # Define shared layers
            self.shared_layers = nn.Sequential(*shared_layers)
        else:
            self.shared_layers = None
            # Define parallel task-specific layers
            self.task_layers_list = nn.ModuleList(task_layers_list)

            # Define task name
            self.tsk_name = tsk_name
        
    def forward(self, x):
        if self.shared_layers:
            # Forward pass through shared layers
            output_dict = self.shared_layers(x)
        else:
            
            # Forward pass through task-specific layers in parallel
            task_outputs = [task_layers(x) for task_layers in self.task_layers_list]

            # Create a dictionary with task IDs as keys and outputs as values
            if isinstance(self.tsk_name,list) and len(self.tsk_name) > 1:
                output_dict = {name: output for name, output in zip(self.tsk_name, task_outputs)}
            else:
                output_dict = {self.tsk_name: torch.cat(task_outputs,dim=1)}
            
        return output_dict


def task_affitity(net,trainingset,testset,task_list,loss_fn,learning_rate,weight_decay,step_size,gamma,device):

    model = copy.deepcopy(net)
    inter_task_affinity = {}
    
    task_subset_i = None


    # create all possible combination of tasks
    all_task_subsets = []

    # else for test the metrics on sinle taks set
    all_task_subsets = all_task_subsets = list(product(task_list, repeat=2))
    
    inter_task_affinity = {}
    model.eval()

    # generate subeset of different size
    for tsk_set in tqdm(all_task_subsets):
        

        if task_subset_i != tsk_set[0]:
            # Extract layers for the current task subset
            # Create a list to store task-specific layers
            task_subset_i = tsk_set[0]

            # Extract and store the task-specific layers for the selected tasks
            task_layers_list_i = []
            for task_name,task in model.named_children():
                if task_name in task_subset_i:
                    task_layers_list_i.append(task)

            print(tsk_set,task_layers_list_i)
            # model for tasks
            temp_model_o = SubModel(task_layers_list=task_layers_list_i,tsk_name=task_subset_i) 
            # use deep copy to ensure to not update the original model
            temp_model_i = copy.deepcopy(temp_model_o) 
            print(temp_model_i)

            # shared layers that are going to be update
            temp_model = SubModel(shared_layers=model.shared) 
            # use deep copy to ensure to not update the original model
            temp_model_sh = copy.deepcopy(temp_model) 

                    # set what to update
            temp_model_sh.train()
            temp_model_i.train() 

            # optimizer
            weight_opt = FAMO(n_tasks=1, device=device,w_lr = learning_rate,gamma=0.001)
            
            optimizer_i = optim.Adam(temp_model_i.parameters(),lr=learning_rate,weight_decay=weight_decay)

            optimizer_sh = optim.Adam(temp_model_sh.parameters(),lr=learning_rate,weight_decay=weight_decay)

            scheduler_i = torch.optim.lr_scheduler.StepLR(optimizer_i,step_size=step_size, gamma=gamma)
            scheduler_sh = torch.optim.lr_scheduler.StepLR(optimizer_sh,step_size=step_size, gamma=gamma)
            
            # update the shared layer with respect to the task i
            for batch_idx, (data) in enumerate(trainingset):
                
                for dd in data.keys():
                    data[dd] = data[dd].to(device)

                training_data = data['subject']


                ##### training #####
                optimizer_i.zero_grad()
                optimizer_sh.zero_grad()
                out_sh = temp_model_sh(training_data)
                output_i = temp_model_i(out_sh)

                ##### loss with CN #####
                combined_loss = []
                for task, output in output_i.items():
                    loss_task = F.binary_cross_entropy(torch.squeeze(output_i[task],dim=1),data[task.replace("fc", "")].float())
                    combined_loss.append(loss_task)

                loss_avg = loss_fn(*combined_loss)
                losses = torch.stack(combined_loss)

                ##### backward #####
                loss = weight_opt.backward(losses=losses, 
                                    shared_parameters = temp_model_sh.parameters()
                                    )
                ##### update FAMO weights #####
                with torch.no_grad():
                    out_sh_new = temp_model_sh(training_data)
                    output_i_new = temp_model_i(out_sh)
                    combined_loss = []
                    for task, output in output_i_new.items():
                        loss_task = F.binary_cross_entropy(torch.squeeze(output_i_new[task],dim=1),data[task.replace("fc", "")].float())
                        combined_loss.append(loss_task)

                    new_losses = torch.stack(combined_loss)
                    weight_opt.update(new_losses.detach())
                
                ##### update optimizer #####
                optimizer_i.step()
                optimizer_sh.step()
                        
            if scheduler_i:
                scheduler_i.step()
            if scheduler_sh:
                scheduler_sh.step()


            # print('-- Training set:')
            # print(f"Loss: {loss_avg.item()}")
            # print(f"loss: {loss}")
        
        
        task_subset_j = tsk_set[1]
        
        # Extract and store the task-specific layers for the selected tasks
        task_layers_list_j = []
        for task_name,task in model.named_children():
            if task_name in task_subset_j:
                task_layers_list_j.append(task)

    
        # model for tasks
        temp_model_o = SubModel(task_layers_list=task_layers_list_j,tsk_name=task_subset_j) 
        # use deep copy to ensure to not update the original model
        temp_model_j = copy.deepcopy(temp_model_o) 
        
        temp_model_sh.eval()
        temp_model_j.eval()
        loss_lookahead = 0
        loss = 0

        # compute the necessary loss to compute the inter task affinity
        with torch.no_grad():
            for batch_idx, (data) in enumerate(testset):

                for dd in data.keys():
                    data[dd] = data[dd].to(device)

                validation_data = data['subject']


                # lookahead loss for j
                out_sh_t1 = temp_model_sh(validation_data)
                output_j_t1 = temp_model_j(out_sh_t1)

                ##### loss with CN #####
                combined_loss = []
                for task, output in output_j_t1.items():
                    loss_task = F.binary_cross_entropy(torch.squeeze(output_j_t1[task],dim=1),data[task.replace("fc", "")].float())
                    combined_loss.append(loss_task)

                loss_avg = loss_fn(*combined_loss)
                losses = torch.stack(combined_loss)

                loss_lookahead += loss_avg

                # loss j
                out_sh_t0 = temp_model(validation_data)
                output_j_t0 = temp_model_j(out_sh_t0)
                
                combined_loss = []
                for task, output in output_j_t1.items():
                    loss_task = F.binary_cross_entropy(torch.squeeze(output_j_t0[task],dim=1),data[task.replace("fc", "")].float())
                    combined_loss.append(loss_task)

                loss_avg = loss_fn(*combined_loss)
                losses = torch.stack(combined_loss)

                loss += loss_avg

        #compute and store the inter task affinity
        test_set = "{}_{}".format(task_subset_i, task_subset_j)
        task_gain = (1.0 - (loss_lookahead/batch_idx+1)/(loss/batch_idx+1))
        inter_task_affinity[test_set] = task_gain.cpu().numpy().tolist()
        
        

    return inter_task_affinity



