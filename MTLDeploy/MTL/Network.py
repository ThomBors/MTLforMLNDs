import torch.nn as nn


class PDNetwork(nn.Module):
    def __init__(self):
        super(PDNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class ADNetwork(nn.Module):
    def __init__(self):
        super(ADNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class CNNetwork(nn.Module):
    def __init__(self):
        super(CNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class EMCINetwork(nn.Module):
    def __init__(self):
        super(EMCINetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class FTDNetwork(nn.Module):
    def __init__(self):
        super(FTDNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class LMCINetwork(nn.Module):
    def __init__(self):
        super(LMCINetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class MCINetwork(nn.Module):
    def __init__(self):
        super(MCINetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(300,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),
            
            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,12),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(12,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class MultiTaskModel(nn.Module):
    def __init__(self,task_specific_models,freeze):
        super(MultiTaskModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            # nn.Linear(402,402),
            # nn.BatchNorm1d(num_features=1),
            # nn.ReLU(),
            # nn.Dropout(0.5),


            # nn.Linear(402,402),
            # nn.BatchNorm1d(num_features=1),
            # nn.ReLU(),
            # nn.Dropout(0.5),


            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            )           
        
        self.task_specific_models = nn.ModuleDict(task_specific_models)

        if freeze:
            # Freeze task-specific models
            for model in self.task_specific_models.values():
                for param in model.parameters():
                    param.requires_grad = False


    def forward(self, inp):
        # Forward pass through shared layers
        inp_enc = self.shared_layers(inp)

        # Forward pass through task-specific layers for each task
        output = {}
        for task_name, task_model in self.task_specific_models.items():
            output[task_name] = task_model(inp_enc)

        return output