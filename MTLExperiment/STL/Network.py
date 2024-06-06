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
    
################## long architectures ###############################

class PDNetworkLong(nn.Module):
    def __init__(self):
        super(PDNetworkLong, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

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


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class ADNetworkLong(nn.Module):
    def __init__(self):
        super(ADNetworkLong, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,200),
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
            nn.Dropout(0.5),

            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class CNNetworkLong(nn.Module):
    def __init__(self):
        super(CNNetworkLong, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

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

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class EMCINetworkLong(nn.Module):
    def __init__(self):
        super(EMCINetworkLong, self).__init__()
        self.net = nn.Sequential(

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),


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
            
            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class FTDNetworkLong(nn.Module):
    def __init__(self):
        super(FTDNetworkLong, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),


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
            #nn.Dropout(0.5),

            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class LMCINetworkLong(nn.Module):
    def __init__(self):
        super(LMCINetworkLong, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),


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

            nn.Linear(50,30),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.3),

            nn.Linear(30,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class MCINetworkLong(nn.Module):
    def __init__(self):
        super(MCINetworkLong, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(403,403),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),


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
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),
            
            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            
            nn.Linear(50,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

################## long50 architectures ###############################
class PDNetworkLong50(nn.Module):
    def __init__(self):
        super(PDNetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class ADNetworkLong50(nn.Module):
    def __init__(self):
        super(ADNetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.3),


            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class CNNetworkLong50(nn.Module):
    def __init__(self):
        super(CNNetworkLong50,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),


            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class EMCINetworkLong50(nn.Module):
    def __init__(self):
        super(EMCINetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),


            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class FTDNetworkLong50(nn.Module):
    def __init__(self):
        super(FTDNetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),


            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class LMCINetworkLong50(nn.Module):
    def __init__(self):
        super(LMCINetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class MCINetworkLong50(nn.Module):
    def __init__(self):
        super(MCINetworkLong50, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out
    
################## long75 architectures ###############################
class PDNetworkLong75(nn.Module):
    def __init__(self):
        super(PDNetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),        

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class ADNetworkLong75(nn.Module):
    def __init__(self):
        super(ADNetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.3),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class CNNetworkLong75(nn.Module):
    def __init__(self):
        super(CNNetworkLong75,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class EMCINetworkLong75(nn.Module):
    def __init__(self):
        super(EMCINetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            

            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class FTDNetworkLong75(nn.Module):
    def __init__(self):
        super(FTDNetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            


            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),


            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class LMCINetworkLong75(nn.Module):
    def __init__(self):
        super(LMCINetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(25,1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        out=self.net(inp)
        return out

class MCINetworkLong75(nn.Module):

    def __init__(self):
        super(MCINetworkLong75, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            


            #-------------------------------
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),

            nn.Linear(50,25),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),

            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out
    
################## "Optimal" architectures ###############################

class PDNetworkOpt(nn.Module):
    def __init__(self):
        super(PDNetworkOpt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,350),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(350,300),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(300,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(150,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),


            #-------------------------------
            nn.Linear(75,50),
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

class ADNetworkOpt(nn.Module):
    def __init__(self):
        super(ADNetworkOpt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,275),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(275,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(150,125),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(125,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),        

            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(75,50),
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
    
class FTDNetworkOpt(nn.Module):
    def __init__(self):
        super(FTDNetworkOpt, self).__init__()
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
            #nn.Dropout(0.5),

            nn.Linear(25,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out  

class MCINetworkOpt(nn.Module):
    def __init__(self):
        super(MCINetworkOpt, self).__init__()
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
            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.1),
            
            nn.Linear(75,50),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            
            nn.Linear(50,1),
            nn.Sigmoid()

        )

    def forward(self, inp):
        out=self.net(inp)
        return out
    
class EMCINetworkOpt(nn.Module):
    def __init__(self):
        super(EMCINetworkOpt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,275),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(275,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(150,125),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(125,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),        

            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(75,50),
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
    
class LMCINetworkOpt(nn.Module):
    def __init__(self):
        super(LMCINetworkOpt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,200),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(200,150),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(150,125),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(125,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),        

            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU6(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(75,50),
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
    
class CNNetworkOpt(nn.Module):
    def __init__(self):
        super(CNNetworkOpt, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(403,325),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(325,250),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(250,175),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(175,100),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(100,75),
            nn.BatchNorm1d(num_features=1),
            nn.ReLU(),
            nn.Dropout(0.5),

            #-------------------------------
            nn.Linear(75,50),
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
  