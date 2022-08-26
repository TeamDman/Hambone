import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramModel(nn.Module):
    def __init__(self) -> None:
        super(SpectrogramModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5)
        self.batch1 = nn.BatchNorm1d(num_features=512)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch2 = nn.BatchNorm1d(num_features=512)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5)
        self.batch3 = nn.BatchNorm1d(num_features=512)
        
        self.lstm1 = nn.LSTM(
            input_size=512,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, padding="same")
        self.batch4 = nn.BatchNorm1d(num_features=512)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding="same")
        self.batch5 = nn.BatchNorm1d(num_features=512)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, padding="same")
        self.batch6 = nn.BatchNorm1d(num_features=512)

        self.lstm3 = nn.LSTM(
            input_size=512,
            hidden_size=1024,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        self.dense = nn.Linear(
            in_features=2048,
            out_features=80
        )

    def forward(self, mel_spectro):
        x = mel_spectro
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = self.batch3(F.relu(self.conv3(x)))
        x = x.transpose(2,1)
        x, _ = self.lstm1(x) # ignore final hidden state
        # downsampling, take every 32nd sample
        reduction_factor = 32 
        x = x[:,::reduction_factor,:]
        # # upsampling, repeat every sample 32 times
        # x = x.repeat(1,1,reduction_factor).reshape(x.size(0),-1,x.size(2))
        x, _ = self.lstm2(x)
        x = x.transpose(2,1)
        x = self.batch1(F.relu(self.conv4(x)))
        x = self.batch2(F.relu(self.conv5(x)))
        x = self.batch3(F.relu(self.conv6(x)))
        x = x.transpose(2,1)
        x, _ = self.lstm3(x)
        x = self.dense(x)
        x = x.transpose(2,1)
        return x

    def get_param_count(self):
        return sum([len(x) for x in list(self.parameters())])
    