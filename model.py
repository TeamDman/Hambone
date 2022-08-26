import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramModel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(SpectrogramModel, self).__init__()
        self.num_classes = num_classes

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
            hidden_size=3,
            num_layers=3,
            bidirectional=True,
            batch_first=True
        )

        self.dense = nn.Linear(
            in_features=702,
            out_features=num_classes
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, mel_spectro):
        x = mel_spectro
        x = self.batch1(F.relu(self.conv1(x)))
        x = self.batch2(F.relu(self.conv2(x)))
        x = self.batch3(F.relu(self.conv3(x)))
        x = x.transpose(2,1)
        x, _ = self.lstm1(x) # ignore final hidden state
        x, _ = self.lstm2(x)
        x = x.transpose(2,1)
        x = self.batch1(F.relu(self.conv4(x)))
        x = self.batch2(F.relu(self.conv5(x)))
        x = self.batch3(F.relu(self.conv6(x)))
        x = x.transpose(2,1)
        x, _ = self.lstm3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = self.softmax(x)
        return x

    def get_param_count(self):
        return sum([len(x) for x in list(self.parameters())])
    