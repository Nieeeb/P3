import torch
from torch import nn
from Models.src.CLARITY_dataloader import LolDatasetLoader 
from torch.utils.data import DataLoader

dataset = LolDatasetLoader()
trainloader = DataLoader(dataset=dataset)



class ImageToImageCNN(nn.Module):
    def __init__(self):
        super(ImageToImageCNN, self).__init__() #her initialiseres nn.Module
        
        '''
        self.encoder:
            er der første lag der modtager data og lærer features 

            Conv2d:
                modtager 3 kanaler - RGB
                genererer 64 featuremaps 
                conv kigger på 3X3 del af input for hver input channel
                stride er 1 så den ikke springer nogle pixels over
                padding er 1 så output har samme størrelse som input
            ReLu:
                Hvert featuremap er herefter gjort nonlineært med en activation funktion som her er ReLu
            MaxPool2d:
                Max værdien af hver 2X2 del af featuremap er taget med i pooling, hvilket halverer størrelsen af hvert feature map
                Find bedre forklaring https://medium.com/@abhishekjainindore24/pooling-and-their-types-in-cnn-4a4b8a7a4611 - 'Max Pooling'
        '''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
        )


        '''
        self.bottleneck:
            er det sidste 'lærende' lag der rent konceptuelt klargører data til decoding

            Conv2d:
                tager x mængde input kanaler tilsvarende output af self.encoder i det her system: 64
            ReLu:
                Der bruges altid en activation function efter et conv lag for at introducere nonlinearitet
        '''
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()           
        )


        '''
        self.decoder:
            Her tages de lærte featuremaps og skaleres tilbage op til det originale billedes dimensioner

            ConvTranspose:
                bedste/simpleste forklaring jeg kan finde for hvad det gør kan findes: https://indico.cern.ch/event/996880/contributions/4188468/attachments/2193001/3706891/ChiakiYanagisawa_20210219_Conv2d_and_ConvTransposed2d.pdf

                funktionen for størelsen af output ved transpose er: O = (I - 1) * S - 2P + K
                og ved S = 2 og K = 2 og P = 0 vil output størrelsen være:
                O = (I - 1) * 2 - 2 * 0 + 2 
                -> 
                O = 2I - 2 + 2
                ->
                O = 2I
                Derved fordobles størrelsen
            ReLu:
                Er igen til for få non lineært output af den forhænværende convolution
            Conv2d:
                Tager de 64 feature maps og mapper dem tilbage til de original 3 kanaler
            Sigmoid:
                Vi vil gerne have at output er mellem 0 og 1 så vi bruger sigmoid istedet for ReLu
                
        '''
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x): #forward definerer hvordan data går igennem modellen
                          #forward bliver implicit kaldt når modellen modtager data
                          # så i tilfælder output = model(input)
                          # her bliver forward kaldt ^^^^^^   
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x
