# Emotion Capture from Image 
## Description 
  - Uploading the image formate of jpeg,png and ** formate the image is predicted to class of emotions ()
  - web app detects emotion from facial expression of image
  - For prediction we use to model with different model training
## Tech Stack 
  - Python
  - PyTorch
  - OpenCV
  - streamlit
  - dlib
## Image Transform 
  - GreyScale (output_channel=1)
  - Resize((112,112))
  - int to ToTensor
  - Normalize(mean=[0.5], std=[0.5])
## Model Training
  - model we train is SimpleCNN model
    - conv2d (input_channel=1,output_channel=8,padding=0,kernael=5,stride=1)
    - AvgPool2d(kernel_size=3, stride=3)
    - Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
    - AvgPool2d(kernel_size=2, stride=2)
    - nn.Flatten()
    - Linear(17*17*16 , 128)
    - Linear(128,64)
    - Linear(64,output class)

  - output size of img = ((height - kernel +2*padding)/stride)+1
    - (112x112x1) - (108x108x8)
    - (108x108x8) - (36x36x8)
    - (36x36x8) - (34x34x16)
    - (34x34x16) - (17x17x16)

  - Forward
    - Linear
    - relu introduce non linearity btw the layers
### model 1
  - with learning rate of 0.001
  - epoch 10
  - dataloader batch_size=1000
### model 2
  - with learning rate of 0.0001
  - epoch 10
  - dataloader batch_size = 250
