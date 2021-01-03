# Road-Pulse Web Application
Road Pulse is a modern solution to improve traffic management and restore laws and regulations for the public. As the violation of traffic rules is a critical and rising issue presently. So, such a system assists the law enforcement departments to keep track of such events and response in real-time. The system is based upon Web and Android application where violations and anomalies will be handled.

# Custom Trained Model YOLO v3 
“You Only Look Once” is an algorithm that uses convolutional neural networks for object detection. You only look once, or YOLO is one of the faster object detection algorithms out there. Though it is not the most accurate object detection algorithm, it is a very good choice when we need real-time detection, without loss of too much accuracy.
In comparison to recognition algorithms, a detection
## Fully Convolutional Network
YOLO makes use of only convolutional layers, making it a fully convolutional network (FCN). In the YOLO v3 paper, the authors present a new, deeper architecture of a feature extractor called Darknet-53. As its name suggests, it contains 53 convolutional layers, each followed by a batch normalization layer and Leaky ReLU activation. No form of pooling is used, and a convolutional layer with stride 2 is used to downsample the feature maps. This helps in preventing the loss of low-level features often attributed to pooling.
 
 ![](https://miro.medium.com/max/495/1*HHn-fcpOEvQnC6WLEj82Jg.png)
 
The network downsamples the image by a factor called the stride of the network. For example, if the stride of the network is 32, then an input image of size 416 x 416 will yield an output of size 13 x 13. Generally, the stride of any layer in the network is equal to the factor by which the output of the layer is smaller than the input image to the network.

# Custom Training Yolo v3:
In training process, we trained our model on custom classes. Dataset used for training process is AI City Challenge 2019. At first faced many issues like dependencies, lack of knowledge etc. but gradually we successfully trained our model using Google Colab GPU environment. 
 
### Custom training Graph
![](https://github.com/Hamza-773/Road-Pulse/blob/main/data/images/download.png)

Above figure shows a graph generating after training the YOLO v3 on custom classes. The graph shows ‘Loss Per Iteration’. We achieved an average loss value of 0.0399 which is the best value for a training process. The max_batch size was set to 5000 according to the class problems requirement. No of iterations performed were 5k almost and custom weight files generated were saved in our mounted G-drive.
