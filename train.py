import argparse

parser = argparse.ArgumentParser('Arguments for the test.py file')

parser.add_argument('-arch', type=str, default='vgg16', help="Input 'vgg11', 'vgg13', 'vgg16' or 'vgg19' - Model architecture of choice. Enclose in ''. DEFAULT IS 'vgg16'")
parser.add_argument('-hul1', type=int, default=5000, help='Input an integer - Number of units in the first hidden layer. Suggested 20000>x>500. DEFAULT IS 5000')
parser.add_argument('-hul2', type=int, default=1000, help='Input an integer - Number of units in the second hidden layer. Suggested 5000>x>200 or smaller than layer 1. DEFAULT IS 1000')
parser.add_argument('-lr', type=float, default=0.001, help='Input a number - Magnitude of weight change between backpasses. Suggested 0.05>x>0.0005. DEFAULT IS 0.001')
parser.add_argument('-gpu', action='store_true', help='Attempts to use GPU to train if readily available. DEFAULT IS False')
parser.add_argument('-epochs', type=int, default=8, help='Input an integer - How many reruns the training completes before concluding. DEFAULT IS 8')
parser.add_argument('-sd', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth', help="Input a directory - establishes a directory and name to save as the trained checkpoint. Suggested either the working directory for permanancy '/home/workspace/ImageClassifier/checkpoint.pth' or '/opt/checkpoint.pth' for temporary holding. Enclose directory within ''. DEFAULT IS '/home/workspace/ImageClassifier/checkpoint.pth'")

args = parser.parse_args()

model_architecture = args.arch
hidden_layer_1_units = args.hul1
hidden_layer_2_units = args.hul2
learning_rate = args.lr
gpu = args.gpu
n_epochs = args.epochs
save_directory = args.sd


if __name__ == "__main__":
    from train_function import trainandsave
    trainandsave(model_architecture, hidden_layer_1_units, hidden_layer_2_units, learning_rate, gpu, n_epochs, save_directory)
    
# Quick Terminal Commands:
# cd /home/workspace/ImageClassifier
# python train.py
# python train.py -gpu -arch 'vgg13' -hul1 4000 -hul2 800 -lr 0.002 -epochs 1 -sd '/opt/checkpoint.pth' 