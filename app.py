from flask import Flask, request, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
class ConvNet(nn.Module):

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.block_model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

            # Residual connection
            if in_channels != out_channels or stride != 1:
                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            return self.block_model(x) + self.skip(x)

    def __init__(self, channels_l0=64, n_blocks=2, num_classes=8):
        super().__init__()
        cnn_layers = [
            nn.Conv2d(in_channels=3, out_channels=channels_l0, kernel_size=11, stride=2, padding=(11 - 1) // 2, bias=False),
            nn.BatchNorm2d(channels_l0),
            nn.ReLU(),
        ]

        c1 = channels_l0
        for _ in range(n_blocks):
            c2 = c1 * 2
            cnn_layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Global Average Pooling
        self.cnn = nn.Sequential(*cnn_layers)
        self.classifier = nn.Linear(c1, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # flatten for the classifier
        x = self.classifier(x)
        return x

model = ConvNet(channels_l0=32, n_blocks=2, num_classes=8)
model.load_state_dict(torch.load("/Users/shubhjavia/Desktop/MSAI/SP'25/AI in Health/High Risk Tutorial/teeth_cnn_model.pth",map_location=torch.device('cpu')))
model.eval()

# Define preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Define class names
class_names = ["Lower Front", "Lower Left", "Lower Occlusal", "Lower Right", "Upper Front", "Upper Left", "Upper Occlusal", "Upper Right"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        # Predict the class
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = output.argmax(1).item()

        return f'Predicted Class: {class_names[predicted_class]}'

if __name__ == '__main__':
    app.run(debug=True) 