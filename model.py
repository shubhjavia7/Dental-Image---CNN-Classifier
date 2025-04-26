{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class ConvNet(nn.Module):\n",
    "\n",
    "    class Block(nn.Module):\n",
    "        def __init__(self, in_channels, out_channels, stride):\n",
    "            super().__init__()\n",
    "            kernel_size = 3\n",
    "            padding = (kernel_size - 1) // 2\n",
    "\n",
    "            self.block_model = nn.Sequential(\n",
    "                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),\n",
    "                nn.BatchNorm2d(out_channels),\n",
    "                nn.GELU(),\n",
    "                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),\n",
    "                nn.BatchNorm2d(out_channels),\n",
    "                nn.GELU(),\n",
    "                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False),\n",
    "                nn.BatchNorm2d(out_channels),\n",
    "                nn.GELU(),\n",
    "            )\n",
    "\n",
    "            # Residual connection\n",
    "            if in_channels != out_channels or stride != 1:\n",
    "                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)\n",
    "            else:\n",
    "                self.skip = nn.Identity()\n",
    "\n",
    "        def forward(self, x):\n",
    "            return self.block_model(x) + self.skip(x)\n",
    "\n",
    "    def __init__(self, channels_l0=64, n_blocks=3, num_classes=8):\n",
    "        super().__init__()\n",
    "        cnn_layers = [\n",
    "            nn.Conv2d(in_channels=3, out_channels=channels_l0, kernel_size=11, stride=2, padding=(11 - 1) // 2, bias=False),\n",
    "            nn.BatchNorm2d(channels_l0),\n",
    "            nn.GELU(),\n",
    "        ]\n",
    "\n",
    "        c1 = channels_l0\n",
    "        for _ in range(n_blocks):\n",
    "            c2 = c1 * 2\n",
    "            cnn_layers.append(self.Block(c1, c2, stride=2))\n",
    "            c1 = c2\n",
    "\n",
    "        cnn_layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # Global Average Pooling\n",
    "        self.cnn = nn.Sequential(*cnn_layers)\n",
    "        self.classifier = nn.Linear(c1, num_classes)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.cnn(x)\n",
    "        x = x.view(x.size(0), -1)  # flatten for the classifier\n",
    "        x = self.classifier(x)\n",
    "        return x\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
