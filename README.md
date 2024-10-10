# dS-Ease
d-Simlex-EASE (ds-Ease) is a new Class-Incremental Learning technique that combines the powerful of EASE (SOTA 2024 for CIL) and the Compatibility Representation properties of the d-Simplex.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Dependencies
- `Python 3.10`
- `torch 2.0.1`
- `torchvision 0.15.2`
- `timm 0.6.12`
- `numpy`
- `scipy`
- `tqdm`
- `easydict`

**Note**: A more recent Python version can lead to library incompatibilities.

## Installation
**Note**: this is an installation example with a Linux OS and Cuda support v11.8, feel free to create your custom enviroment according to the dependencies described above.

1. Clone the repository:
   ```bash
   git clone https://github.com/thomasdelmoro21/dS-Ease.git
   ```
2. Navigate into the project directory:
   ```bash
   cd dS-Ease
   ```
3. Install PyTorch with Cuda support (ex. Linux, Cuda 11.8):
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
4. Install other requirements:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage
1. Navigate into the project directory:
   ```bash
   cd dS-Ease
   ```
2. Start Python script:
   ```bash
   python main.py
   ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
   
