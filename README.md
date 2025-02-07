# dS-Ease
**dS-Ease** (d-Simplex-EASE) is a novel Class-Incremental Learning (CIL) technique that combines **EASE** (Expandable Subspaces for pretrained model-based class-incremental learning) with the **d-Simplex** compatibility representation. This combination provides a robust solution to the challenges posed by incremental learning scenarios, where a model needs to continually learn new classes while retaining previously acquired knowledge.

Class-Incremental Learning often suffers from catastrophic forgetting, where the model forgets earlier classes as it learns new ones. By integrating EASE and d-Simplex, dS-Ease offers a balanced approach that addresses both knowledge retention and effective adaptation to new data.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [EASE](#ease)
- [d-Simplex](#d-simplex)
- [License](#license)

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
This is an installation example with a Linux OS and Cuda support v11.8.
**Note**: For *Anaconda* environment, please refer to the [specific guide]() reported in the next section
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

### Installation with Conda Environment
If you use a Conda environment, installing dependencies with *conda* can lead to incompatibilities. So, please follow exactly the following procedure:
1. Create a new environment:
   ```bash
   conda create -n <env_name> python=3.10
   ```  
   ```bash
   conda activate <env_name>
   ```

2. Install PyTorch with CUDA support (ex. Linux, Cuda 11.8)
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. Install Pip
   ```bash
   conda install pip
   ```

4. Install other requirements with Pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Navigate into the project directory:
   ```bash
   cd dS-Ease
   ```
2. Navigate into `./exps/` to change hyperparameters and device settings.

3. Start Python script on ImageNet-R:
   ```bash
   python main.py --model psrd_ease_inr
   ```


## EASE
**EASE** ([Expandable Subspaces for pretrained model-based class-incremental learning](https://arxiv.org/abs/2403.12030)) is a technique designed to tackle Class-Incremental Learning by dynamically expanding the model’s capacity to accommodate new classes without forgetting previous ones. EASE works by creating expandable subspaces in the feature space, ensuring that each newly learned class is added in a way that minimizes interference with existing class representations.

This approach is particularly powerful for CIL tasks involving pretrained models, as it leverages existing knowledge while effectively managing new data. EASE allows the model to maintain high accuracy across both old and new classes, making it a state-of-the-art solution in this domain.

## d-Simplex
**d-Simplex** introduces a structured geometric representation to ensure compatibility between class representations as new classes are added ([Regular Polytope Networks](https://arxiv.org/abs/2103.15632), [CoReS: Compatible Representations via Stationarity](https://arxiv.org/abs/2111.07632)). The simplex structure provides a framework in which new classes are geometrically embedded in a way that minimizes interference with previous representations. This helps maintain a smooth integration of new classes, even in the absence of earlier data.

By organizing class representations geometrically, d-Simplex ensures that the overall model maintains coherence as it learns incrementally, preventing the overlap of new and old classes and supporting long-term retention of knowledge.

## PSRD
**PSRD-Ease** comes from the idea of the paper [N. Asadi et al. 2023](https://arxiv.org/abs/2303.14771): The method consists of incrementally training the network with a loss that combines various representation learning components. Th loss is composed by three distinct terms:
- Supervised Constrastive Loss: it is a contrastive loss computed on features obtained from the classes and projected to a less-dimension space.
- Prototypes Loss: it is a classic linear loss based on the similarity between the representation of a class and the prototype associated to the same class.
- Relation Distillation Loss: it is a KL-Divergence that between prototype-sample similarity distribution estimated with the previous model and the current model.
The loss ideally guarantees orthogonality between the task classes representations. The method, originally, is implemented using a simple architecture, similar to a simple ResNet, and it is specified to evaluate task-incremental performance.  
However, with little changes, it can be adapted to our architecture and our class-incremental learning problem to obtain a hybrid method that combines Ease typical subspaces with this revolutionary loss.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
   
