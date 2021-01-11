from argparse import Namespace
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ..chemprop.models.mpn import MPN
from ..chemprop.nn_utils import get_activation_function, initialize_weights

class EvaluationDropout(nn.Dropout):
    def forward(self, input):
        return nn.functional.dropout(input, p = self.p)


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network
    followed by a feed-forward neural network.

    Attributes
    ----------
    uncertainty_method : Optional[str]
        the uncertainty method this model will use
    uncertainty : bool
        whether this model predicts its own uncertainty values
        (e.g. Mean-Variance estimation)
    classification : bool
        whether this model is a classification model
    device : Optional[Union[torch.device, str, Tuple]]
        the device on which to run all calculations
    output_size : int
        the size of the output layer for the feed-forward network
    encoder : MPN
        the message-passing encoder of the message-passing network
    ffn : nn.Sequential
        the feed-forward network of the message-passing network

    Properties
    ----------
    device : torch.device
        the device on which calculations will be run

    Parameters
    ----------
    uncertainty_method: Optional[str] (Default = None)
        the uncertainty method to use. None if the network will not be used to
        predict uncertainty.
    dataset_type : str (Default = 'regression')
        the type of data this model will be trained on
    num_tasks : int (Default = 1)
        the number of tasks to train on
    atom_messages : bool (Default = False)
        whether messages will be passed on atoms instead of bonds during the
        message passing step
    bias : bool (Default = False)
        whether to learn an additive bias in the message passing layers
    depth : int (Default = 3)
        the number of message passing iterations
    dropout : float (Default = 0.0)
        the dropout probability during model training
    undirected : bool (Default = False)
        whether messages will be passed on undirected rather than directed bonds
    aggregation : str (Default = 'mean')
        the method used to aggregate individual atomic feature vectors into a
        molecule-level feature vector
    aggregation_norm : int (Default = 100)
        the normalization factor to use if using the 'norm' aggregation method.
    activation : str (Default = 'ReLU')
        the layer activation function
    hidden_size : int (Default = 300)
        the size of hidden layers in the message passing portion of the network
    ffn_hidden_size : Optional[int] (Default = None)
        the size of the hidden layers in the feed-forward portion of the 
        network. If None, use the same hidden size as the message passing 
        portion.
    ffn_num_layers : int (Default = 2)
        the number of hidden layers in the feed-foward portion of the network
    device: Optional[Union[torch.device, str, Tuple]] (Default= None)
        specifying none will auto-detect whether cuda is available and use
        it possible
    """
    def __init__(self,
                 uncertainty_method: Optional[str] = None,
                 dataset_type: str = 'regression', num_tasks: int = 1,
                 atom_messages: bool = False, bias: bool = False,
                 depth: int = 3, dropout: float = 0.0, undirected: bool = False,
                 aggregation: str = 'mean', aggregation_norm: int = 100,
                 activation: str = 'ReLU', hidden_size: int = 300, 
                 ffn_hidden_size: Optional[int] = None,
                 ffn_num_layers: int = 2,
                 device: Optional[Union[torch.device, str, Tuple]] = None):
        super().__init__()

        self.uncertainty_method = uncertainty_method
        self.uncertainty = uncertainty_method in {'mve'}
        self.classification = dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        self.output_size = num_tasks
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = self.build_encoder(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            aggregation=aggregation, aggregation_norm=aggregation_norm,
            activation=activation, device=device
        )
        self.ffn = self.build_ffn(
            output_size=num_tasks, hidden_size=hidden_size, dropout=dropout, 
            activation=activation, ffn_num_layers=ffn_num_layers, 
            ffn_hidden_size=ffn_hidden_size
        )
        self.device = device

        initialize_weights(self)

    @property
    def device(self) -> torch.device:
        return self.__device
    
    @device.setter
    def device(self, device: Optional[Union[torch.device, str, Tuple]]):
        device = torch.device(device)
        self.__device = device
        self.encoder.device = device
        self = self.to(device)

    def build_encoder(self, atom_messages: bool = False, bias: bool = False,
                      hidden_size: int = 300, depth: int = 3,
                      dropout: float = 0.0, undirected: bool = False,
                      aggregation: str = 'mean', aggregation_norm: int = 100,
                      activation: str = 'ReLU',
                      device: Optional[Union[torch.device, str, Tuple]] = None
                      ) -> None:
         return MPN(Namespace(
            atom_messages=atom_messages, hidden_size=hidden_size,
            bias=bias, depth=depth, dropout=dropout, undirected=undirected,
            features_only=False, use_input_features=False,
            aggregation=aggregation, aggregation_norm=aggregation_norm,
            activation=activation, number_of_molecules=1,
            atom_descriptors=None, mpn_shared=False, device=device
        ))

    def build_ffn(self, output_size: int, 
                  hidden_size: int = 300, dropout: float = 0.0,
                  activation: str = 'ReLU', ffn_num_layers: int = 2,
                  ffn_hidden_size: Optional[int] = None) -> None:
        first_linear_dim = hidden_size

        # If dropout uncertainty method, use for both evaluation and training
        if self.uncertainty_method == 'dropout':
            dropout = EvaluationDropout(dropout)
        else:
            dropout = nn.Dropout(dropout)

        activation = get_activation_function(activation)

        if self.uncertainty:
            output_size *= 2

        if ffn_num_layers == 1:
            ffn = [dropout,
                   nn.Linear(first_linear_dim, output_size)]
        else:
            if ffn_hidden_size is None:
                ffn_hidden_size = hidden_size

            ffn = [dropout,
                   nn.Linear(first_linear_dim, ffn_hidden_size)]
            for _ in range(ffn_num_layers - 2):
                ffn.extend([activation,
                            dropout,
                            nn.Linear(ffn_hidden_size, ffn_hidden_size)])
            ffn.extend([activation,
                        dropout,
                        nn.Linear(ffn_hidden_size, output_size)])

        return nn.Sequential(*ffn)

    def featurize(self, *inputs):
        """Compute feature vectors of the input."""
        return self.ffn[:-1](self.encoder(*inputs))

    def forward(self, *inputs):
        """Runs the MoleculeModel on the input."""
        output = self.ffn(self.encoder(*inputs))

        if self.uncertainty:
            mean_idxs = torch.tensor(range(0, output.shape[1], 2))
            mean_idxs = mean_idxs.to(self.device)
            pred_means = torch.index_select(output, 1, mean_idxs)

            var_idxs = torch.tensor(range(1, output.shape[1], 2))
            var_idxs = var_idxs.to(self.device)
            pred_vars = torch.index_select(output, 1, var_idxs)
            capped_vars = nn.functional.softplus(pred_vars)

            output = torch.stack(
                (pred_means, capped_vars), dim=2
            ).view(output.size())

        # Don't apply sigmoid during training b/c using BCEWithLogitsLoss
        if self.classification and not self.training:
            output = self.sigmoid(output)

        # if self.multiclass:
        #     # batch size x num targets x num classes per target
        #     output = output.reshape((output.size(0), -1, self.num_classes))
        #     if not self.training:
        #         # to get probabilities during evaluation, but not during
        #         # training as we're using CrossEntropyLoss
        #         output = self.multiclass_softmax(output)

        return output
