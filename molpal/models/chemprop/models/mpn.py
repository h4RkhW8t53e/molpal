from argparse import Namespace
from typing import List, Optional, Union
from functools import reduce

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from ..features import BatchMolGraph, get_atom_fdim, get_bond_fdim, mol2graph
from ..nn_utils import index_select_ND, get_activation_function

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding 
    a molecule."""

    def __init__(self, args: Namespace, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`Namespace` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.atom_messages = args.atom_messages
        self.hidden_size = args.hidden_size
        self.bias = args.bias
        self.depth = args.depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.device = torch.device(args.device)
        self.aggregation = args.aggregation
        self.aggregation_norm = args.aggregation_norm

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        self.cached_zero_vector = nn.Parameter(
            torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size,
                             self.hidden_size)

        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(
                self.hidden_size + self.atom_descriptors_size,
                self.hidden_size + self.atom_descriptors_size
            )

    def forward(self,
                mol_graph: BatchMolGraph,
                atom_descriptors_batch: List[np.ndarray] = None,
                device: Optional = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.
            BatchMolGraph` representing a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing 
            additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` 
            containing the encoding of each molecule.
        """
        device = device or self.device

        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [
                np.zeros([1, atom_descriptors_batch[0].shape[1]])
            ] + atom_descriptors_batch
            atom_descriptors_batch = torch.from_numpy(np.concatenate(
                atom_descriptors_batch, axis=0)
            ).float().to(device)

        components = mol_graph.get_components(atom_messages=self.atom_messages)
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, _ = components
        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(device), f_bonds.to(device),
            a2b.to(device), b2a.to(device), b2revb.to(device)
        )

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(device)

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)
        else:
            input = self.W_i(f_bonds)
        message = self.act_func(input)

        # Message passing
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)
                nei_f_bonds = index_select_ND(f_bonds, a2b)
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)
                message = nei_message.sum(dim=1)
            else:

                nei_a_message = index_select_ND(message, a2b)
                a_message = nei_a_message.sum(dim=1)
                rev_message = message[b2revb]
                message = a_message[b2a] - rev_message

            message = self.W_h(message)
            message = self.act_func(input + message)
            message = self.dropout_layer(message)

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)
        a_message = nei_a_message.sum(dim=1)
        a_input = torch.cat([f_atoms, a_message], dim=1)  
        atom_hiddens = self.act_func(self.W_o(a_input))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            atom_hiddens = torch.cat(
                [atom_hiddens, atom_descriptors_batch], dim=1
            )
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)
            atom_hiddens = self.dropout_layer(atom_hiddens)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                if self.aggregation=='mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation=='sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation=='norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class MPN(nn.Module):
    """An :class:`MPN` is a wrapper around :class:`MPNEncoder`
    which featurizes input as needed."""

    def __init__(self, args: Namespace,
                 atom_fdim: int = None,
                 bond_fdim: int = None):
        """
        :param args: A :class:`~chemprop.args.Namespace` object containing 
            model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPN, self).__init__()
        self.atom_fdim = atom_fdim or get_atom_fdim()
        self.bond_fdim = bond_fdim or get_bond_fdim(
            atom_messages=args.atom_messages)

        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.device = args.device
        self.atom_descriptors = args.atom_descriptors

        if self.features_only:
            return

        if args.mpn_shared:
            self.encoder = nn.ModuleList([
                MPNEncoder(args, self.atom_fdim, self.bond_fdim)
            ] * args.number_of_molecules)
        else:
            self.encoder = nn.ModuleList([
                MPNEncoder(args, self.atom_fdim, self.bond_fdim)
                for _ in range(args.number_of_molecules)
            ])

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], 
                             BatchMolGraph],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                device: Optional = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecules.

        :param batch: A list of list of SMILES, a list of list of RDKit 
            molecules, or a
            :class:`~chemprop.features.featurization.BatchMolGraph`.
        :param features_batch: A list of numpy arrays containing additional 
            features.
        :param atom_descriptors_batch: A list of numpy arrays containing 
            additional atom descriptors.
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` 
            containing the encoding of each molecule.
        """
        device = device or self.device

        if type(batch[0]) != BatchMolGraph:
            if self.atom_descriptors == 'feature':
                if len(batch[0]) > 1:
                    raise NotImplementedError(
                        'Atom descriptors are currently only supported with '
                        'one molecule per input '
                        '(i.e., number_of_molecules = 1).'
                    )

                batch = [mol2graph(b, atom_descriptors_batch) for b in batch]
            else:
                batch = [mol2graph(b) for b in batch]

        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(
                features_batch
            )).float().to(device)

            if self.features_only:
                return features_batch

        if self.atom_descriptors == 'descriptor':
            if len(batch) > 1:
                raise NotImplementedError(
                    'Atom descriptors are currently only supported with one '
                    'molecule per input (i.e., number_of_molecules = 1).'
                )

            encodings = [enc(ba, atom_descriptors_batch)
                         for enc, ba in zip(self.encoder, batch)]
        else:
            encodings = [enc(ba) for enc, ba in zip(self.encoder, batch)]

        output = reduce(lambda x, y: torch.cat((x, y), dim=1), encodings)

        if self.use_input_features:
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view(1, -1)

            output = torch.cat([output, features_batch], dim=1)

        return output
