from typing import List, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from ..chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from ..chemprop.features import BatchMolGraph, mol2graph

from molpal.models.mpnn.model import MoleculeModel

def predict(data_loader: Iterable, model: MoleculeModel,
            device: Optional[Union[torch.device, str, Tuple]] = None,
            disable_progress_bar: bool = False,
            scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Predict the output values of a dataset

    Parameters
    ----------
    data_loader : MoleculeDataLoader
        an iterable of MoleculeDatasets
    model : MoleculeModel
        the MoleculeModel to use
    disable_progress_bar : bool (Default = False)
        whether to disable the progress bar
    scaler : Optional[StandardScaler] (Default = None)
        A StandardScaler object fit on the training targets

    Returns
    -------
    predictions : np.ndarray
        an NxM array where N is the number of inputs for which to produce 
        predictions and M is the number of prediction tasks
    """
    if device:
        model.device = device

    model.eval()

    pred_batches = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Batch inference', unit='minibatch',
                          leave=False):
            batch_graph = batch.batch_graph()
            pred_batch = model(batch_graph)
            pred_batches.append(pred_batch.data.cpu().numpy())
    preds = np.concatenate(pred_batches)

    if model.uncertainty:
        means = preds[:, 0::2]
        variances = preds[:, 1::2]

        if scaler:
            means = scaler.inverse_transform(means)
            variances = scaler.stds**2 * variances

        return means, variances

    # Inverse scale if regression
    if scaler:
        preds = scaler.inverse_transform(preds)

    return preds
