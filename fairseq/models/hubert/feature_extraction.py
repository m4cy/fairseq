# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import torchaudio.functional as func


logger = logging.getLogger(__name__)


def complex_spec(source: torch.Tensor) -> torch.Tensor:
    # input of transform is waveform in the form of tensor, Tensor of audio of dimension (…, time).
    spectrogram = transforms.Spectrogram(
        n_fft=512,
        power=None, # returns complex spectrogram - raw spectrogram complex-valued
    )
    # Perform transform
    spectrogram = spectrogram.to(device='cuda')
    spec = spectrogram(source)
    spec = torch.transpose(spec, 1, 2)
    magnitude = torch.abs(spec) ** 2
    angle = torch.angle(spec)
    concat = torch.cat((magnitude, angle),dim=2)
    linear_layer = nn.Linear(514,512)
    linear_layer = linear_layer.to(device='cuda')
    concat = linear_layer(concat)
    concat = torch.transpose(concat, 1, 2)
    return concat.type(torch.HalfTensor).to(device='cuda')

def linear_power(source: torch.Tensor):
    logger.info("linear power")
    spectrogram = transforms.Spectrogram(
        n_fft=512,
        power=2, # returns power spectrogram
    )
    # Perform transform
    spectrogram = spectrogram.to(device='cuda')
    spec = spectrogram(source)
    spec = torch.abs(spec)
    print(spec.type())
    logspec = func.amplitude_to_DB(spec,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(spec)))
    logspec = torch.transpose(logspec, 1, 2)
    linear_layer = nn.Linear(257,512).to(device='cuda')
    logspec = linear_layer(logspec)
    logspec = torch.transpose(logspec, 1, 2)
    return logspec.type(torch.HalfTensor).to(device='cuda')
    
def logmel_power(source: torch.Tensor):
    logger.info("logmel power")
    spectrogram = transforms.Spectrogram(
        n_fft=512,
        power=2, # returns power spectrogram
    )
    # Perform transform
    spectrogram = spectrogram.to(device='cuda')
    spec = spectrogram(source)
    spec = torch.abs(spec)
    melbasis = func.melscale_fbanks(n_freqs=257, f_min=0.0, f_max=3000, n_mels=160, sample_rate=16000).to(device='cuda')
    melbasis = torch.transpose(melbasis, 0, 1)
    linear_layer = nn.Linear(257,512).to(device='cuda')
    melbasis = linear_layer(melbasis)
    spec = torch.transpose(spec, 1, 2)
    spec = linear_layer(spec)
    spec = torch.transpose(spec, 1, 2)
    melspec = torch.matmul(melbasis, spec)
    linear_layer2 = nn.Linear(160,512).to(device='cuda')
    melspec = torch.transpose(melspec, 1, 2)
    melspec = linear_layer2(melspec)
    logspec = func.amplitude_to_DB(melspec,amin=1e-10,top_db=80,multiplier=10,db_multiplier=torch.log10(torch.max(spec)))
    logspec = torch.transpose(logspec, 1, 2)

    return logspec.type(torch.HalfTensor).to(device='cuda')