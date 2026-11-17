"""Pack independent placement tasks; all validation is on CPU before transfer."""
import numpy as np
import torch


def collate_records(records, asset_features, device):
    if not records:
        raise ValueError('Cannot batch zero tasks')
    device = torch.device(device)
    sizes = [len(r['features']) for r in records]
    if min(sizes) < 1:
        raise ValueError('Skipped/empty tasks cannot enter policy batches')
    batch, count = len(records), max(sizes)
    arrays = dict(features=np.zeros((batch,count,141),np.float32),
                  texts=np.empty((batch,768),np.float32),
                  candidate_mask=np.zeros((batch,count),bool),
                  positive=np.zeros((batch,count,8),bool),
                  rank_prob=np.zeros((batch,count,8),np.float32),
                  offset_xy=np.zeros((batch,count,8,2),np.float32),
                  offset_mask=np.zeros((batch,count,8),bool))
    for i,(record,n) in enumerate(zip(records,sizes)):
        for name in ('features','positive','rank_prob','offset_xy','offset_mask'):
            value=np.asarray(record[name])
            expected=arrays[name][i,:n].shape
            if value.shape!=expected:
                raise ValueError('Malformed cached field '+name)
            if name in ('features','rank_prob','offset_xy') and not np.isfinite(value).all():
                raise ValueError('Nonfinite cached field '+name)
            arrays[name][i,:n]=value
        mass=float(arrays['rank_prob'][i].sum(dtype=np.float64))
        if np.any(arrays['rank_prob'][i]<0) or not (mass==0 or abs(mass-1)<1e-5):
            raise ValueError('Each task rank target must have mass zero or one')
        if np.shape(record['text'])!=(768,) or not np.isfinite(record['text']).all():
            raise ValueError('Malformed cached contextual text')
        arrays['texts'][i]=record['text']
        arrays['candidate_mask'][i,:n]=True
    def transfer(array):
        tensor=torch.from_numpy(array)
        if device.type=='cuda':
            return tensor.pin_memory().to(device,non_blocking=True)
        return tensor.to(device)
    packed={name:transfer(value) for name,value in arrays.items()}
    # Asset features are validated once by the strict V6 loader and remain on GPU.
    packed['assets']=torch.stack([asset_features[r['asset_uuid']] for r in records]).to(device)
    return packed
