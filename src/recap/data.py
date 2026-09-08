"""Validate episode boundaries and dataset-specific action contracts before inference."""
from pathlib import Path

def validate_episode(path, dataset, observed_frames, horizon):
    import numpy as np
    with np.load(path,allow_pickle=False) as archive:
        images=archive['image'];actions=archive['action']
        if images.ndim!=4 or images.shape[1:]!=(256,320,3) or images.dtype!=np.uint8:
            raise ValueError('image must be uint8 [T,256,320,3]')
        if actions.shape!=(len(images),13) or not np.issubdtype(actions.dtype,np.floating) or not np.isfinite(actions).all():
            raise ValueError('action must be finite floating point [T,13]')
        if len(images)<observed_frames+horizon:raise ValueError('episode too short for protocol')
        if dataset=='bridge' and np.any(actions[:,7:12]!=0):raise ValueError('Bridge slots 7–11 must be zero')
        if dataset=='calvin' and np.any(actions[:,[0,1,2,7,8,9]]!=0):raise ValueError('CALVIN unused slots must be zero')
        return {'episode':Path(path).name,'frames':len(images),'shape':list(images.shape),'action_dim':13}
