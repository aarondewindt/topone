from math import sin, cos

import numpy as np
import xarray as xr
import xarray.ufuncs as xru


def sim_post_processing(result: xr.Dataset):
    data_var_names = set(result.data_vars)
    n_samples = len(result.t)

    env_surface_diameter = result.attrs['dynamics1_surface_diameter']
    env_initial_latitude = result.attrs['dynamics1_initial_latitude']

    if {"theta", "xii"} <= data_var_names:
        xii = result.xii.values

        zero_position = xii[0] / np.linalg.norm(xii[0]) * env_surface_diameter

        til = np.array(((-sin(env_initial_latitude), cos(env_initial_latitude)),
                        (cos(env_initial_latitude), sin(env_initial_latitude))))

        xie = np.ones((n_samples, 2)) * np.nan
        for i in range(n_samples):
            xie[i, :] = til @ (xii[i] - zero_position)

        result["xie"] = xr.DataArray(
            data=xie,
            coords=result.xii.coords,
            dims=result.xii.dims,
            name="xie",
        )
