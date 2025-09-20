import jax.numpy as jnp
import numpy as np
from dsps.data_loaders import retrieve_fake_fsps_data

ssp_data = retrieve_fake_fsps_data.load_fake_ssp_data()
ssp_lgmet = ssp_data.ssp_lgmet
ssp_lg_age_gyr = ssp_data.ssp_lg_age_gyr

n_met = ssp_lgmet.size
n_age = ssp_lg_age_gyr.size


def load_fake_ssp_halpha():
    arr = jnp.array(
        [
            18.99350973,
            15.18340321,
            18.59590888,
            14.51430117,
            15.50420145,
            18.89018259,
            14.6840321,
            15.44170473,
            18.62229038,
            18.23519039,
            15.46710684,
            18.09569229,
            18.72593381,
            18.74917029,
            18.84602614,
            18.90889461,
            20.18508488,
            19.64529091,
            20.39642718,
            19.94077289,
            20.36111634,
            20.77151474,
            21.80102465,
            22.86217362,
            24.00485424,
            25.53092415,
            26.12223314,
            29.38864892,
            30.45950998,
            34.16892198,
            39.6796686,
            39.59340563,
            33.72510107,
            29.60391916,
            23.58247219,
            16.62595202,
            13.70698426,
            8.85376548,
            2.83265497,
            2.0991581,
            1.48929108,
            0.96965843,
            0.58217645,
            0.35032972,
            0.22578957,
            0.14377021,
            0.09140591,
        ]
    )
    noise_std = 2
    ssp_halpha_line_luminosity = [
        np.clip(
            arr + np.random.normal(0, noise_std, size=arr.shape), a_min=0, a_max=None
        )
        for _ in range(n_met)
    ]
    zeros = np.zeros(n_age - len(arr))
    ssp_halpha_line_luminosity = [
        np.append(_, zeros) for _ in ssp_halpha_line_luminosity
    ]
    return jnp.array(ssp_halpha_line_luminosity)
