import collections
import numpy as np

NNLOopt_LECs = collections.OrderedDict()
NNLOopt_LECs = {'c1':-0.91863953,
                'c3':-3.88868749,
                'c4':+4.31032716,
                'Ct1S0nn':-0.15176475,
                'Ct1S0np':-0.15214109,
                'Ct1S0pp':-0.15136604,
                'Ct3S1':-0.15843418,
                'C1S0':+2.40402194,
                'C3P0':+1.26339076,
                'C1P1':+0.41704554,
                'C3P1':-0.78265850,
                'C3S1':+0.92838466,
                'CE1':+0.61814142,
                'C3P2':-0.67780851}

NNLOsat_LECs = collections.OrderedDict()
NNLOsat_LECs = {'c1'      :-1.12152119963259,
                'c3'      :-3.92500585648682,
                'c4'      : 3.76568715858592,
                'Ct1S0np' :-0.15982244957832,
                'Ct1S0nn' :-0.15915026828018,
                'Ct1S0pp' :-0.15814937937011,
                'Ct3S1'   :-0.17767436449900,
                'C1S0'    : 2.53936778505038,
                'C3P0'    : 1.39836559187614,
                'C1P1'    : 0.55595876513335,
                'C3P1'    :-1.13609526332782,
                'C3S1'    : 1.00289267348351,
                'CE1'     : 0.60071604833596,
                'C3P2'    :-0.80230029533846, 
                'cD'      : 0.81680589148271, 
                'cE'      :-0.03957471270351}

Delta_NNLOgo_394_LECs = collections.OrderedDict()
Delta_NNLOgo_394_LECs = {'c1'      :-0.74,
                         'c2'      :-0.49,
                         'c3'      :-0.65,
                         'c4'      : 0.96,
                         'Ct1S0np' :-0.3392496800000000,
                         'Ct1S0nn' :-0.3387459600000000,
                         'Ct1S0pp' :-0.3381420300000000,
                         'Ct3S1'   :-0.2598390600000000,
                         'C1S0'    : 2.5053886400000001,
                         'C3P0'    : 0.7004985700000000,
                         'C1P1'    :-0.3879598000000000,
                         'C3P1'    :-0.9648562000000001,
                         'C3S1'    : 1.0021888800000001,
                         'CE1'     : 0.4525225300000000,
                         'C3P2'    :-0.8831224000000000, 
                         'cD'      : 0.08137682,
                         'cE'      :-0.00239415}


Delta_NNLO_394_nucmat01_LECs = collections.OrderedDict()
Delta_NNLO_394_nucmat01_LECs = {'c1'      :-0.721424901712,
                                'c2'      :-0.617986316591,
                                'c3'      :-0.451627711076,
                                'c4'      : 0.862854374381,
                         'Ct1S0np' :-0.352416516590,
                         'Ct1S0nn' :-0.356747503889,
                         'Ct1S0pp' :-0.359090309522,
                         'Ct3S1'   :-0.260787546252,
                         'C1S0'    : 2.762007724416,
                         'C3P0'    : 1.272966214161,
                         'C1P1'    :-0.090158831418,
                         'C3P1'    :-1.091669238413,
                         'C3S1'    : 1.023222882268,
                         'CE1'     : 0.426412052216,
                         'C3P2'    :-0.864003568904,
                         'cD'      :-3.998467503525,
                         'cE'      :-0.812274176846}


Single_LEC = collections.OrderedDict()
Single_LEC = {'C1S0'      :2.762007724416 }

Delta_NNLOgo_394_parameters_names = list(Delta_NNLOgo_394_LECs.keys())
NNLOsat_parameter_names = list(NNLOsat_LECs.keys())
SingleLEC_parameter_name = list(Single_LEC.keys())

def setup_parameter_domain(
    LECvalues,
    mode="percentage",            # ready for future modes
    scale_factor=0.1,
    factor_overrides=None,        # e.g. {'cE':1.0,'cD':1.0,'C1S0':1.0,'CtLO':1.0}
):
    """
    Build parameter bounds according to `mode`.

    Current modes:
      - 'percentage': symmetric box around each LEC: val ± (scale_factor * |val|)
                      with optional per-LEC multiplicative overrides.

    Returns
    -------
    mid_point : np.ndarray
    lim_lo    : np.ndarray
    lim_hi    : np.ndarray
    """
    if factor_overrides is None:
        factor_overrides = {'cE':1.0, 'cD':1.0, 'C1S0':1.0, 'CtLO':1.0}

    if mode == "percentage":
        return _build_percentage_box(LECvalues, scale_factor, factor_overrides)
    else:
        raise NotImplementedError(f"mode='{mode}' not implemented yet.")


def _build_percentage_box(LECvalues, scale_factor, factor_overrides):
    """val ± (fraction * |val|), with optional per-LEC overrides (no double-counting)."""
    import numpy as np
    ctlo_set = {'Ct1S0pp','Ct1S0nn','Ct1S0np','Ct3S1'}

    lim_lo, lim_hi = [], []

    for lec, val in LECvalues.items():
        frac = scale_factor
        # per-LEC factor (e.g., 'cE', 'cD', 'C1S0')
        frac *= factor_overrides.get(lec, 1.0)
        # group factor for the CtLO family
        if lec in ctlo_set:
            frac *= factor_overrides.get('CtLO', 1.0)

        lim = frac * abs(val)
        lim_lo.append(val - lim)
        lim_hi.append(val + lim)

    lim_lo = np.array(lim_lo, dtype=float)
    lim_hi = np.array(lim_hi, dtype=float)
    mid_point = (lim_lo + lim_hi) / 2.0
    return mid_point, lim_lo, lim_hi
