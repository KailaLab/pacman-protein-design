
# for no processing
# the generators for non essential parameters need to provide the
# default value when called with None


def nothing(strin):
    if strin is None:
        return 'all'
    return strin


def prot_core(strin):
    return strin.split(',')


def check_enum(strin):
    if strin is None:
        return 'vdw'
    elif any([strin == i for i in ['elec', 'vdw', 'nonbonded', 'total']]):
        return strin
    else:
        mes = """ERROR: The value `{}` is not defined.\n
              \rFor this parameter the following values can
             \rbe used: \n {}
            \r""".format(strin, '\n'.join(['elec', 'vdw', 'nonbonded', 'total']))
        raise ValueError(mes)


def check_mode(strin):
    if strin is None:
        return "intra"
    elif any([strin == i for i in ['intra', 'inter']]):
        return strin
    else:
        mes = """ERROR: The value `{}` is not defined.\n
              \rFor this parameter the following values can
             \rbe used: \n {}
            \r""".format(strin, '\n'.join(['inter', 'intra']))
        raise ValueError(mes)


def check_sel2(strin):
    if check_mode(parameters['mode']) != "intra":
        return nothing(strin)
    else:
        return ''


def value2float(value):
    if value is None:
        return None
    else:
        try:
            return float(value)
        except ValueError:
            raise ValueError('ERROR: Not a valid float!')


# {'keyword in input file':(essential?,method to call(processing))}
# method should take exactly one string
# for non-essential parameters: method must return default value
parameters = {'selection': (False, nothing),
              'mode': (False, check_mode),
              'selection_2': (False, nothing),
              'energy': (False, check_enum),
              'w': (False, value2float)}
