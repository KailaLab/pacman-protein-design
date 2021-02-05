
# for no processing
# the generators for non essential parameters need to provide the
# default value when called with None


def nothing(strin):
    if strin is None:
        return 'default'
    return strin


def prot_core(strin):
    return strin.split(',')

# {'keyword in input file':(essential?,method to call(processing))}
# method should take exactly one string
# for non-essential parameters: method must return default value
parameters = {'michaels_test_message': (True, nothing)}