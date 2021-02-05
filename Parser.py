import sys
import os
import glob
import Plugin
import numpy as np


# defines encoding for file i/o
encoding = 'utf-8'


class Parser(object):
    """

    This class is designed to extract all necessary parameters for a PACMAN optimization from the main input file
    (per default: input.dat) and possibly any supplemental input files for snapshots named therein. Futhermore it
    handles the processing of additional keywords defined in Plugins as implemented in the class Plugin.py

    """

    def __init__(self, input_file):

        self.multiple_allowed = ['conformation', 'plugin']

        # read the input file
        with open(input_file, 'r', encoding=encoding) as f:

            self.parameters = dict()

            for line in f.readlines():
                # the last check seems to eager.
                # The line split as the iterable should catch that as a new line
                if not any((line.startswith('#'),
                            line.startswith(' '),
                            line.startswith('\n'))):
                    frags = []
                    for i in line.split():
                        if i.startswith("#"):
                            break
                        frags.append(i.strip())
                    if frags[0] not in self.parameters:  # that means if the key appears for the first time
                        if frags[0] in self.multiple_allowed:  # i.e. if the key is 'conformation'
                            self.parameters[frags[0]] = []
                            self.parameters[frags[0]].append(' '.join(frags[1:]))
                        else:
                            self.parameters[frags[0]] = ' '.join(frags[1:])
                    else:
                        if frags[0] in self.multiple_allowed:
                            self.parameters[frags[0]].append(' '.join(frags[1:]))
                        else:
                            self.write2output('Ignoring second definition of: {}'.format(frags[0]))

        # check which enviroment is used
        if "enviroment" not in self.parameters:
            self.write2output("DEFAULT: PACMAN assumes standard local configuration.\n")
            self.write2output("DEFAULT: Modules are not loaded using the  `module` command.\n")
            self.write2output("DEFAULT: To use PACMAN using modules set `enviroment` to `cluster`.\n")
            self.parameters['enviroment'] = 'local'

        # essential self.parameters that have to be present
        essential_conf = ('pdbname', 'psfname', 'segname', 'parameter_path',)
        check_econf = [i in self.parameters for i in essential_conf]

        if all(check_econf):
            self.parameters['enzyme_resids_list'] = self.create_pdb_resid_list(self.parameters['segname'])
            self.parameters['prms'] = glob.glob(os.path.join(self.parameters['parameter_path'], '*.prm'))
            self.parameters['rtfs'] = glob.glob(os.path.join(self.parameters['parameter_path'], '*.rtf'))
            self.parameters['strs'] = glob.glob(os.path.join(self.parameters['parameter_path'], '*.str'))
            self.parameters['prms'] += self.parameters['strs']
            self.parameters['rtfs'] += self.parameters['strs']
        # if not all essential keywords were in the main input file, check for supplemental input files
        else:
            if "conformation" in self.parameters:
                self.write2output(('INFO: Reading parameters for conformation from {}\n' +
                                   '').format(self.parameters['conformation']))

                self.parameters["conf_par"] = list()
                for conformation in self.parameters['conformation']:
                    self.parameters["conf_par"].append(self.read_conformation(conformation))
                # normalize weights for conformations so that the sum of absolute weights is 1
                weights = [confo['weight'] for confo in self.parameters["conf_par"]]
                abso = [abs(i) for i in weights]
                norm = [i/sum(abso) for i in weights]
                for i in range(len(self.parameters["conf_par"])):
                    self.parameters["conf_par"][i]['weight'] = norm[i]
                self.parameters['enzyme_resids_list'] = self.parameters["conf_par"][0]['enzyme_resids_list']
                # TODO: Think about this workaround
                self.parameters['rtfs'] = self.parameters["conf_par"][0]['rtfs']
            else:
                mes = "ERROR: The parameter(s) `{}` is/are missing or misspelled but crucial!"
                mes = mes.format(", ".join(i for i in essential_conf if i not in self.parameters))
                raise KeyError(mes)
        # set number of cores used by PACMAN
        if 'cpus' in self.parameters:
            self.parameters2int('cpus')
        else:
            self.parameters['cpus'] = 1
            self.write2output('DEFAULT: NAMD simulations will be run on a single core!\n')
        # set maximum number of attempted mutations
        if "cycles" in self.parameters:
            self.parameters2int('cycles')
        else:
            self.parameters['cycles'] = 10
            self.write2output('DEFAULT: PACMAN will run for 10 cycles!\n')
        # set maximum change in total charge of the system
        if "max_charge_change" in self.parameters:
            self.parameters2int('max_charge_change')
        else:
            self.parameters['max_charge_change'] = np.inf
            self.write2output('DEFAULT: The change in charge during the design process is unlimited!\n')
        # set maximum number of accepted mutations
        if "max_mutations" in self.parameters:
            self.parameters2int('max_mutations')
        else:
            self.parameters['max_mutations'] = 10
            self.write2output('DEFAULT: A maximum of 10 mutations will be performed!\n')
        # set name of main output file
        if "logfile" not in self.parameters:
            self.parameters['logfile'] = 'design.log'
            self.write2output('DEFAULT: Output will be written to {}\n'.format(self.parameters['logfile']))
        # set force with which dihedrals are restrained to optimal values during placment
        if "dihedral_force" in self.parameters:
            self.parameters2int('dihedral_force')
        else:
            self.parameters['dihedral_force'] = 90
            self.write2output('DEFAULT: A force constant of 90 will be' +
                              ' used for dihedral restraints.')
        # set temperature in the Boltzmann factor of the Metropolis-Monte-Carlo criterion
        if 'MC_temperature' in self.parameters:
            self.parameters2float('MC_temperature')
        else:
            self.parameters['MC_temperature'] = 500
            self.write2output('DEFAULT: PACMAN will use a temperature of 500' +
                              ' K for MC steps.')

        # set probability of single mutations
        if 'p_single' in self.parameters:
            self.parameters2float('p_single')
        else:
            self.parameters['p_single'] = 1.0
            self.write2output('DEFAULT: Using {} for single mutation probabilities.\n'
                              .format(self.parameters['p_single']))
        p_total = self.parameters['p_single']
        # set probability of double mutations
        if 'p_double' in self.parameters:
            self.parameters2float('p_double')
        else:
            self.parameters['p_double'] = 0.0
            self.write2output('DEFAULT: Using {} for double mutation probabilities.\n'
                              .format(self.parameters['p_double']))
        p_total += self.parameters['p_double']
        # set probabilty of triple mutations
        if 'p_triple' in self.parameters:
            self.parameters2float('p_triple')
            self.write2output('DEFAULT: Using {} for triple mutation probabilities.\n'
                              .format(self.parameters['p_triple']))
        else:
            self.parameters['p_triple'] = 0.0
        p_total += self.parameters['p_triple']
        # check whether probabilitis add up to 1, are positive deifinite and inidividually smaller than 1
        if any([(i < 0 or i > 1)
                for i in (p_total, self.parameters['p_single'],
                          self.parameters['p_double'],
                          self.parameters['p_triple'])]) \
                or abs(1 - p_total) > 1e-6:
            raise ValueError("ERROR: Please consider the axioms of Kolmogorov when" +
                             "choosing your mutation probabilities. Thank you.\n")
        # read amino acids defined in  rtfs
        full = []
        print(self.parameters['rtfs'])
        for item in self.parameters['rtfs']:
            print(item)
            with open(item, 'r', encoding=encoding) as topf:
                for line in topf.readlines():
                    if line.startswith('RESI'):
                        full.append(line.split()[1].strip())
        print(full)
        full = list(set(full))
        # set set of amino acids which can be mutation targets
        if 'aa_set' in self.parameters:
            natural = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HSD', 'ILE', 'LEU', 'LYS', 'MET',
                       'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
            # set of amino acids without CYS and PRO
            if self.parameters['aa_set'] == 'soluble':
                aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'GLY', 'HSD', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
                          'SER', 'THR', 'TRP', 'TYR', 'VAL']
            # set of apolar amino acids without CYS and PRO
            elif self.parameters['aa_set'] == 'transmembrane':
                aminos = ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'TYR', 'VAL']
            # set of all amino acids
            elif self.parameters['aa_set'] == 'natural':
                aminos = natural
            # use this with extreme caution if at all!
            elif self.parameters['aa_set'] == 'full':
                sys.stderr.write("WARNING: You are allowing all RESI entries in your rtfs as mutation targets!\n" +
                                 "WARNING: This is probably a very bad idea as it easily leads to problems.\n" +
                                 "WARNING: Consider specifying a list of allowed residues instead!\n" +
                                 "WARNING: Make sure all of them can fit in a protein chain and add them to\n" +
                                 "WARNING: the dictionary of dihedral angles if necessary! Good Luck!\n")
                aminos = full
                self.write2output("INFO: The following residues are possible mutation targets: \n")
                self.write2output(full)
            else:
                aminos = [i.strip() for i in self.parameters['aa_set'].split(',')]
                if not all([i in full for i in aminos]):
                    raise ValueError("ERROR: You can only specify either 'full', 'soluble', 'transmembrane' or a " +
                                     "comma separated list of the following amino acids: {}".format(' '.join(full)))
            self.parameters['aa_set'] = aminos
        else:
            self.parameters['aa_set'] = ['ALA', 'ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'GLY', 'HSD', 'ILE', 'LEU', 'LYS',
                                         'MET', 'PHE', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
            self.write2output('DEFAULT: PACMAN assumes all 20 natural amnino acids are legit targets for mutation\n' +
                              'DEFAULT: Please consult the manual for other options.\n')

        check_list = [i in full for i in self.parameters['aa_set']]
        if not all(check_list):
            inexistents = [bi for bi, ci in zip(self.parameters['aa_set'], check_list) if not ci]
            raise ValueError('ERROR: kindly consider checking whether your rtf files contain all possible' +
                             'mutation targets. Hint: These residues {} are not contained.\n'.format(inexistents))

        # set whether global side chain relaxation is allowed after mutation
        if 'fixed_sidechains' in self.parameters:
            self.noyesbool('fixed_sidechains')
        else:
            self.parameters['fixed_sidechains'] = False
            self.write2output('DEFAULT: It is assumed that you do not want to restrain global adaption of' +
                              'your protein.\n')
        # set whether an initial minimization is performed before the evaluation of the input structure
        if 'minimize' in self.parameters:
            self.noyesbool('minimize')
        else:
            self.parameters['minimize'] = True
            self.write2output("DEFAULT: Your starting structure will be minimized before evaluation\n")
        # generate a list of enzyme resids that will not be mutated
        # set residues which should not be mutated
        if 'immutables' in self.parameters:
            self.parameters['immutables'] = self.parameter2residlist('immutables')
            self.parameters['immutables_vmd'] = ('resid {} and segname {}' +
                                                 '').format(' '.join([str(i) for i in self.parameters['immutables']]),
                                                            self.parameters['segname'])

            # remove immutables from the resid list
            if 'conf_par' in self.parameters:
                for snap in self.parameters['conf_par']:
                    snap['enzyme_resids_list'] = list(set(snap['enzyme_resids_list']).difference(
                        self.parameters['immutables']))
            else:
                self.parameters['enzyme_resids_list'] = list(set(self.parameters['enzyme_resids_list']).difference(
                    self.parameters['immutables']))

        else:
            self.parameters['immutables'] = []
            self.parameters['immutables_vmd'] = ''

        # set the solvent used during namd simulations (affects which templates are selected
        if 'solvent' in self.parameters:
            self.parameter_check_enum('solvent', ['vacuum',
                                                  'GBIS',
                                                  'water',
                                                  'membrane'])
        else:
            self.parameters['solvent'] = 'vacuum'
            self.write2output("DEFAULT: All MD simulations during the design process will be performed in vacuo.\n")
        # set the criterion for Monte-Carlo evalutaion of the mutations
        if 'MC_criterion' in self.parameters:
            self.parameter_check_enum('MC_criterion', ['vdw',
                                                       'elec',
                                                       'nonbonded',
                                                       'external'])
            # enable Plugins for advanced properties to be evaluated
            if self.parameters['MC_criterion'] == 'external':

                self.plugin_pars = {}
                if 'plugin' not in self.parameters:
                    raise ValueError("ERROR: If you want to use an external MC criterion you have " +
                                     "to name the plugin to use." +
                                     "Try again or read the README")

                tmp_plugin = []
                try:
                    for i in self.parameters['plugin']:
                        """plugins should be specified as plugin _label_:_pluginname_ """

                        try:
                            label, plug = i.split(':')
                        except ValueError:
                            raise ValueError("Plugins should be specified as _label_:_pluginname_!")

                        plugin_par_func = Plugin.Plugin.get_plugin_pars(plug)

                        plugin_pars = {}

                        for par in plugin_par_func.keys():
                            par_w_label = "{}:{}".format(label, par)
                            try:
                                plugin_pars[par] = plugin_par_func[par][1](self.parameters[par_w_label])
                            except KeyError:
                                if plugin_par_func[par][0]:
                                    mes = 'Plugin `{}` is missing essential argument `{}`!'.format(plug, par_w_label)
                                    raise Plugin.PluginArgError(mes)
                                else:
                                    plugin_pars[par] = plugin_par_func[par][1](None)
                        try:
                            key = '{}:w'.format(label)
                            self.parameters2float(key)
                            plugin_pars['w'] = self.parameters[key]
                            if plugin_pars['w'] < 0:
                                mes = "ERROR: Plugin weights must be > 0!"
                                raise ValueError(mes)
                            if plugin_pars['w'] == 0:
                                sys.stderr.write("WARNING: Ypu set a plugin weight to 0.0 . If you just want to" +
                                                 " monitor a property" +
                                                 " without designing according to it,a weight of 0 might be useful," +
                                                 " IF the plugin is inexpensive! Do you really want to do this?\n")
                        except KeyError:
                            plugin_pars['w'] = None
                        tmp_plugin.append((plug, plugin_pars))

                    if any([t[1]['w'] is not None for t in tmp_plugin]) and not all([t[1]['w'] for t in tmp_plugin]):
                        mes = "ERROR: Either all plugins have a specified weight attribute _label_:w or none."
                        raise RuntimeError(mes)

                    elif all([t[1]['w'] is None for t in tmp_plugin]):
                        for t in tmp_plugin:
                            t[1]['w'] = 1/len(tmp_plugin)

                    else:
                        sum_w = sum([t[1]['w'] for t in tmp_plugin])
                        for t in tmp_plugin:
                            t[1]['w'] = t[1]['w']/sum_w

                except:
                    raise ImportError("plugin definition did not work...")

                self.parameters['plugin'] = tmp_plugin
            else:
                self.parameters['plugin'] = []

        else:
            self.parameters['MC_criterion'] = 'vdw'
            self.parameters['plugin'] = ' '
            self.parameters['plugin_par'] = {}
            self.write2output("DEFAULT: Mutations will be accepted or rejected" +
                              "according to the change in vdw-energy.\n")
        # set the property-dependent threshold to be surpassed by the properties
        if 'threshold' not in self.parameters:
            if self.parameters['MC_criterion'] == 'vdw':
                self.parameters['threshold'] = -2.5
            elif self.parameters['MC_criterion'] == 'elec':
                self.parameters['threshold'] = -20.0 * 4184
            elif self.parameters['MC_criterion'] == 'nonbonded':
                self.parameters['threshold'] = -50.0 * 4184
            else:  # only possibilty left is external
                raise ValueError("ERROR: If you want to use an external MC criterion you have " +
                                 "to define a custom threshold. Try again or read the README")
        else:
            self.parameters['threshold'] = self.value2float(self.parameters['threshold'])

        # set inter-atomic distance restraints
        if 'constraint_pairs' in self.parameters:
            self.parameters['constraint_pairs'] = self.get_constraints_pairs()
            self.parameters['constraint'] = True

            if 'constraint_lengths' in self.parameters:
                self.parameters['constraint_lengths'] = self.get_constraints_attr('constraint_lengths')
            else:
                self.parameters['constraint_lengths'] = [4] * len(self.parameters['constraint_pairs'])
                self.write2output('DEFAULT: PACMAN will use standard lengths of 0.4 nm for constraints.\n')

            if 'constraint_forces' in self.parameters:
                self.parameters['constraint_forces'] = self.get_constraints_attr('constraint_forces')
            else:
                self.parameters['constraint_forces'] = [200] * len(self.parameters['constraint_pairs'])
                self.write2output('DEFAULT: PACMAN will use a standard force constant of 200 for constraints.\n')

        else:
            self.parameters['constraint'] = False
            self.parameters['constraint_pairs'] = []
            self.parameters['constraint_forces'] = []
            self.parameters['constraint_lengths'] = []
        if 'prefix' not in self.parameters and 'conformation' not in self.parameters:
            self.parameters['prefix'] = ''  # self.parameters['pdbname'].split('.pdb')[0]
            self.write2output('DEFAULT: Files will be named after the provided PDB file.\n')

        self.write_full_config()

    def get_constraints_attr(self, key, col=None):
        """
        This function is used to set constraint lengths and constraint forces if pairwise constraints are to be used.
        :param key: key in the Parser.parameters dictionary given as col
        :type key: String
        :param col: a Parser.parameters dictionary with keywords from input.dat as keys
        and entries from input.dat as entries
        :type col: parameters dictionary
        :return:
        :rtype: list
        """

        if col is None:
            col = self.parameters

        attr = []
        if '/' in col[key]:
            items = col[key].split('/')

            if len(items) == len(col['constraint_pairs']):
                for item in items:
                    try:
                        attr.append(self.value2float(item))
                    except ValueError:
                        raise ValueError('ERROR: The conversion of a constraint distance failed!')
        else:
            attr = [self.value2float(col[key])] * \
                   len(col['constraint_pairs'])

        return attr

    def get_constraints_pairs(self, col=None):
        """
        Read and process constraint pairs from a parameters dict. Differerent pairs are separated by '/' and members of
        a single pair by ,'. Residues involved in constraints must be immutable.
        :param col: The parameters dict from which constraint pairs are read
        :type col: dict
        :return: list of tuples of atom identifiers (format: chain-resname-resid-atomname)
        :rtype: list(tuple(str))
        """

        residues = []

        if col is None:
            col = self.parameters

        for item in col['constraint_pairs'].split(','):

            try:
                resids = [int(i.split('-')[2]) for i in item.split('/')]
            except ValueError:
                raise ValueError('Constraint pairs not well defined! Check form and int!')

            if not all([i in self.parameters['immutables'] for i in resids]):
                raise ValueError("ERROR: The atoms you wanted to constrain could potentially be mutated. " +
                                 "They will not be constrained. " +
                                 "Please add them to the immutable keyword\n")
            else:
                residues.append(item.split('/'))

        return residues

    def parameter2residlist(self, key):
        """
        This converts residue descriptions of the form 1,2,3,4-8,11,... (str) to a list of integers.
        :param key: the key under which the resid list is stored as str in the parameters dict
        :type key: str
        :return: The list of resids described before as str
        :rtype: list(int)
        """

        element_list = [i.strip() for i in self.parameters[key].split(',')]

        resids = []
        for item in element_list:

            if '-' in item:
                try:
                    lower, upper = item.split('-')
                except ValueError:
                    raise ValueError(('ERROR: Convention for `{}` hurt with {}.' +
                                      'Use comma separated integers or ' +
                                      'ranges i.e. 1-2, 5').format(key, item))
                try:
                    resids += list(range(int(lower), int(upper) + 1))
                except ValueError:
                    mes = ('ERROR: Only use integers for resids in {}' +
                           '').format(key)
                    raise ValueError(mes)

            else:
                try:
                    i = int(item)
                    if i != float(item):
                        raise ValueError
                    else:
                        resids.append(int(item))
                except ValueError:
                    mes = ('ERROR: Only use integers for resids in {}' +
                           '').format(key)
                    raise ValueError(mes)

        self.parameters[key] = resids
        check_list = [resid in self.parameters['enzyme_resids_list'] for resid in resids]
        if not all(check_list):
            inexistents = [bi for bi, ci in zip(resids, check_list) if not ci]
            mes = "ERROR: Inexistent residue identifier ({}) used in {} !".format(inexistents, key)
            raise ValueError(mes)
        return resids

    @staticmethod
    def value2int(value):
        """
        This returns value as a integer point number (if possible)
        :param value: the value to be converted
        :type value: something convertible to int
        :return: the converted value
        :rtype: int
        """

        try:
            # very explicit sanity checking for a valid int
            if int(value) != float(value):
                raise ValueError

            return int(value)
        except ValueError:
            raise ValueError('ERROR: Not a valid integer!')

    @staticmethod
    def value2float(value):
        """
        This returns value as a floating point number (if possible)
        :param value: the value to be converted
        :type value: something convertible to float
        :return: the converted value
        :rtype: float
        """

        try:
            return float(value)
        except ValueError:
            raise ValueError('ERROR: Not a valid float!')

    def noyesbool(self, key):
        """
        This converts the no/yes syntax of the input file to boolean values
        :param key: The key under which the entry to be converted is stored in self.parameters
        :type key: str
        """
        if self.parameters[key] == 'no' or self.parameters[key] == 'NO' or self.parameters[key] == 'No':
            self.parameters[key] = False
        elif self.parameters[key] == 'yes' or self.parameters[key] == 'YES' or self.parameters[key] == 'Yes':
            self.parameters[key] = True
        else:
            raise ValueError('ERROR: `{}` can only be set yes/no.'.format(key))

    def parameters2int(self, key):
        """
        This converts a entry which was read from the input file and stored in self.parameters to int
        :param key: The key under which the entry to be converted is stored in self.parameters
        :type key: str
        """

        try:
            self.parameters[key] = self.value2int(self.parameters[key])
        # raised if eiter a non numeric value is provided,
        # or if it is not an interger i.e. 0.9
        except ValueError:
            mes = ('ERROR: The number (`{}`) you provided for ' +
                   '`{}` can not be converted to an integer!' +
                   '').format(self.parameters[key], key)
            raise ValueError(mes)

    def parameters2float(self, key):
        """
        This converts an entry which was read from the input file and stored in self.parameters to float
        :param key: The key under which the entry to be converted is stored in self.parameters
        :type key: str
        """

        try:
            self.parameters[key] = self.value2float(self.parameters[key])
        except ValueError:
            raise ValueError(('ERROR: The number (`{}`) you provided for ' +
                              '`{}` can not be converted to a float!' +
                              '').format(self.parameters[key], key))

    def parameter_check_enum(self, key, valid_values):
        """
        Checks if the entry for a certain key matches any of the allowed values for this keyword.
        :param key: The key under which the entry to be checked is stored in self.parameters
        :type key: str
        :param valid_values: Allowed values for the entry at this key
        :type valid_values: list(str)
        """

        if any([self.parameters[key] == i for i in valid_values]):
            pass
        else:
            mes = """ERROR: The value `{}` for `{}` is not defined.\n
                  \rFor this parameter the following values can
                  \rbe used: \n {}
                  \r""".format(self.parameters[key],
                               key,
                               '\n'.join(valid_values))
            raise ValueError(mes)

    def create_pdb_resid_list(self, segname=None, pdbfile=None):
        """
        Creates a list of all residue ids in a pdb/ filete. Buffers against discontinuous .pdb files.
        :param pdbfile: Path to the .pdb file to be read
        :type pdbfile: str
        :param segname: the name of the segment on which mutations should be performed later on
        :type segname: str
        :return: The list of all residue ids found in the .pdb file (only reading lines starting with ATOM)
        :rtype: list(int)
        """

        # the following lines ensure that also discontinuous pdb files can be used
        tmplist = []
        enzyme_resids_list = []

        if pdbfile is None:
            pdbfile = self.parameters['pdbname']
        with open(pdbfile, 'r', encoding=encoding) as inpdb:

            for line in inpdb:
                if line.startswith("ATOM") and (line[72:76].strip() == segname or segname is None):
                    tmplist.append(line[22:26].strip())
        for item in tmplist:
            if int(item) not in enzyme_resids_list:
                enzyme_resids_list.append(int(item))

        return enzyme_resids_list

    @staticmethod
    def write2output(st):
        """flushing the stdout should keep the output in order, even on highly buffered systems."""

        sys.stdout.write(st)
        sys.stdout.flush()

    def write_full_config(self):
        """
        Output function to write everything stored in self.parameters to the terminal.
        """

        for key in self.parameters:
            if key != 'conf_par':
                self.write2output('INPUT: {:<30}{:<40}\n'.format(key, str(self.parameters[key])))
            else:
                for snap in self.parameters[key]:
                    self.write2output("\n")
                    for subkey in snap:
                        self.write2output('INPUT: {:<30}{:<40}\n'.format(subkey, str(snap[subkey])))
                    self.write2output("\n")

    def read_conformation(self, conformation):
        """
        Reads supplemental input file for a snapshot and writes a dict in the style of self.parameters
        :param conformation: The Path to the input file for the conformation
        :type conformation: str
        :return: Dictionary supplementing self.parameters with all other input key/entry pairs
        :rtype: dict
        """

        conf = dict()

        with open(conformation, 'r', encoding=encoding) as f:

            for line in f:
                # the last check seems to eager.
                # The line split as the iterable should catch that as a new line
                if not any((line.startswith('#'),
                            line.startswith(' '),
                            line.startswith('\n'))):
                    frags = [i.strip() for i in line.split()]
                    conf[frags[0]] = ' '.join(frags[1:])
        conf['prefix'] = conformation.split('/')[-2]
        essential_conf = ('pdbname', 'psfname', 'parameter_path',)
        check_econf = [i in conf for i in essential_conf]

        if all(check_econf):

            for i in essential_conf:
                if not conf[i].startswith('/'):
                    conf[i + "_abs"] = os.path.join('/'.join(conformation.split('/')[:-1]), conf[i])
                else:
                    conf[i + "_abs"] = conf[i]

            conf['enzyme_resids_list'] = self.create_pdb_resid_list(segname=self.parameters['segname'],
                                                                    pdbfile=conf['pdbname_abs'])
            conf['prms'] = glob.glob(os.path.join(conf['parameter_path_abs'], '*.prm'))
            conf['rtfs'] = glob.glob(os.path.join(conf['parameter_path_abs'], '*.rtf'))
            conf['strs'] = glob.glob(os.path.join(conf['parameter_path_abs'], '*.str'))
            conf['rtfs'] += conf['strs']
            conf['prms'] += conf['strs']

        else:
            mes = "ERROR: The parameter(s) `{}` in {} is/are missing or misspelled but crucial!"
            mes = mes.format(", ".join(i for i in essential_conf if i not in self.parameters),
                             conformation)
            raise KeyError(mes)

        # inter-atomic distance restraints
        if 'constraint_pairs' in conf:
            conf['constraint_pairs'] = self.get_constraints_pairs(col=conf)
            conf['constraint'] = True

            if 'constraint_lengths' in conf:
                conf['constraint_lengths'] = self.get_constraints_attr('constraint_lengths',
                                                                       col=conf)
            else:
                conf['constraint_lengths'] = [4] * len(conf['constraint_pairs'])
                self.write2output('DEFAULT: PACMAN will use standard lengths of 0.4 nm for constraints.\n')

            if 'constraint_forces' in conf:
                conf['constraint_forces'] = self.get_constraints_attr('constraint_forces',
                                                                      col=conf)
            else:
                conf['constraint_forces'] = [200] * len(conf['constraint_pairs'])
                self.write2output('DEFAULT: PACMAN will use a standard force of 0.4 nm for constraints.\n')

        else:
            conf['constraint'] = False
            conf['constraint_forces'] = []
            conf['constraint_pairs'] = []
            conf['constraint_lengths'] = []
        if 'weight' in conf:
            conf['weight'] = self.value2float(conf['weight'])
        else:
            conf['weight'] = 1
        return conf
