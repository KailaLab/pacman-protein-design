import os
import sys
import random
import stat
import shutil
import subprocess
import pandas as pd
import numpy as np
import re
import itertools
import copy
import Plugin
import dictionaries_AA as Dcts
from math import ceil


# this has to change once the program is packed!
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
encoding = 'utf-8'
debuging = False


class Snap(object):

    """

    This class combines all the intrinsic attributes of a specific snapshot. It contains all methods to access and
    manipulate structural properties of the snapshot. Since each snapshot has the same sequence, sequential changes
    are addressed by a function in main.

    """

    def __init__(self, prefix, pdbfile, psffile, cpus, weight, env, solv, mode, threshold, outputfile,
                 immutable_resids, mutable_resids, parameters, topologies, segname,
                 constraint_pairs, constraint_forces, constraint_lengths,
                 plugin, fixed, max_charge_change):

        self.directory = os.path.join(os.getcwd(), prefix, 'run/')
        self.input_directory = os.path.join(os.getcwd(), prefix)
        # names without path or relative path
        self.pdbfile = pdbfile
        self.psffile = psffile
        if prefix != '':
            self.name = prefix
        else:
            self.name = pdbfile.split('.')[0]
        self.mode = mode
        self.threshold = threshold
        # NAMD submission scripts for single points and dynamics
        if mode == 'external':
            self.external = True
            self.plugins = plugin
        else:
            self.external = False
        self.cpus = cpus
        self.weight = weight
        self.env = env
        self.solvent = solv
        self.parameterfiles = parameters
        self.topologyfiles = topologies
        self.segname = segname
        self.immutable_resids = immutable_resids

        self.mutable_resids = mutable_resids
        self.constraint_pairs = constraint_pairs
        self.constraint_forces = constraint_forces
        self.constraint_lengths = constraint_lengths
        self.constrained = len(self.constraint_pairs) > 0
        self.mut_ids = ['-', '-', '-']
        self.mut_nms = ['-', '-', '-']
        self.curres = ['-', '-', '-']
        # dummy initial values for structural properties (because pep8 wants them in __init__)
        self._charge = 0
        self._ref_electrostatics = 0
        self._ref_nonbonded = 0
        self._ref_vdw = 0
        self._ref_ext = 0
        self._ref_ext_out = 0
        self._mutations = 0
        self._current_electrostatics = 0
        self._current_nonbonded = 0
        self._current_vdw = 0
        self._current_ext = 0
        self._current_ext_out = 0
        self.center = []
        self.vectors = []
        self.pmes = []
        self.files = dict()
        self.dirs = dict()

        # for safety reasons first delete the self.directory I am about to create
        if os.path.exists(self.directory):
            shutil.rmtree(os.path.join(self.input_directory, 'run_bak/'), ignore_errors=True)
            shutil.move(self.directory, os.path.join(self.input_directory, 'run_bak/'))

        # create a brand new empty self.directory
        os.mkdir(self.directory)

        # copy the initial def files to the newly created self.directory
        shutil.copy2(os.path.join(self.input_directory, pdbfile), self.directory)
        shutil.copy2(os.path.join(self.input_directory, psffile), self.directory)

        # make yet another set of pdb and psf files
        shutil.copy2(os.path.join(self.input_directory, psffile),
                     os.path.join(self.directory, 'sp.psf'))
        shutil.copy2(os.path.join(self.input_directory, pdbfile),
                     os.path.join(self.directory, 'sp.pdb'))

        # store the file names for easy access
        self.files['pdb'] = os.path.join(self.directory, pdbfile)
        self.files['psf'] = os.path.join(self.directory, psffile)
        self.files['psf_mutated'] = os.path.join(self.directory, 'mutated.psf')
        self.files['pdb_mutated'] = os.path.join(self.directory, 'mutated.pdb')
        self.files['log'] = os.path.join(self.directory, outputfile)

        self.files['psf_sp'] = os.path.join(self.directory, 'sp.psf')
        self.files['pdb_sp'] = os.path.join(self.directory, 'sp.pdb')
        self.files['sp_out'] = os.path.join(self.directory, 'sp.out')

        # same goes for all the related directories
        self.dirs['pdb_dir'] = os.path.join(self.directory, 'PDBS')
        self.dirs['acc_dir'] = os.path.join(self.directory, 'Accepted')
        self.dirs['crash_dir'] = os.path.join(self.directory, 'Crash_reports')
        # make all auxiliary/results directories
        for aux_dir in self.dirs.values():
            shutil.rmtree(aux_dir, ignore_errors=True)
            os.mkdir(aux_dir)

        self.records = self.read_pdb()
        self.updated_records = self.read_pdb()

        self.fixed_indices, self.fixed_entries = self.get_fixed()
        if len(self.fixed_indices) > 0:
            self.fixed_indices = list(self.fixed_indices)
            self.fixed_entries = list(self.fixed_entries)
            self.fixed_vmd = self.get_vmd_sel_from_entries(self.fixed_entries)
        else:
            self.fixed_vmd = "none"

        self.fixed_resids = []
        for entry in self.fixed_entries:
            if self.get_segname_from_entry(entry) == self.segname:
                self.fixed_resids.append(self.get_resid_from_entry(entry))
        self.mutable_resids = list(set(self.mutable_resids).difference(set(self.immutable_resids + self.fixed_resids)))

        self.fixed = fixed

        self.uglies = self.get_hammond(1000)

        # write necessary scripts
        self.write_all()

        self._initial_charge = self.charge
        self._charge_limit = max_charge_change

    @property
    def charge(self):
        """
        Calculates and returns  the charge of a snapshot via vmd
        :return: the charge attribute of the Snap
        :rtype: float
        Attributes modified:
            self._charge
        """

        vmd_cmd = "vmd -dispdev text -e {}".format(self.files['chargescript'])
        self.run_vmd_command(vmd_cmd)
        with open(self.files['chargefile'], 'r', encoding=encoding) as f:
            self._charge = float(f.read())

        return self._charge

    @charge.setter
    def charge(self, value):
        """
        :param value: the charge to be set as the charge of the Sanpshot
        :type value: float
        Attributes modified:
            self._charge
        """

        self._charge = value

####################################
#           Utilities              #
####################################

    @staticmethod
    def find_fft(n, uglies):
        """
        Finds the first integer from a list of integers that is higher than n.
        :param n: The value to be surpassed
        :type n: float
        :param uglies: list of integers from which the smallest greater than n is chosen
        :type uglies: list
        :return: the first number from the list greater then n
        :rtype: int
        """
        m = ceil(n)
        for i in uglies:
            if i >= m:
                return i

    def get_attributes(self, stage='mut'):
        """
        Calculates and returns nonbonded energies (and if applicable externally calculated stuff) for the Snap structure
        Attributes modified:
            self._current_nonbonded
            self._current_electrostatics
            self._current_vdw
        """

        self.get_nonbonded_energy()
        if self.external:
            out = self.run_plugin(stage)
            current_ext = np.asarray([i[0] for i in out])
            current_ext_out = np.asarray([i[1] for i in out])
            self._current_ext = current_ext.sum(axis=None)
            self._current_ext_out = current_ext_out.sum(axis=None)

    def get_box(self, boxpdb):
        """
        Generates box origin and basis vectors of the box used for PBC accroding to namd conventions. Generates also
        appropriate PME parameters to handle long range electrostatics for the generated box.
        :param boxpdb: name of the .pdb file containing coordinates of the system
        :type boxpdb: str
        Attributes modified:
            self.centers
            self.vectors
            self.pmes
        """
        self.write_get_box(boxpdb)
        self.run_vmd_command(vmd_cmd="vmd -dispdev text -e {}".format(self.files['boxscript']))
        with open(self.files['boxfile'], 'r', encoding=encoding) as boxvectors:
            minmax = boxvectors.readline().split()
            mm = []
            for item in minmax:
                mm.append(item.strip('{').strip('}'))
            self.center = boxvectors.readline().split()
            self.vectors = [float(mm[3])-float(mm[0]),
                            float(mm[4])-float(mm[1]),
                            float(mm[5])-float(mm[2])]
            self.pmes = [self.find_fft(self.vectors[0], self.uglies),
                         self.find_fft(self.vectors[1], self.uglies),
                         self.find_fft(self.vectors[2], self.uglies)]

    def get_entry(self, index, records='default'):
        """
        Generates entry in pacman format from atom index
        :param index: atom index
        :type index: int
        :param records: The dataframe in which the records for the protein in question are stored
        :type records: pandas dataframe
        :return: entry in pacman format (segname-resname-resnr-atomName)
        :rtype: str
        """
        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        segname = records[(records['atom_serial'] == str(index))]['segname'].values[0]
        resname = records[(records['atom_serial'] == str(index))]['resname'].values[0]
        resnr = records[(records['atom_serial'] == str(index))]['resid'].values[0]
        aname = records[(records['atom_serial'] == str(index))]['atom_name'].values[0]
        entry = segname+"-"+resname+"-"+resnr+"-"+aname
        if len(entry) == 0:
            if str(records) == 'default':
                raise ValueError('Entry {} not in {}!'.format(entry, self.files['pdb']))
            else:
                raise ValueError('Entry {} not in {}!'.format(entry, self.files['pdb_mutated']))
        return str(entry)

    def get_fasta(self, outfile=None, records='default'):
        """
        Generates the sequence information for a protein in fasta format from a dataframe (as generated by read_pdb())
        and writes it to a file.
        :param outfile: The name of the file to which the sequence information is written
        :type outfile: str
        :param records: The dataframe in which the records for the protein in question are stored
        :type records: pandas dataframe
        :return: sequence
        :rtype: str
        """
        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        aadict = {
          'GLY': 'G',
          'ALA': 'A',
          'VAL': 'V',
          'LEU': 'L',
          'ILE': 'I',
          'CYS': 'C',
          'MET': 'M',
          'PHE': 'F',
          'TYR': 'Y',
          'TRP': 'W',
          'ASP': 'D',
          'GLU': 'E',
          'ASN': 'N',
          'GLN': 'Q',
          'HIS': 'H',
          'HSE': 'H',
          'HSP': 'H',
          'HSD': 'H',
          'ARG': 'R',
          'LYS': 'K',
          'SER': 'S',
          'THR': 'T',
          'PRO': 'P'
        }
        fasta_string = []
        resids = []
        for resid in records['resid'].values:
            if resid not in resids:
                resname = self.get_name_from_resid(resid)
                if resname in aadict.keys():
                    fasta_string.append(aadict[resname])
                resids.append(resid)
        if outfile:
            with open(os.path.join(self.directory, outfile), 'w', encoding=encoding) as fasta_file:
                outstring = ''
                for i, item in enumerate(fasta_string):
                    outstring += item
                    if (int(i)+1) % 80 == 0:
                        outstring += "\n"
                fasta_file.write(outstring)
                fasta_file.write("\n")
        else:
            return fasta_string

    def get_fixed(self, records='default'):
        """
        Finds fixed residues as specified by 1.00 in the occupancy column of a .pdb file and returns their indices and
        entries (pacman format)
        :param records: dataframe generated by the read_pdb() method
        :type records: pandas dataframe
        :return: lists containing indices and entries of fixed residues
        :rtype: tuple of lists
        """
        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        fixed_entries = []
        fixed_indices = [] + list(records[(records['occ'] == '1.00')]['atom_serial'].values)
        for i in fixed_indices:
            fixed_entries.append(self.get_entry(int(i)))
        return fixed_indices, fixed_entries

    @staticmethod
    def get_hammond(n):
        """
        Calculates the first n Hamming numbers. Useful for efficient PME.
        :param n: number of calculated ugly numbers
        :type n: int
        :return: list of the first n Hamming numbers
        :rtype: list
        """
        hammond = [0] * n
        hammond[0] = 1
        i2 = i3 = i5 = 0
        next_x2 = 2
        next_x3 = 3
        next_x5 = 5
        for i in range(1, n):
            hammond[i] = min(next_x2, next_x3, next_x5)
            if hammond[i] == next_x2:
                i2 += 1
                next_x2 = hammond[i2] * 2
            if hammond[i] == next_x3:
                i3 += 1
                next_x3 = hammond[i3] * 3
            if hammond[i] == next_x5:
                i5 += 1
                next_x5 = hammond[i5] * 5
        return hammond

    def get_index(self, entry, records='default'):
        """
        Accesses the records of a snapshot to find the index of the atom specified by the entry.
        :param entry: Unique atom identifier which has the format Chain-Resname-Resid-Atomname.
        :type entry: str
        :param records: Flag deciding whether the dataframe containing sequence information is the one for the
        mutated or original sequence
        :type records: str
        :return: Index of the atom specified by the entry
        :rtype: int
        """

        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        segname = entry.split('-')[0].strip()
        resname = entry.split('-')[1].strip()
        resnr = entry.split('-')[2].strip()
        aname = entry.split('-')[3].strip()
        index = records[(records['segname'] == segname) & (records['resname'] == resname) &
                        (records['resid'] == resnr) & (records['atom_name'] == aname)]['atom_serial']
        if len(index) == 0:
            if str(records) == 'default':
                raise ValueError('Entry {} not in {}!'.format(entry, self.files['pdb']))
            else:
                raise ValueError('Entry {} not in {}!'.format(entry, self.files['pdb_mutated']))
        elif len(index) > 1:
            if str(records) == 'default':
                raise ValueError('Entry {} more than once in {}!'.format(entry, self.files['pdb']))
            else:
                raise ValueError('Entry {} more than once in {}!'.format(entry, self.files['pdb_mutated']))
        return int(index)

    def get_name_from_index(self, index, records='default'):
        """
        Accesses the records of a snapshot to find the name of the residue containing a atom with a certain index.
        :param index: Atom serial index
        :type index: int
        :param records
        :type records
        :return: The name of the residue containing the atom specified by index
        :rtype: Dataframe entry?
        """

        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        resname = records[records['atom_serial'] == str(index)]['resname']
        if len(resname) == 0:
            if str(records) == 'default':
                raise ValueError('Resid {} not in {}!'.format(index, self.files['pdb']))
            else:
                raise ValueError('Resid {} not in {}!'.format(index, self.files['pdb_mutated']))
        elif len(resname) > 1:
            if str(records) == 'default':
                raise ValueError('Index {} more than once in {}!'.format(index, self.files['pdb']))
            else:
                raise ValueError('Index {} more than once in {}!'.format(index, self.files['pdb_mutated']))
        return resname.values[0]

    def get_name_from_resid(self, resid, records='default'):
        """
        Accesses the records of a snapshot to find the name of the residue with a certain residue id.
        :param resid: The id of the residue
        :type resid: int
        :param records
        :type records
        :return: The name of the residue containing the atom specified by index
        :rtype: Dataframe entry?
        """

        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        resname = records[records['resid'] == str(resid)]['resname']
        if len(resname) == 0:
            if str(records) == 'default':
                raise ValueError('Resid {} not in {}!'.format(resid, self.files['pdb']))
            else:
                raise ValueError('Resid {} not in {}!'.format(resid, self.files['pdb_mutated']))

        return resname.values[0]

    def get_nonbonded_energy(self):
        """
        Calculates and returns the nonbonded energies (total, electrostatic and vdw) in kcal/mol
        :return: The total nonbonded energy, electrostatic energy term and vdw energy term of the Snap as calculated by
        a namd single point calculation
        :rtype: float, float, float
        Attributes modified:
            self._current_nonbonded
            self._current_electrostatics
            self._current_vdw
        """

        energy_line = ''
        with open(self.files['sp_out'], 'r', encoding=encoding) as namd_log:
            for line in namd_log:
                if line.startswith('ENERGY'):
                    energy_line = line
        columns = energy_line.split()
        nonbonded_j_mol = 4184.0 * float(columns[6]) + float(columns[7])
        electrostatics_j_mol = 4184.0 * float(columns[6])
        vdw_j_mol = 4184.0 * float(columns[7])
        self._current_nonbonded = nonbonded_j_mol
        self._current_electrostatics = electrostatics_j_mol
        self._current_vdw = vdw_j_mol
        return nonbonded_j_mol, electrostatics_j_mol, vdw_j_mol



    @staticmethod
    def get_resid_from_entry(entry):
        """
        Extracts the resid from a pacman style entry
        :param entry: entry in pacman style
        :type entry: str
        :return: residue id of the residue specified by the entry
        :rtype: int
        """
        return int(entry.split('-')[2].strip())

    def get_segnames_files(self):
        """
        This function writes .pdb files containing one segmnet of the pdb stored in self.pdbs each.
        Attributes modified:
            self.files['seg_pdbs']
        """

        bb = os.path.join(self.directory, 'seg_{}.pdb')
        self.files['seg_pdbs'] = {}

        with open(self.files['pdb'], 'r', encoding=encoding) as f:
            head = f.readline()
            end = 'END'

            line = f.readline()

            all_segments = []
            current_segment = [line]
            current_segment_identifier = line[72:76].strip()

            all_segments.append(current_segment_identifier)

            for line in f:
                if line[72:76].strip() == current_segment_identifier:
                    current_segment.append(line)
                else:
                    filename = bb.format(current_segment_identifier.strip())
                    with open(filename, 'w', encoding=encoding) as of:
                        of.write(''.join([head] + current_segment + [end]))

                    self.files['seg_pdbs'][current_segment_identifier] = filename

                    current_segment = [line]
                    current_segment_identifier = line[72:76].strip()

    @staticmethod
    def get_segname_from_entry(entry):
        """
        Extracts the segname from a pacman style entry
        :param entry: entry in pacman style
        :type entry: str
        :return: name of the segment specified in the entry
        :rtype: str
        """
        return str(entry.split('-')[0].strip())

    def get_segnames_from_resid(self, resid, records='default'):
        """
        Uses a dataframe as generated by read_pdb() to find the name of segments containing a certain residue id
        :param resid: the id of the residue to be queried
        :type resid: int
        :param records: The dataframe in which the records for the protein in question are stored
        :type records: pandas dataframe
        :return: unique list of all segments containing the specified residue id
        :rtype: list
        """
        if str(records) == 'default':
            records = self.records
        elif str(records) == 'updated':
            records = self.updated_records
        else:
            raise ValueError('The records you specified ({}) do not exist'.format(records))
        segnames = records[(records['resid'] == str(resid))]['segname'].values[0]
        return list(set(segnames))

    def get_structural_details_from_psf(self, filename='default'):
        """
        Reads bonds, angles, dihedrals and impropers from a .psf structure file
        :param filename: name of the .psf file to be read
        :type filename: str
        :return:
        :rtype:
        """
        if filename == 'default':
            filename = self.files['psf_sp']
        details = {
                    'bonds': 0,
                    'nr_bonds': 0,
                    'angles': 0,
                    'nr_angles': 0,
                    'dihedrals': 0,
                    'nr_dihedrals': 0,
                    'impropers': 0,
                    'nr_impropers': 0,
                    'donors': 0
                  }
        detail_keys = {
                        'bonds': 2,
                        'angles': 3,
                        'dihedrals': 4,
                        'impropers': 4
                      }

        with open(filename, 'r') as f:
            content = f.read()

        aggregate = ''
        current_kw = 'bonds'
        # read all the indices from the psf file
        for line in content.split('\n'):
            if not line.startswith(' REMARKS'):
                kw = [i in line for i in details.keys()]
                if any(kw):
                    details[current_kw] = aggregate
                    aggregate = ''
                    current_kw = list(itertools.compress(list(details.keys()), kw))[0]
                    details['nr_{}'.format(current_kw)] = int(line.split('!')[0])
                else:
                    aggregate += line

        # form and reshape index arrays
        details = self.shape_indices(details, detail_keys)
        # check integrity of psf file
        check = [details[key].shape[0] == details['nr_{}'.format(key)]
                 for key in detail_keys]
        if not all(check):
            problem = list(itertools.compress(list(detail_keys.keys()),
                                              [not i for i in check]))
            err_str = ""
            err_temp = "\n\t Found indices for {} {}, but expected \n\t" + \
                       " indices for {} {} from psf file!"
            for prob in problem:
                err_str += err_temp.format(details[prob].shape[0],
                                           prob,
                                           details['nr_{}'.format(prob)],
                                           prob)
            raise ValueError(err_str)
        return {key: details[key]-1 for key in detail_keys}

    @staticmethod
    def get_vmd_sel_from_entries(entries):
        """
        Creates a vmd selection string from a list of pacman style entries
        :param entries: list of pacman style entries
        :type entries: list
        :return: vmd selection string
        :rtype: str
        """
        sel = []
        for entry in entries:
            segname = entry.split('-')[0].strip()
            resnr = entry.split('-')[2].strip()
            aname = entry.split('-')[3].strip()
            sel.append("("+" segname "+segname+" and resid "+resnr+" and name "+aname+")")
        selstr = " or ".join(sel)
        return selstr

    def read_pdb(self, filename='default'):
        """
        Reads a pdb file and safes for each ATOM line the contained information to a dataframe
        :param filename: The name of the .pdb to be read. If default, self.files['pdb'] is read.
        :type filename: str
        :return: A dataframe containing all information about ATOMs in the .pdb file
        :rtype: pd.DataFrame
        """

        if filename == 'default':
            filename = self.files['pdb']
        with open(filename, 'r') as f:
            content = f.read()
        ff_start = [0, 6, 12, 16, 17, 21, 22, 26, 30, 38, 46, 54, 60, 72, 76, 78]
        ff_end = [4, 11, 16, 17, 21, 22, 26, 27, 38, 46, 54, 60, 66, 76, 78, 80]
        atom_entries = []
        for line in content.split('\n'):
            if line.startswith('ATOM'):
                tmp = []
                for s, e in zip(ff_start, ff_end):
                    if e > len(line):
                        continue
                    tmp.append(line[s:e].strip())
                atom_entries.append(tmp)
        columns = ['record_type', 'atom_serial', 'atom_name', 'alt_loc',
                   'resname', 'chain_identifier', 'resid', 'ins',
                   'x', 'y', 'z', 'occ', 'beta', 'segname', 'element', 'charge']
        return pd.DataFrame(atom_entries, columns=columns[:len(atom_entries[0])])

    def read_psf(self, filename='default'):
        """
        read PSF... more like read the first parts,
        only !NATOM all other sections are ignored.
        """
        if filename == 'default':
            filename = self.files['psf_sp']
        with open(filename, 'r') as f:
            content = f.read()

        section = ''
        for section in re.compile("[0-9].*?!").split(content):
            if section.startswith('NATOM'):
                break

        entries = [i.split() for i in section[5:].split('\n')
                   if i != "" and i != "NATOM"][:-1]
        columns = ['id', 'segname', 'resid', 'resname', 'atom_name', 'atom_type',
                   'charge', 'mass', 'unused']

        return pd.DataFrame(entries, columns=columns)

    @staticmethod
    def write_pdb(records, filename):
        """
        Reads a pdb file and safes for each ATOM line the contained information to a dataframe
        :param filename: The name of the .pdb to be written.
        :type filename: str
        :param records: records to be written.
        :type filename: str
        """
        columns = ['record_type', 'atom_serial', 'atom_name', 'alt_loc',
                   'resname', 'chain_identifier', 'resid', 'ins',
                   'x', 'y', 'z', 'occ', 'beta', 'segname', 'element', 'charge']

        row_temp = ["{:4s}  ", "{:>5s}  ", "{:<4s}", "{:0s}", "{:<4s}", "{:1s}", "{:>4s}", "{:1s}   ",
                    "{:>8s}", "{:>8s}", "{:>8s}", "{:>6s}", "{:>6s}      ", "{:<4s}", "{:>2s}", "{:2s}"]

        with open(filename, 'w') as f:
            for index, row in records.iterrows():
                r = ""
                for c, rt in zip(columns, row_temp):
                    try:
                        r += rt.format(str(row[c]))
                    except:
                        pass

                f.write(r + "\n")
            f.write('END')

    def run_namd_sp(self):
        """
        Uses the namd .conf file generated by write_namd_sp_conf to run a single point calculation on the Snap
        Attributes modified:
            self.files['namd_submit']
            self.files['sp_out']
        """
        # if self.solvent == 'water' or self.solvent == 'membrane':
        #     self.write_namd_sp_conf()
        outfile = os.path.join(self.directory, 'sp.out')
        self.write_namd_submit(self.files['namd_sp'], outfile)
        self.files['sp_out'] = outfile
        p = subprocess.Popen(self.files['namd_submit'])
        p.wait()

    def run_plugin(self, stage):
        """
        Wrapper function to call a plugin to calculate a property and return it. If the plugin crashes during
        the initial evaluation, pacman exits with an error.
        :param stage: Flag to specify whether the initial structure or a mutant is evaluated
        :type stage: str
        :return: tuple of floats returned by the plugin. Returns np.inf if plugin crashed
        :rtype: float, float
        """
        self.write2output("INFO: Starting external calculation using a plugin!\n")

        out_values = []
        for plug in self.plugins:
            self.write2output("INFO: PLUGIN: {}\n".format(plug.name))
            try:
                out = plug.exec_plugin(copy.deepcopy(self))
            except Plugin.PluginReturnError as e:
                sys.stderr.write(e)
            except Exception as e:
                sys.stderr.write(e)
                if stage != 'min':
                    self._current_ext = np.inf
                    self._current_ext_out = np.inf
                else:
                    sys.exit("ERROR: The plugin {} did abort during the evaluation" +
                             "of the initial structure!\n".format(self.plugins.name) +
                             "ERROR: PACMAN aborted!\n")
            else:
                out_values.append(out)

        return out_values

    # TODO: check modules usage
    def run_vmd_command(self, vmd_cmd, debug=debuging):
        """
        Shorthand for running VMD commands. Debug Flag to be set manually to get vmd output for troubleshooting.
        """
        #  debug = True
        # very unspecific clause! better check for shark and lrz
        if self.env == 'cluster':
            if debug:
                os.system("source /etc/profile.d/modules.sh; module load gcc; module load vmd; module load namd; {}"
                          .format(vmd_cmd))
            else:
                os.system("source /etc/profile.d/modules.sh; module load gcc; module load namd; " +
                          "module load vmd; {}".format(vmd_cmd+" >/dev/null 2>&1"))
        elif self.env == 'supermuc':
            if debug:
                os.system(". /etc/profile; . /etc/profile.d/modules.sh; " +
                          "module load gcc; module load vmd; module load namd; {}".format(vmd_cmd))
            else:
                os.system(". /etc/profile; . /etc/profile.d/modules.sh; " +
                          "module load gcc; module load vmd; module load namd; {}".format(vmd_cmd+" >/dev/null 2>&1"))
        else:
            if debug:
                os.system(vmd_cmd)
            else:
                os.system(vmd_cmd+" >/dev/null 2>&1")

    @staticmethod
    def shape_indices(indices, keys):
        """

        :param indices:
        :type indices:
        :param keys:
        :type keys:
        :return:
        :rtype:
        """
        for key in keys:
            try:
                indices[key] = np.asarray(indices[key].split(),
                                          dtype=int).reshape(-1,
                                                             keys[key])
            except ValueError:
                raise ValueError("""Could not form details array for {}.
                                    \r\tThere are {} indices given, which is not
                                    \r\tdivisible by {}.
                                    \r\tFix your psf file and come back!
                                 """.format(key,
                                            len(indices[key].split()),
                                            keys[key]))
        return indices

    @staticmethod
    def write2output(st):
        """flushing the stdout should keep the output in order, even on highly buffered systems."""

        sys.stdout.write(st)
        sys.stdout.flush()

    def write_all(self):
        """
        Convenience function that calls all of the write script methods of Snap
        Attributes modified:
            self.files['chargefile']            (write_get_charge())
            self.files['chargescript']          (write_get_charge())
            self.files['coor2pdbscript']        (write_coor2pdb())
            self.files['tcl_mutate_1]           (write_tcl_mutates())
            self.files['tcl_mutate_2]           (write_tcl_mutates())
            self.files['tcl_mutate_3]           (write_tcl_mutates())
            self.files['namd_sp']               (write_namd_sp_conf())
            self.files['namd_mini']             (write_namd_mini_conf())
            self.files['namd_place_free']       (write_namd_place_free_conf())
            self.files['namd_place_restraint']  (write_namd_place_restraint_conf())
        """

        self.write_get_charge()
        self.write_coor2pdb()
        self.write_tcl_mutates()

        self.write_namd_sp_conf()
        self.write_namd_place_free_conf()
        self.write_namd_place_restraint_conf()

    def write_constraints(self, output='bonds.dat', records='default'):
        """
        Generates an extrabonds file for namd simulations to constrain the distances between atom pairs.
        :param output: Name of the file to which the extrabonds entries are written.
        :type output: str
        :param records: Flag deciding the dataframe to be used for getting indices of constrained atoms
        :type records: str
        """

        script = []
        for pair, force, length in zip(self.constraint_pairs,
                                       self.constraint_forces,
                                       self.constraint_lengths):
            i1 = self.get_index(pair[0], records)
            i2 = self.get_index(pair[1], records)
            script.append('bond {} {} {} {}'.format(i1, i2, force, length))
        with open(os.path.join(self.directory, output), 'w', encoding=encoding) as outfile:
            for item in script:
                outfile.write('{0}\n'.format(item))

    def write_coor2pdb(self):
        """
        Write coor2pdb.tcl script to be used as vmd -dispdev text -e coor2pdb.tcl -args psffile coord outpdb
        Attributes modified:
             self.files['coor2pdbscript']
        """

        coor2pdb = os.path.join(self.directory, 'coor2pdb.tcl')
        # copy the template
        shutil.copy2(os.path.join(template_dir, '_coor2pdb.tmp'), coor2pdb)
        # store the coor2pdb name for later use
        self.files['coor2pdbscript'] = coor2pdb

    def write_get_box(self, pdbfile):
        """
        Generates a .tcl script which is used by get_box to calculate the simulation box for a system.
        :param pdbfile: name of the .pdb file containing coordinates of the system
        :type pdbfile: str
        """
        with open(os.path.join(template_dir, '_get_box.tmp'), 'r', encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        boxscript = os.path.join(self.directory, "get_box.tcl")
        with open(boxscript, 'w', encoding=encoding) as f:
            boxfile = os.path.join(self.directory, 'box.dat')
            f.write(template.format(pdb=pdbfile,
                                    outfile=boxfile))
        self.files['boxfile'] = boxfile
        self.files['boxscript'] = boxscript

    def write_get_charge(self):
        """
        Reads the template _get_charge.tmp and writes the tcl script to get the charges of a protein via vmd.
        Attributes modified:
            self.files['chargefile']
            self.files['chargescript']
        """

        # read the template
        with open(os.path.join(template_dir, '_get_charge.tmp'), 'r', encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        chargescript = os.path.join(self.directory,
                                    "get_charge.tcl")

        with open(chargescript, 'w', encoding=encoding) as f:
            chargefile = os.path.join(self.directory, 'charge.dat')
            f.write(template.format(psf=self.files['psf'],
                                    name=chargefile))

        # store the chargefile name for later use
        self.files['chargefile'] = chargefile
        self.files['chargescript'] = chargescript

    def write_log(self, header=False, cycle=0, acceptance=1, reason="original structure"):
        """
        This method is used to write the file design.log for each Snap, which contains for the original structure and
        each attempted mutation the energies, attempted mutations, acceptance of mutations and reason for accepting
        or declining a mutation.
        :param header: Flag deciding whether column headers or column entries should be written.
        :type header: bool
        :param cycle: number of the mutation cycle
        :type cycle: int
        :param acceptance: Whether the attempted mutation was accepted or not
        :type acceptance: bool
        :param reason: The reason for accepting or declining a mutation attempt
        :type reason: str
        """

        if header:
            if self.external:
                template = '{0:7s}   {1:14s}  {2:14s}\t{3:14s}\t{4:14s}    {5:30s}\t{6:>8s}\t{7:>10s}\t{8:>11s}   '
                template += '{9:>8s}\t{10:>10s}\t{11:>11s}   {12:>8s}\t{13:>10s}\t{14:>11s}\t{15:<20s}\n'
                values = ['attempt', 'criterion', 'nonbonded', 'electrostatics',
                          'vdW energy', 'acceptance', 'resid 1', 'resname 1',
                          'mutation 1', 'resid 2', 'resname 2', 'mutation 2',
                          'resid 3', 'resname 3', 'mutation 3', 'reason']
            else:
                template = '{0:7s}   {1:14s}\t{2:14s}\t{3:14s}\t{4:30s}\t{5:>8s}\t{6:>10s}\t{7:>11s}   '
                template += '{8:>8s}\t{9:>10s}\t{10:>11s}   {11:>8s}\t{12:>10s}\t{13:>11s}\t{14:<20s}\n'
                values = ['attempt', 'nonbonded', 'electrostatics',
                          'vdW energy', 'acceptance', 'resid 1', 'resname 1',
                          'mutation 1', 'resid 2', 'resname 2', 'mutation 2',
                          'resid 3', 'resname 3', 'mutation 3', 'reason']
        else:
            if self.external:
                template = '{0:7d}   {1:14.2f}  {2:14.2f}\t{3:14.2f}\t{4:14.2f}\t{5:30d}\t{6:>8s}\t{7:>10s}\t{8:>11s} '
                template += '  {9:>8s}\t{10:>10s}\t{11:>11s}   {12:>8s}\t{13:>10s}\t{14:>11s}\t{15:<20s}\n'
                values = [cycle,
                          self._current_ext_out,
                          self._current_nonbonded/1000,
                          self._current_electrostatics/1000,
                          self._current_vdw/1000,
                          acceptance]
                values += ' '.join(['{} {} {}'.format(r, rn, mut) for r, rn, mut in zip(self.mut_ids,
                                                                                        self.curres,
                                                                                        self.mut_nms)]).split()
                values += [reason]
            else:
                template = '{0:7d}   {1:14.2f}\t{2:14.2f}\t{3:14.2f}\t{4:30d}\t{5:>8s}\t{6:>10s}\t{7:>11s}   '
                template += '{8:>8s}\t{9:>10s}\t{10:>11s}   {11:>8s}\t{12:>10s}\t{13:>11s}\t{14:<20s}\n'
                values = [cycle,
                          self._current_nonbonded/1000,
                          self._current_electrostatics/1000,
                          self._current_vdw/1000,
                          acceptance]
                values += ' '.join(['{} {} {}'.format(r, rn, mut) for r, rn, mut in zip(self.mut_ids,
                                                                                        self.curres,
                                                                                        self.mut_nms)]).split()
                values += [reason]
        with open(os.path.join(self.directory, "design.log"), 'a', encoding=encoding) as outfile:
            outfile.write(template.format(*values))

    def write_namd_sp_conf(self, stage='initial'):
        """
        Writes a namd .conf file suitable to run a single point calculation. Could be adapted to use a higher
        level of theory for this calculation (And was used that way historically).
        Attributes modified:
            self.files['namd_sp']
        """
        if stage == 'initial':
            psffile = self.files['psf']
            pdbfile = self.files['pdb']
            self.files['xscfile'] = os.path.join(self.directory, "minim.xsc")
        else:
            psffile = self.files['psf_sp']
            pdbfile = self.files['pdb_sp']
            self.files['xscfile'] = os.path.join(self.directory, "place_free.xsc")
        # read the template
        with open(os.path.join(template_dir, '_sp_'+str(self.solvent)+'_conf.tmp'), 'r', encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        filename = os.path.join(self.directory, "sp.conf")

        parameters = '\n'.join('parameters       {}'.format(i) for i in self.parameterfiles)
        if self.solvent == 'vacuum' or self.solvent == 'GBIS':
            with open(filename, 'w', encoding=encoding) as f:
                f.write(template.format(psf=psffile,
                                        pdb=pdbfile,
                                        output=os.path.join(self.directory, "sp_out"),
                                        parameters=parameters))
        else:
            with open(filename, 'w', encoding=encoding) as f:
                f.write(template.format(psf=psffile,
                                        pdb=pdbfile,
                                        output=os.path.join(self.directory, "sp_out"),
                                        parameters=parameters,
                                        xsc=self.files['xscfile']
                                        ))
        # store the file name for later use
        self.files['namd_sp'] = filename

    def write_namd_mini_conf(self, mini=True):
        """
        Writes the namd .conf file used to perform the initial minimisation / equilibration + constraint enforcement
        of the snapshots.
        Attributes changed:
            self.files['namd_mini']
        """

        # read the template
        if not mini:
            with open(os.path.join(template_dir, '_no_mini_'+str(self.solvent)+'_conf.tmp'),
                      'r', encoding=encoding) as f:
                template = f.read()
        else:
            with open(os.path.join(template_dir, '_mini_'+str(self.solvent)+'_conf.tmp'), 'r', encoding=encoding) as f:
                template = f.read()

        # write the tcl script with all additions
        filename = os.path.join(self.directory, "mini.conf")

        parameters = '\n'.join('parameters       {}'.format(i) for i in self.parameterfiles)
        if self.solvent == 'vacuum' or self.solvent == 'GBIS':
            with open(filename, 'w', encoding=encoding) as f:
                f.write(template.format(psf=self.files['psf'],
                                        pdb=self.files['pdb'],
                                        parameters=parameters,
                                        bonds=os.path.join(self.directory, 'bonds.dat')))
        else:
            self.get_box(self.files['pdb'])
            with open(filename, 'w', encoding=encoding) as f:
                f.write(template.format(psf=self.files['psf'],
                                        pdb=self.files['pdb'],
                                        parameters=parameters,
                                        bonds=os.path.join(self.directory, 'bonds.dat'),
                                        x=self.vectors[0],
                                        y=self.vectors[1],
                                        z=self.vectors[2],
                                        cx=self.center[0],
                                        cy=self.center[1],
                                        cz=self.center[2],
                                        pmex=self.pmes[0],
                                        pmey=self.pmes[1],
                                        pmez=self.pmes[2]))
        # store the file name for later use
        self.files['namd_mini'] = filename

    def write_namd_place_free_conf(self):
        """
        Writes the namd .conf file used to equilibrate and minimize a snapshot after successful placement of the mutated
        side chain.
        Attributes modified:
            self.files['namd_place_free']
        """

        # read the template
        with open(os.path.join(template_dir, '_place_free_'+str(self.solvent)+'_conf.tmp'),
                  'r',
                  encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        filename = os.path.join(self.directory,
                                "place_free.conf")

        parameters = '\n'.join('parameters       {}'.format(i) for i in self.parameterfiles)
        with open(filename, 'w', encoding=encoding) as f:
            f.write(template.format(psf=self.files['psf_mutated'],
                                    pdb=self.files['pdb_mutated'],
                                    output=os.path.join(self.directory, 'place_free'),
                                    inputs=os.path.join(self.directory, 'place_restraint'),
                                    parameters=parameters,
                                    bonds=os.path.join(self.directory, 'bonds.dat'),
                                    consref=self.files['pdb_mutated'],
                                    consfile=os.path.join(self.directory, 'constr.pdb')))
        # store the file name for later use
        self.files['namd_place_free'] = filename

    def write_namd_place_restraint_conf(self):
        """
        Writes the namd .conf file used to minimise the protein side chains around the newly mutated residue and
        equilibrate afterwards.
        Attributes changed:
            self.files['namd_place_restraint']
        """

        # read the template
        with open(os.path.join(template_dir, '_place_restraint_'+str(self.solvent)+'_conf.tmp'),
                  'r',
                  encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        filename = os.path.join(self.directory,
                                "place_restraint.conf")

        parameters = '\n'.join('parameters       {}'.format(i) for i in self.parameterfiles)
        with open(filename, 'w', encoding=encoding) as f:
            f.write(template.format(psf=self.files['psf_mutated'],
                                    pdb=self.files['pdb_mutated'],
                                    output=os.path.join(self.directory, 'place_restraint'),
                                    parameters=parameters,
                                    bonds=os.path.join(self.directory, 'dihedrals.dat'),
                                    consref=self.files['pdb_mutated'],
                                    consfile=os.path.join(self.directory, 'constr.pdb')))
        # store the file name for later use
        self.files['namd_place_restraint'] = filename

    def write_namd_submit(self, conf, outfile, cores=None):
        """
        This function writes a tempolate based submit.sh file, which can be executed to run a namd simulation.
        :param conf: Name of a namd .conf file containing information about the simulation to be run.
        :type conf: str
        :param outfile: Name of the namd .log file to which the simulation log will be written.
        :type outfile: str
        :param cores
        Attributes modified:
            self.files['namd_submit']
        """
        if cores is None:
            cores = self.cpus
        filename = os.path.join(self.directory, "submit_namd.job")
        with open(os.path.join(template_dir, '_namd_script_'+self.env+'.tmp'), encoding=encoding) as f:
            template = f.read()
        values = {'direc': self.directory,
                  'cpus': cores,
                  'conf': conf,
                  'out': outfile}

        with open(filename, 'w', encoding=encoding) as f:
            f.write(template.format(**values))

        os.chmod(filename, os.stat(filename).st_mode | stat.S_IEXEC)
        # store the file name for later use
        self.files['namd_submit'] = filename

    # TODO: user defined patches
    def write_tcl_mutation_script(self, resids, resnames):
        """
        Writes a .tcl script to be executed via vmd that will use the procedure written by write_tcl_mutates to initiate
        the mutation of one, two or three amino acids to specified target residues.
        :param resids: The residue ids in the amino acid sequence at which the mutation(s) should be performed
        :type resids: list(int)
        :param resnames: The residue names of the amino acids to which the residue ids should be muatated
        :type resnames: list(str)
        Attributes changed:
            self.files['tcl_do_mutation']
        """
        # read the template
        with open(os.path.join(template_dir, '_tcl_do_mutation.tmp'), 'r', encoding=encoding) as f:
            template = f.read()

        # write the tcl script with all additions
        filename = os.path.join(self.directory, "do_mutation.tcl")

        num = len(resids)
        assert num < 4
        args = ' '.join(['{} {}'.format(resid, resname) for resid, resname in zip(resids, resnames)])
        mutate_tcl = self.files['tcl_mutate_{}'.format(num)]
        if self.fixed and len(self.fixed_indices) > 0:
            vmd_fix = self.fixed_vmd + ' or (all and (not same residue as within 3 of (resid {} and segname {})) '
            vmd_fix += ')'
        elif len(self.fixed_indices) > 0 and not self.fixed:
            vmd_fix = self.fixed_vmd
        elif self.fixed and not len(self.fixed_indices) > 0:
            vmd_fix = 'all and (not same residue as within 3 of (resid {} and segname {}))'
            vmd_fix += ')'
        else:
            vmd_fix = 'none'
        if self.fixed:
            if len(resids) > 1:
                vmd_fix = vmd_fix.format([' '.join(str(resids[x])) for x in range(len(resids))], self.segname)
            else:
                vmd_fix = vmd_fix.format(str(resids[0]), self.segname)
        with open(filename, 'w', encoding=encoding) as f:
            f.write(template.format(mutate_tcl=mutate_tcl,
                                    args=args,
                                    seg_pdb=self.files['seg_pdbs'][self.segname],
                                    pdb_updated=self.files['pdb_mutated'],
                                    fixed_vmd=vmd_fix,
                                    fix_namd_pdb=os.path.join(self.directory, 'fix_namd.pdb'),
                                    constr_pdb=os.path.join(self.directory, 'constr.pdb')))

        # store the file name for later use
        self.files['tcl_do_mutation'] = filename

    def write_tcl_mutates(self):
        """
        Write .tcl scripts that contain the procedure to perform a single, double or triple point mutation on a given
        protein structure. This procedure will in practice be sourced by the .tcl file generated by
        write_tcl_mutation_script
        Attributes modified:
            self.files['tcl_mutate_1]
            self.files['tcl_mutate_2]
            self.files['tcl_mutate_3]
        """

        # read the template
        with open(os.path.join(template_dir, '_tcl_mutate.tmp'), 'r', encoding=encoding) as f:
            template = f.read()
        topology = '\n'.join('topology     {}'.format(i) for i in self.topologyfiles)
        for i in range(3):
            # write the tcl script with all additions
            filename = os.path.join(self.directory,
                                    "mutate{}.tcl".format(i+1))
            args = ' '.join(['pos{} mut{}'.format(ii+1, ii+1) for ii in range(i+1)])
            mutations = '\n'.join(['mutate $pos{} $mut{}'.format(ii+1, ii+1) for ii in range(i+1)])
            pos = ' '.join(['pos{}'.format(ii+1) for ii in range(i+1)])
            with open(filename, 'w', encoding=encoding) as f:
                f.write(template.format(psf_updated=self.files['psf_mutated'],
                                        pdb_updated=self.files['pdb_mutated'],
                                        topology=topology,
                                        args=args,
                                        mutations=mutations,
                                        pos=pos,
                                        segname=self.segname,
                                        dir=self.directory))
            # store the file name for later use
            self.files['tcl_mutate_{}'.format(i+1)] = filename

####################################
#           Actions                #
####################################

    def namd_minimize(self, mini):
        """
        Use the namd .conf file written by write_namd_mini_conf to minimize and equilibrate the initial structure of
        the Snap while enforcing constraints.
        Attributes modified:
            self.files['namd_submit']           (write_namd_submit())
            self.files['minim_out']
        """
        self.write_namd_mini_conf(mini)
        outfile = os.path.join(self.directory, 'minim.out')
        if mini:
            self.write_namd_submit(self.files['namd_mini'], outfile)
        else:
            self.write_namd_submit(self.files['namd_mini'], outfile, 1)
        self.write_constraints()
        self.files['minim_out'] = outfile
        # start process and wait for it to finish
        p = subprocess.Popen(self.files['namd_submit'])
        p.wait()
        vmd_cmd = "vmd -dispdev text -e {mini} -args {pdb_in} {coor_in} {pdb_out}"
        self.run_vmd_command(vmd_cmd.format(mini=self.files['coor2pdbscript'],
                                            pdb_in=self.files['pdb'],
                                            coor_in=os.path.join(self.directory, 'minim.coor'),
                                            pdb_out=self.files['pdb']))
        shutil.copy(self.files['pdb'], self.files['pdb_sp'])
        shutil.copy(self.files['psf'], self.files['psf_sp'])
        self.run_namd_sp()
        self.get_attributes(stage='min')
        self.write_log(header=True)
        self.write_log()

    def mutate(self, resids, resnames, curres):
        """
        Splits the Snap structure into segment pdbs using get_segname_files and performs mutations on the segment
        specified in the input using scripts generated by write_tcl_mutation_scripts.
        :param resids: The sequence position(s) to be mutated
        :type resids: list(int)
        :param resnames: The new residue(s) at said position(s) after the mutation
        :type resnames: list(str)
        :param curres: The residue(s) at said position(s) before the mutation
        :type curres: list(str)
        Attributes modified:
            self.files['seg_pdbs']

        """
        ref_charge = self.charge
        self.get_segnames_files()
        self.write_tcl_mutation_script(resids, resnames)
        for i in range(3):
            if i < len(resids):
                self.mut_ids[i] = resids[i]
                self.mut_nms[i] = resnames[i]
                self.curres[i] = curres[i]
            else:
                self.mut_ids[i] = '-'
                self.mut_nms[i] = '-'
                self.curres[i] = '-'
        self.run_vmd_command(vmd_cmd="vmd -dispdev text -e {}".format(self.files['tcl_do_mutation']))
        self.files['psf_sp'] = self.files['psf_mutated']
        if abs(ref_charge - self.charge) > 2 or abs(self._initial_charge - self.charge) > self._charge_limit:
            self.write_log(cycle=np.nan, acceptance=False, reason='charge')
            return True
        else:
            return False

    def place_sidechain(self, dihedral_force):
        """
        This function reads the optimal dihedral angles for the mutated amino acid(s) from a dict based on
        rotamer libraries and writes a extrabond file to enforce these together with possible other constraints. Then,
        a constrained namd simulation of the mutated structure is started as a subprocess to minimize the protein
        side chains around the new residue(s) and equilibrate the mutated structure.
        :param dihedral_force: Force constant for dihedral restraints
        :type dihedral_force: int
        :return: The subprocess in which the namd simulation is run
        :rtype: subprocess
        """

        self.updated_records = self.read_pdb(self.files['pdb_mutated'])
        if self.constrained:
            self.write_constraints(output='dihedrals.dat', records='updated')
            self.write_constraints(records='updated')  # constraints for the free equilibration
        # write dummy files for bond.dat and dihdral.dat if there are no constraints
        else:
            with open(os.path.join(self.directory, 'bonds.dat'), 'w', encoding=encoding) as f:
                f.write("\n")
            with open(os.path.join(self.directory, 'dihedrals.dat'), 'w', encoding=encoding) as f:
                f.write("\n")
        # initiate list with needed dictionaries
        list_used_dicts = []
        for AAresname, AAresid in zip(self.mut_nms, self.mut_ids):
            if AAresname != '-':
                Dcts.res[AAresname]['curresid'] = str(AAresid)
                list_used_dicts.append(Dcts.res[AAresname])
        # if dihedral constraints are required (not only ALAs and GLYs as AAresnames):
        if any(dictionary['n_dihedrals'] > 0 for dictionary in list_used_dicts):
            # make list with all dihedral angles of all AAresnames
            list_dihedral_angles = []
            for dictionary in list_used_dicts:
                for chi_value in ['chi1_value', 'chi2_value', 'chi3_value', 'chi4_value']:
                    if chi_value in dictionary:
                        list_dihedral_angles.append(dictionary[chi_value])
            # initiate list for atom indices for all dihedral angles of all AAresnames
            list_atom_indices = []
            for dictionary in list_used_dicts:
                # for every dihedral angle that is defined for this amino acid
                for chi_definition in ['chi1_definition', 'chi2_definition', 'chi3_definition', 'chi4_definition']:
                    if chi_definition in dictionary:
                        # search in every line for the AAresid and the four atom names forming the dihedral angles
                        # (defined in the dictionary) of the mutated amino acid
                        index = [self.get_index('-'.join([self.segname,
                                                          dictionary["abbreviation"], dictionary["curresid"],
                                                          dictionary[chi_definition][0]]), records='updated'),
                                 self.get_index('-'.join([self.segname,
                                                          dictionary["abbreviation"], dictionary["curresid"],
                                                          dictionary[chi_definition][1]]), records='updated'),
                                 self.get_index('-'.join([self.segname,
                                                          dictionary["abbreviation"], dictionary["curresid"],
                                                          dictionary[chi_definition][2]]), records='updated'),
                                 self.get_index('-'.join([self.segname,
                                                          dictionary["abbreviation"], dictionary["curresid"],
                                                          dictionary[chi_definition][3]]), records='updated')]
                        list_atom_indices.append(index)
            if self.constrained:
                with open(os.path.join(self.directory, 'dihedrals.dat'), 'a', encoding=encoding) as dihedrals_dat:
                    # write file 'dihedrals.dat' for extrabonds in namd
                    # (one line: four atom indices, force constant, constrained dihedral angle)
                    for i in np.arange(len(list_dihedral_angles)):
                        dihedrals_dat.write('{0:8s} {1:5d} {2:5d} {3:5d} {4:5d} {5:3d} {6:5.1f}\n'.format('dihedral',
                                            list_atom_indices[i][0], list_atom_indices[i][1], list_atom_indices[i][2],
                                            list_atom_indices[i][3], dihedral_force, list_dihedral_angles[i]))
            else:
                with open(os.path.join(self.directory, 'dihedrals.dat'), 'w', encoding=encoding) as dihedrals_dat:
                    for i in np.arange(len(list_dihedral_angles)):
                        dihedrals_dat.write('{0:8s} {1:5d} {2:5d} {3:5d} {4:5d} {5:3d} {6:5.1f}\n'.format('dihedral',
                                            list_atom_indices[i][0], list_atom_indices[i][1], list_atom_indices[i][2],
                                            list_atom_indices[i][3], dihedral_force, list_dihedral_angles[i]))

        outcons = os.path.join(self.directory, 'restraint.out')
        # if self.solvent == 'water' or self.solvent == 'membrane':
        #    self.write_namd_place_restraint_conf()
        self.write_namd_submit(self.files['namd_place_restraint'], outcons)
        self.files['place_restraint_out'] = outcons
        # start process and wait for it to finish
        p = subprocess.Popen(self.files['namd_submit'])
        return p

    def check_log(self, total_mutations):
        """
        Checks the namd log file generated during the placement of the mutated side chain(s) for errors.
        :param total_mutations: The current mutation cycle
        :type total_mutations: int
        :return: Whether an error occurred during placement of the new side chain(s) or not
        :rtype: bool
        """

        with open(self.files['place_restraint_out'], 'r', encoding=encoding) as check_namd_log:
            checkerr = check_namd_log.read()
        if 'ERROR' in checkerr:
            self.write_log(cycle=total_mutations, acceptance=False, reason='ERROR')
            shutil.copy(self.files['place_restraint_out'],
                        os.path.join(self.dirs['crash_dir'], "{}_crash.log".format(total_mutations)))
            return True
        else:
            return False

    def equilibrate_mutation(self):
        """
        Starts a equilibration + minimisation of the Snap after placement of the mutated side chain(s) as a subprocess.
        Uses the .conf generated by write_namd_free_conf and the results of place_sidechain as input.
        :return: The subprocess in which the namd simulation was started
        :rtype: subprocess
        """

        outfree = os.path.join(self.directory, 'free.out')
        self.write_namd_submit(self.files['namd_place_free'], outfree)
        self.files['place_free_out'] = outfree
        p = subprocess.Popen(self.files['namd_submit'])
        return p

    def evaluate_mutation(self, temperature):
        """
        Evaluates the Metropolis-Monte-Carlo-criterion for the change in either an nonbonded energy of
        the Snap structure. Returns the product of the Boltzmann factor and the weight if the MMC criterion suggests
        declining the change or 0 if the change would be accepted.
        :param temperature: Temperature in the Boltzmann factor
        :type temperature: float
        :return: Contribution of this mutation to the total acceptance score for all Snaps
        :rtype: float
        """
        vmd_cmd = "vmd -dispdev text -e {mini} -args {pdb_in} {coor_in} {pdb_out}"
        self.run_vmd_command(vmd_cmd.format(mini=self.files['coor2pdbscript'],
                                            pdb_in=self.files['pdb_mutated'],
                                            coor_in=os.path.join(self.directory, 'place_free.coor'),
                                            pdb_out=self.files['pdb_sp']))
        self.write_namd_sp_conf(stage='mutation')
        self.run_namd_sp()
        self._ref_vdw = self._current_vdw
        self._ref_electrostatics = self._current_electrostatics
        self._ref_nonbonded = self._current_nonbonded
        if self.external:
            self._ref_ext = self._current_ext
            self._ref_ext_out = self._current_ext_out
        self.get_attributes()
        change = 0
        if self.mode == 'elec':
            change = self._current_electrostatics - self._ref_electrostatics
        elif self.mode == 'vdw':
            change = self._current_vdw - self._ref_vdw
        elif self.mode == 'nonbonded':
            change = self._current_nonbonded - self._ref_nonbonded
        elif self.mode == 'external' and self._current_ext != np.inf:
            change = self._current_ext - self._ref_ext
        elif self.mode == 'external' and self._current_ext == np.inf:
            return np.inf

        boltzmann = np.exp(-float(change)/(8.3144598*float(temperature)))
        if change < self.threshold:
            self.write2output('INFO: {0:10} has just been accepted with a change of {1:7.4f}!\n'
                              .format(self.name, change))
            return self.weight * change
        elif change > -1 * self.threshold and boltzmann > random.random():
            self.write2output('INFO: {0:10} has just tunnelled with a change of {1:7.4f}!'
                              .format(self.name, change))
            return self.weight * self.threshold
        else:
            self.write2output('INFO: {0:10} has just been declined with a change of {1:7.4f}!\n'
                              .format(self.name, change))
            return self.weight * change

    def process_score(self, accepted, total_mutations):
        """
        This function processes the result of the MC evaluation. If the mutation is rejected, the energies are reset
        to the pre-mutation stage. If the mutation is accepted, the pdb and psf files are set to the mutated stage
        and copied to the 'Accepted' directory. In any case the log for the Snap is written and the pdb and psf
        copied to the 'PDB' directory.
        :param accepted: Flag indicating, whether the mutation was accepted or not
        :type accepted: bool
        :param total_mutations: The current mutation cycle
        :type total_mutations: int
        """

        shutil.copy(self.files['pdb_sp'], os.path.join(self.dirs['pdb_dir'], str(total_mutations) + ".pdb"))
        shutil.copy(self.files['psf_sp'], os.path.join(self.dirs['pdb_dir'], str(total_mutations) + ".psf"))
        self.get_fasta(os.path.join(self.dirs['pdb_dir'], str(total_mutations) + ".fasta"), records='updated')
        if not accepted:
            if not (self.external and self._current_ext == np.inf):
                self.write_log(cycle=total_mutations, acceptance=False, reason='score diminished')
            else:
                self.write_log(cycle=total_mutations, acceptance=False, reason='ERROR in external')
            self._current_vdw = self._ref_vdw
            self._current_electrostatics = self._ref_electrostatics
            self._current_nonbonded = self._ref_nonbonded
            if self.external:
                self._current_ext = self._ref_ext
                self._current_ext_out = self._ref_ext_out
        else:
            self.write_log(cycle=total_mutations, acceptance=True, reason='score improved')
            # I think the problem is here
            shutil.copy(self.files['psf_sp'], self.files['psf'])
            shutil.copy(self.files['pdb_sp'], self.files['pdb'])
            self.records = self.read_pdb()
            self.get_fasta(os.path.join(self.dirs['acc_dir'], str(total_mutations) + ".fasta"))
            # safe improved structure and coordinates
            shutil.copy(self.files['pdb'], os.path.join(self.dirs['acc_dir'], str(total_mutations) + ".pdb"))
            shutil.copy(self.files['psf'], os.path.join(self.dirs['acc_dir'], str(total_mutations) + ".psf"))
