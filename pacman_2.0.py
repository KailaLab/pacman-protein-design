#!/usr/bin/env python
import os
import sys
import numpy as np
from copy import deepcopy
import argparse
import random
import bisect
from Snap import Snap
from Parser import Parser
from Plugin import Plugin


# defines encoding for file i/o
encoding = 'utf-8'
pwd = os.getcwd()


def write2output(st):
    """flushing the stdout should keep the output in order, even on highly buffered systems."""

    sys.stdout.write(st)
    sys.stdout.flush()


def mutate_residues(window, num_mutations):
    """
    This function makes a number of random mutation on a given amino acid sequence.
    :param window: An arbitrary snapshot of a structure. Used to access sequence information
    :type window: class Snap
    :param num_mutations: The number of mutations to be performed on the amino acid sequence
    :type num_mutations: int
    :return: The lists of mutated residue ids, new amino acids at these positions and former amino acids there
    :rtype: list(int), list(str), list(str)
    """

    aminos = main_config['aa_set']
    # initiate lists
    list_newnames = []
    list_resids = np.random.choice(window.mutable_resids, num_mutations, replace=True)
    list_current = [window.get_name_from_resid(resid) for resid in list_resids]
    for x in range(num_mutations):
        aminos_copy = deepcopy(aminos)
        if list_current[x] in aminos_copy:
            aminos_copy.remove(list_current[x])
        list_newnames.append(random.choice(aminos_copy))

    # don't mutate NTER to prolin
    for x in range(num_mutations):
        while list_resids[x] == window.mutable_resids[0] and list_newnames[x] == 'PRO':
            aminos_copy = deepcopy(aminos)
            if list_current[x] in aminos_copy:
                aminos_copy.remove(list_current[x])
            list_newnames[x] = random.choice(aminos_copy)
    return list_resids, list_current, list_newnames


def write_master(header=False, cycle=0, mc_score=0, ids=(), old=(), nms=()):
    """
    This function writes the structure wise log after a mutation was accepted. It either generates the file 'master.log'
    and writes column headers if run in header mode or appends to said file mutation results. (attempt, resid x,
    resname x, mutation x, MC score)
    :param header: Flag deciding whether this instance should write file headers or content
    :type header: Boolean
    :param cycle: Mutation cycle in which the output line was generated
    :type cycle: Int
    :param mc_score: Score resulting from linear weighting of the MC results of all snaps
    :type mc_score: float
    :param ids: residue numbers at which mutations where performed
    :type ids: list(int)
    :param old: original aa at these positions
    :type: old: list(str)
    :param nms: new aa at these positions
    :type nms: list(str)
    """

    if header:
        write2output("INFO: Initiating master.log.\n")
        template = '{0:7s}   {1:>8s} {2:>10s} {3:>11s}   '
        template += '{4:>8s} {5:>10s} {6:>11s}   {7:>8s} {8:>10s} {9:>11s} {10:<10s}\n'
        values = ['attempt',
                  'resid 1', 'resname 1', 'mutation 1',
                  'resid 2', 'resname 2', 'mutation 2',
                  'resid 3', 'resname 3', 'mutation 3',
                  'score']
        with open(os.path.join(pwd, "master.log"), 'w', encoding=encoding) as outfile:
            outfile.write(template.format(*values))
    else:
        write2output("INFO: Writing master.log.\n")
        template = '{0:7d}   {1:>8} {2:>10s} {3:>11s}   '
        template += '{4:>8} {5:>10s} {6:>11s}   {7:>8} {8:>10s} {9:>11s} {10:<10.2f}\n'
        values = [cycle]
        values += ' '.join(['{} {} {}'.format(r, rn, mut) for r, rn, mut in zip(ids,
                                                                                old,
                                                                                nms)]).split()
        values += [float(mc_score)]
        with open(os.path.join(pwd, "master.log"), 'a', encoding=encoding) as outfile:
            outfile.write(template.format(*values))


if __name__ == '__main__':

    write2output("\nINFO: Parsing input files\n")
    parser = argparse.ArgumentParser(description='Monte Carlo Protein design program. Possible arguments:' +
                                     '--dir inputfile\n')
    parser.add_argument('--input', dest='inputfile',  required=False, default='input.dat', type=str)
    args = parser.parse_args()
    main_config = Parser(args.inputfile).parameters

    if main_config['MC_criterion'] == 'external':
        plugins = [Plugin(i[0], i[1]) for i in main_config['plugin']]
    else:
        plugins = []

    # if 'conformation' in main_config.parameters:
    if 'conformation' in main_config.keys():
        snaps = [Snap(config['prefix'],
                      config['pdbname'],
                      config['psfname'],
                      int(main_config['cpus'] / len(main_config['conformation'])),
                      config['weight'],
                      main_config['enviroment'],
                      main_config['solvent'],
                      main_config['MC_criterion'],
                      main_config['threshold'],
                      main_config['logfile'],
                      main_config['immutables'],
                      config['enzyme_resids_list'],
                      config['prms'],
                      config['rtfs'],
                      main_config['segname'],
                      config['constraint_pairs'],
                      config['constraint_forces'],
                      config['constraint_lengths'],

                      plugins,
                      main_config['fixed_sidechains'],
                      main_config['max_charge_change']
                      )
                 for config in main_config['conf_par']]

    else:
        conformation1 = Snap(main_config['prefix'],
                             main_config['pdbname'],
                             main_config['psfname'],
                             main_config['cpus'],
                             1.0,
                             main_config['enviroment'],
                             main_config['solvent'],
                             main_config['MC_criterion'],
                             main_config['threshold'],
                             main_config['logfile'],
                             main_config['immutables'],
                             main_config['enzyme_resids_list'],
                             main_config['prms'],
                             main_config['rtfs'],
                             main_config['segname'],
                             main_config['constraint_pairs'],
                             main_config['constraint_forces'],
                             main_config['constraint_lengths'],
                             plugins,
                             main_config['fixed_sidechains'],
                             main_config['max_charge_change']
                             )
        snaps = [conformation1]

    # initialise master output file
    write_master(header=True)
    # minimise and equilibrate starting structure of each snapshot and calculate initial properties
    write2output("\nINFO: Minimising initial structure(s) and applying constraints if specified\n")
    for snap in snaps:
        snap.namd_minimize(main_config['minimize'])

    # initialise mutation protocol
    total_mutations = 0
    accepted_mutations = 0
    list_moves = [1, 2, 3]
    list_cumsum_moves = np.cumsum([main_config['p_single'], main_config['p_double'], main_config['p_triple']])
    totstop = False



    # start mutation cycle
    while total_mutations < main_config['cycles'] and accepted_mutations < main_config['max_mutations']:
        write2output("\nINFO: Starting mutation cycle {:d}\n".format(total_mutations+1))
        random_number_move = random.random()
        index_moves = bisect.bisect(list_cumsum_moves, random_number_move)
        n_mutations = list_moves[index_moves]
        # generate mutated sequence (is the same for each snapshot)
        list_AAresids, list_current_AAs, list_AAresnames = mutate_residues(snaps[0], n_mutations)

        mutations = ' and '.join(['resid {} from {} to {}'.format(r, rn, mut) for r, rn, mut in zip(list_AAresids,
                                                                                                    list_current_AAs,
                                                                                                    list_AAresnames)])
        write2output("INFO: The following mutations will be performed: "+str(mutations)+"\n")

        processes = []
        # do mutation for each snapshot
        for snap in snaps:
            write2output("INFO: Mutating snap {}\n".format(snap.name))
            totstop = snap.mutate(list_AAresids, list_AAresnames, list_current_AAs)
            if totstop:
                break
            write2output("INFO: Placing new side chain for snapshot {}\n".format(snap.name))
            processes.append(snap.place_sidechain(main_config['dihedral_force']))
        if totstop:
            totstop = False
            for p in processes:
                p.terminate()
            write2output("\nINFO: Cycle aborted because of too high a change in total charge.\n")
            continue

        done = [p.wait() for p in processes]
        write2output("INFO: Finished placement of new side chains for all snapshots\n")
        total_mutations += 1
        for i in range(3):
            if i >= len(list_AAresids):
                list_current_AAs.append('-')
                list_AAresids = np.append(list_AAresids, [0])
                list_AAresnames.append('-')
        # check for ERRORS in logfiles --> discard
        for snap in snaps:
            write2output("INFO: Checking for errors during placement of side chain for snap {}\n".format(snap.name))
            totstop = snap.check_log(total_mutations)
            if totstop:
                break
        # start next cycle if there is a reason for discarding this one
        if totstop:
            totstop = False
            for p in processes:
                p.terminate()
            write2output("\nINFO: Cycle aborted because of crashes during the placement of the new side chain.\n")
            continue
        processes = []
        # if placing the side chain was successful, equilibrate the mutated structure
        for snap in snaps:
            write2output("INFO: Equilibrating mutated structure for snapshot {}\n".format(snap.name))
            processes.append(snap.equilibrate_mutation())
        done = [p.wait() for p in processes]
        write2output("INFO: Finished equilibration of mutated structure for all snapshots\n")

        score = 0
        # calculate a single point, extract the energies from the output and calculate the acceptance score
        for snap in snaps:
            write2output("INFO: Evaluating Metropolis-Monte-Carlo criterion for snapshot {}\n".format(snap.name))
            score += snap.evaluate_mutation(main_config['MC_temperature'])
        # accept the new structure if the score is below a threshold (right now arbitrary)
        if score > main_config['threshold']:
            write2output("\nINFO: *** Mutation attempt #{0:7d} rejected ***\n".format(total_mutations))
            write2output("INFO: *** with a score of {0:7.4f} ***\n".format(score))
            for snap in snaps:
                snap.process_score(False, total_mutations)
        else:
            write2output("\nINFO: *** Mutation attempt #{0:7d} accepted ***\n".format(total_mutations))
            write2output("INFO: *** with a score of {0:7.4f} ***\n".format(score))
            accepted_mutations += n_mutations
            for snap in snaps:
                snap.process_score(True, total_mutations)
                # note successful mutation in the master logfile
            write_master(header=False, cycle=total_mutations, mc_score=score, ids=list_AAresids,
                         old=list_current_AAs, nms=list_AAresnames)
    # end of the mutation cycle
    with open(os.path.join(pwd, "master.log"), 'a', encoding=encoding) as master_out:
        master_out.write("\nDONE\n")

    write2output("\nINFO: End of program. Enjoy your optimised structure(s)!\n")
