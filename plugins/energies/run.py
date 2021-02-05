import os
import sys

encoding = 'utf-8'
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
debug = False


def write_get_energies(snap, mode, selection, selection_2):
    # read the template
    with open(os.path.join(template_dir, '_energies_'+snap.solvent+'_'+mode+'.tmp'),
              'r',
              encoding=encoding) as f:
        template = f.read()

    # write the tcl script with all additions
    script = os.path.join(snap.directory, "energies.tcl")
    parameters = ' -par '.join(snap.parameterfiles)
    outfile = os.path.join(snap.directory, 'energies.dat')
    subscript = os.path.join(snap.directory, "submit_namd.job")
    conf = os.path.join(snap.directory, "namd-temp.namd")
    out = os.path.join(snap.directory, "namd-temp.log")
    snap.write_namd_submit(conf, out, 1)
    if mode == 'intra':
        with open(script, 'w', encoding=encoding) as f:
            f.write(template.format(psf=snap.files['psf_sp'],
                                    pdb=snap.files['pdb_sp'],
                                    selection='"'+selection+'"',
                                    outfile=outfile,
                                    xsc=snap.files['xscfile'],
                                    prms=parameters,
                                    exe=subscript))
    else:

        with open(script, 'w', encoding=encoding) as f:
            f.write(template.format(psf=snap.files['psf_sp'],
                                    pdb=snap.files['pdb_sp'],
                                    selection='"'+selection+'"',
                                    selection_2='"'+selection_2+'"',
                                    outfile=outfile,
                                    xsc=snap.files['xscfile'],
                                    prms=parameters,
                                    exe=subscript))
    return script, outfile


def grep_energies(infile, mode):
    with open(infile, 'r', encoding=encoding) as log:
        log.readline()
        energy_line = log.readline()
    columns = energy_line.split()
    if mode == 'inter':
        electrostatics_j_mol = 4184.0 * float(columns[2])
        vdw_j_mol = 4184.0 * float(columns[3])
        nonbonded_j_mol = 4184.0 * float(columns[4])
        total_j_mol = 4184.0 * float(columns[5])
    else:
        electrostatics_j_mol = 4184.0 * float(columns[6])
        vdw_j_mol = 4184.0 * float(columns[7])
        nonbonded_j_mol = 4184.0 * float(columns[9])
        total_j_mol = 4184.0 * float(columns[10])
    return nonbonded_j_mol, electrostatics_j_mol, vdw_j_mol, total_j_mol


def run(snap, plugin_args):
    mode = plugin_args['mode']
    sel1 = plugin_args['selection']
    sel2 = plugin_args['selection_2']
    energy = plugin_args['energy']
    os.chdir(snap.directory)
    vmd_script, energyfile = write_get_energies(snap, mode, sel1, sel2)
    command = "vmd -dispdev text -e {}".format(vmd_script)
    snap.run_vmd_command(command, debug=debug)
    nonbonded, elec, vdw, total = grep_energies(energyfile, mode)
    if energy == 'vdw':
        sys.stdout.write("INFO: PLUGIN: This selection has a vdw energy of {0:.2f} kJ/mol\n".format(vdw/1000))
        sys.stdout.flush()
        return vdw, vdw/4184
    elif energy == 'elec':
        sys.stdout.write("INFO: PLUGIN: Selection has a electrostatic energy of {0:.2f} kJ/mol\n".format(elec/1000))
        sys.stdout.flush()
        return elec, elec/4184
    elif energy == 'nonbonded':
        sys.stdout.write("INFO: PLUGIN: Selection has a nonbonded energy of {0:.2f} kJ/mol\n".format(nonbonded/1000))
        sys.stdout.flush()
        return nonbonded, nonbonded/4184
    elif energy == 'total':
        sys.stdout.write("INFO: PLUGIN: Selection has a total energy of {0:.2f} kJ/mol\n".format(total/1000))
        sys.stdout.flush()
        return total, total/4184
    else:
        sys.exit("ERROR: PLUGIN: How the hell did this happen?")
