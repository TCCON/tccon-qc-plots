from argparse import ArgumentParser
from pathlib import Path
import tarfile
import toml  # alternate toml library, tomli is read only

_DEFAULT_TOML_FILE = Path(__file__).resolve().parent.parent / 'inputs' / 'site_ancillary_data.toml'


def parse_level_from_tarball(tar_path, level=0):
    with tarfile.open(tar_path) as tf:
        f = tf.extractfile(tf.firstmember)
        nhead = int(f.readline().split()[0])
        for _ in range(1, nhead):
            line = f.readline()
            
        header = line.decode().split()
        for _ in range(level+1):
            line = f.readline()
        data = [float(el) for el in line.split()]
        return {k: v for k, v in zip(header, data)}


def get_level_values_for_sites(std_site_root, level=0):
    site_dirs = sorted(Path(std_site_root).glob('??'))
    site_data = dict()
    for site_dir in site_dirs:
        site_id = site_dir.name
        first_tarball = sorted(site_dir.glob('*.tgz'))[0]
        print(f'Reading {site_id} data from {first_tarball}')
        level_data = parse_level_from_tarball(first_tarball, level=level)
        site_data[site_id] = level_data
    return site_data


def write_site_data_toml(std_site_root, level=0, toml_file=_DEFAULT_TOML_FILE, variables=('Height',)):
    site_data = get_level_values_for_sites(std_site_root, level=level)
    level_key = f'level{level}'
    output_data = {
        'site_data': {k: {'mod_data': {level_key: dict()}} for k in site_data}
    }
    for site_id, data in site_data.items():
        output_data['site_data'][site_id]['mod_data'][level_key] = {k: data[k] for k in variables}

    print(f'Writing to {toml_file}')
    with open(toml_file, 'w') as f:
        toml.dump(output_data, f)


def main():
    p = ArgumentParser('Update the ancillary site data file')
    p.add_argument('std_site_root', type=Path, help='Root directory of the standard site tarballs (has site directories in it)')
    p.add_argument('--toml-file', type=Path, default=str(_DEFAULT_TOML_FILE), help='Location to write the .toml file. Default = %(default)s')
    clargs = vars(p.parse_args())

    write_site_data_toml(**clargs)


if __name__ == '__main__':
    main()
