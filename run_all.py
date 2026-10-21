"""
Run all notebooks of the thermal springs analysis in the correct order

Usage, from the main directory of this repository:

    python run_all.py              run all notebooks, executed copies are saved in the directory executed_notebooks
    python run_all.py --inplace    run all notebooks and save the results in the notebooks themselves
    python run_all.py calculate_heat_flux make_figures    run only these notebooks

The script stops at the first notebook that fails.
"""

import argparse
import os
import subprocess
import sys
import time

# notebooks in the order in which they need to be run, each notebook uses the results of the notebooks above it
notebooks = ['GIS_analysis_spring_data',
             'calculate_heat_flux',
             'make_figures',
             'map_thermal_springs',
             'background_heat_flow_Alps',
             'background_heat_flow_NA',
             'compare_orogens_v2']

# directory for the executed copies of the notebooks
output_dir = 'executed_notebooks'

# maximum run time of a single notebook cell (s)
cell_timeout = 3600


def run_notebook(notebook, inplace=False):
    """
    Execute a notebook with nbconvert, returns True if the notebook ran without errors
    """

    command = [sys.executable, '-m', 'nbconvert', '--to', 'notebook', '--execute',
               '--ExecutePreprocessor.timeout=%i' % cell_timeout]
    if inplace is True:
        command += ['--inplace']
    else:
        command += ['--output-dir', output_dir]
    command += [notebook + '.ipynb']

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr[-3000:])

    return result.returncode == 0


def main():

    parser = argparse.ArgumentParser(description='Run the notebooks of the thermal springs analysis')
    parser.add_argument('notebooks', nargs='*', default=notebooks,
                        help='names of the notebooks to run, without .ipynb (default: all notebooks)')
    parser.add_argument('--inplace', action='store_true',
                        help='save the results in the notebooks instead of in the directory %s' % output_dir)
    args = parser.parse_args()

    # the notebooks use paths relative to the main directory of the repository
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # directory for the figures
    os.makedirs('fig', exist_ok=True)

    for notebook in args.notebooks:
        notebook = notebook.replace('.ipynb', '')
        print('running %s.ipynb' % notebook, end=' ', flush=True)
        start = time.time()

        if run_notebook(notebook, inplace=args.inplace) is False:
            print('\nerror, %s.ipynb failed, see the error message above' % notebook)
            sys.exit(1)

        print('done in %0.0f s' % (time.time() - start))

    print('all notebooks finished successfully')


if __name__ == '__main__':
    main()
