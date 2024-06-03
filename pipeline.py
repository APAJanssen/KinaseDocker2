# Pipeline imports
import argparse
import itertools
import os
import shutil
import time
import zlib

import docker
import pandas as pd
# from pymol import cmd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Extra VinaGPU imports
import datetime
import re
import subprocess as sp

from meeko import MoleculePreparation
from meeko import PDBQTMolecule
from rdkit.Chem import AllChem


'''
Pipeline
'''

class Pipeline:
    '''
    Pipeline class that runs the docking and scoring.
    '''
    def __init__(self, run_name, smiles_list, kinase_families, accessions, docking_software, scoring_algorithm, output_path):
        # Store variables
        self.run_name = run_name if not os.path.exists(os.path.join(output_path, run_name, 'results', f'{run_name}_{docking_software}_results.csv')) else run_name + '_copy' # Prevent overwriting existing results
        self.smiles_list = smiles_list
        self.kinase_families = kinase_families
        self.accessions = accessions
        self.docking_software = docking_software
        self.scoring_algorithm = scoring_algorithm

        # Setup paths
        self.output_path = os.path.abspath(os.path.join(output_path, self.run_name))
        self.pdb_path = os.path.join(self.output_path, 'pdb')
        self.docking_path = os.path.join(self.output_path, 'intermediate_input_' + self.docking_software)
        self.model_path = os.path.join(self.output_path, 'intermediate_input_' + self.scoring_algorithm)
        self.results_path = os.path.join(self.output_path, 'docking_results')
        self.setup_folders()

        # Load the kinase data and retrieve the KLIFS structures
        self.kin_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'kinase_data.csv'))
        self.structures = self.get_structures()

        # Setup the base docker container
        dev_req = docker.types.DeviceRequest
        self.container = None
        self.client = docker.from_env()

        self.docker_kwargs = dict(image='apajanssen/kinasedocker2',
                                  device_requests=[dev_req(device_ids=['0'], capabilities=[['gpu']])])

    def run(self):
        '''
        Start the pipeline.
        '''
        # Store processing and docking function based on the docking software
        if self.docking_software == 'vina':
            preprocess = self.preprocessing_vina
            dock = self.dock_vina
        else: # diffdock
            preprocess = self.preprocessing_diffdock
            dock = self.dock_diffdock

        self.start_time = time.time() # Start timer

        print(f'[{self.get_current_runtime()}] Preprocessing...')
        preprocess() # Preprocess the structures and smiles to an input file for the docking software
        dock() # Dock everything with the respective docking software
        self.preprocessing_dnn() # Preprocess the docking results to an input file for the DNN
        self.run_dnn() # Run the DNN on the docking results
        self.postprocess_results() # Postprocess the results
        self.cleanup() # Cleanup the output folders
        print(f'[{self.get_current_runtime()}] Pipeline finished!')

    def start_docker_container(self, docker_kwargs): # Default with basic docker_kwargs
        '''
        Start the docker container.
        '''
        container = self.client.containers.run(
            command='sleep infinity', # Keeps the container running until it is killed
            detach=True,              # Run container in background
            **docker_kwargs)
        
        return container

    def remove_docker_container(self):
        """
        Stop Vina-GPU docker container
        """
        self.container.remove(force=True) 
        self.container = None

    def dock_vina(self, threads=8192, search_depth=10):
        '''
        Dock the structures with vina.

        Threads: number of threads to use for docking (8192 was found to be optimal)
        Search depth: Algorithm search depth (10 was found to be optimal)
        '''
        print(f'[{self.get_current_runtime()}] Docking with Vina...')
        print('-'*50)

        # Load input data
        input_data = pd.read_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_input.csv'))

        # Create the Vina runner
        vina_gpu = VinaGPU(out_path=self.docking_path)

        # Loop over the structures and dock all associated compounds
        for i, pdb in enumerate(input_data['klifs_ID'].unique(), 1):
            # Get the input data for the current pdb
            smiles = input_data[input_data['klifs_ID'] == pdb]['smiles'].tolist()
            box_center = self.kin_data[self.kin_data['klifs_ID'] == pdb][['box_center_x', 'box_center_y', 'box_center_z']].values[0]
            box_size = self.kin_data[self.kin_data['klifs_ID'] == pdb][['box_size_x', 'box_size_y', 'box_size_z']].values[0]

            print(f'[{self.get_current_runtime()}] [VINA] Docking {len(self.smiles_list)} compound(s) in {pdb} ({i}/{len(input_data["klifs_ID"].unique())})')

            # Dock the current pdb
            vina_gpu.dock(target_pdb_path=os.path.join(self.pdb_path, f'{pdb}.pdb'),
                          smiles=smiles,
                          output_subfolder=str(pdb), 
                          box_center=box_center, 
                          box_size=box_size, 
                          threads=threads, 
                          threads_per_call=threads,
                          num_modes=3, # num poses
                          search_depth=search_depth)

        self.postprocess_vina_output()

        print('-'*50)

    def dock_diffdock(self):
        '''
        Dock the structures with diffdock.
        '''
        print(f'[{self.get_current_runtime()}] Docking with DiffDock...')
        print('-'*50)

        # Setup folder links in docker container
        docker_kwargs = self.docker_kwargs.copy()
        docker_kwargs['volumes'] = {self.docking_path: {'bind': '/diffdock/input/data', 'mode': 'rw'},
                                    os.path.join(self.docking_path, 'output'): {'bind': '/diffdock/results', 'mode': 'rw'}}

        # Start the docker container
        self.container = self.start_docker_container(docker_kwargs)

        # Loop over the structures and dock all associated compounds
        try:
            for i, klifs_id in enumerate(self.structures, 1):
                print(f'[{self.get_current_runtime()}] [DIFFDOCK] Docking {len(self.smiles_list)} compound(s) in {klifs_id} ({i}/{len(self.structures)})')

                cmd = f'python3 -m inference_JS --protein_ligand_csv input/data/{self.run_name}_{self.docking_software}_input_{klifs_id}.csv --inference_steps 10 --samples_per_complex 3 --batch_size 10 --actual_steps 10 --no_final_step_noise'

                _, (stdout, stderr) = self.container.exec_run(cmd=cmd, workdir='/diffdock', demux=True)
        except Exception as e:
            print(f'[{self.get_current_runtime()}] Error has occurred while docking: {e}')
            raise e
        except KeyboardInterrupt:
            print(f'[{self.get_current_runtime()}] Docking interrupted by user')
        finally:
            self.remove_docker_container()
            self.postprocess_diffdock_output()

        print('-'*50)
            
    def run_dnn(self):
        '''
        Run the DNN on the docking results.
        '''
        print(f'[{self.get_current_runtime()}] Running DNN...')

        # Setup folder links in docker container
        docker_kwargs = self.docker_kwargs.copy()
        docker_kwargs['volumes'] = {self.model_path: {'bind': '/DNN/DNN_data/input', 'mode': 'rw'},
                                    os.path.join(self.model_path, 'output'): {'bind': '/DNN/results', 'mode': 'rw'}}

        # Start the docker container
        self.container = self.start_docker_container(docker_kwargs)

        # Run the DNN
        try:
            cmd = f'python3 DNN_eval.py --input_file {self.run_name}_DNN_input.csv --docking_type {self.docking_software}'

            _, (stdout, stderr) = self.container.exec_run(cmd=cmd, workdir='/DNN', demux=True)

            print(f'[{self.get_current_runtime()}] Determine clashing...')

            cmd = f'python3 clashing.py --input_file {self.run_name}_DNN_input.csv'

            _, (stdout, stderr) = self.container.exec_run(cmd=cmd, workdir='/DNN', demux=True)
        except Exception as e:
            print(f'[{self.get_current_runtime()}] Error has occurred while running DNN: {e}')
            raise e
        except KeyboardInterrupt:
            print(f'[{self.get_current_runtime()}] DNN interrupted by user')
        finally:
            self.remove_docker_container()

    def postprocess_vina_output(self):
        '''
        postprocess the vina output by concatenating everything into one file.
        '''
        final_data = pd.DataFrame()
        folders = os.listdir(os.path.join(self.docking_path, 'output')) # Process everything in the output folder, may cause issues when previous runs are not cleaned up properly

        if len(folders) == 0:
            raise Exception('No docking results found, something went wrong!')

        # Extract the log.tsv files from the output folder tree and append it to the final dataframe
        for folder in folders:
            folder_path = os.path.join(self.docking_path, 'output', folder)

            if os.path.isfile(os.path.join(folder_path, 'log.tsv')):
                data = pd.read_csv(os.path.join(folder_path, 'log.tsv'), sep='\t')
                final_data = pd.concat([final_data, data], ignore_index=True)

        # Change some headers and extract the molblock from the compressed molfile, save the first line as the pose_ID
        final_data.rename(columns={'target': 'klifs_ID', 'score': 'vina_score', 'molfile': 'molblock'}, inplace=True)
        final_data['klifs_ID'] = final_data['klifs_ID'].astype(int)
        final_data['molblock'] = final_data['molblock'].apply(self.decompress)
        final_data['pose_ID'] = final_data['molblock'].apply(lambda x: x.split('\n')[0])

        final_data = final_data.reindex(columns=['pose_ID', 'klifs_ID', 'smiles', 'vina_score', 'molblock']) # Change Dataframe column order
        final_data.to_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_docking_results.csv'), index=False)

    def postprocess_diffdock_output(self):
        '''
        postprocess the diffdock output by concatenating everything into one file.
        '''
        final_data = pd.DataFrame()
        files = os.listdir(os.path.join(self.docking_path, 'output')) # Process everything in the output folder, may cause issues when previous runs are not cleaned up properly

        if len(files) == 0:
            raise Exception('No docking results found, something went wrong!')

        # Append the results from all files to the final dataframe
        for filename in files:
            file_path = os.path.join(self.docking_path, 'output', filename)

            data = pd.read_csv(file_path)
            final_data = pd.concat([final_data, data], ignore_index=True)

        # Change some headers and extract the molblock from the compressed molfile, save the first line as the pose_ID
        final_data['klifs_ID'] = final_data['klifs_ID'].astype(int)
        final_data.rename(columns={'SMILES_input': 'smiles_input', 
                                'SMILES_output': 'smiles_output', 
                                'molfile_compressed': 'molblock', 
                                'DiffDock_confidence': 'diffdock_confidence'}, inplace=True)
        final_data['molblock'] = final_data['molblock'].apply(self.decompress)
        final_data['pose_ID'] = final_data['molblock'].apply(lambda x: x.split('\n')[0])
        final_data = final_data.reindex(columns=['pose_ID', 'klifs_ID', 'smiles_input', 'smiles_output', 'diffdock_confidence', 'molblock']) # Change Dataframe column order
        final_data.to_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_docking_results.csv'), index=False)

    def postprocess_results(self):
        '''
        Postprocess the results by saving the results to a csv and SDF.
        '''
        docking_results = pd.read_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_docking_results.csv')) # Original docking results
        scoring_results = pd.read_csv(os.path.join(self.model_path, 'output', f'{self.run_name}_{self.scoring_algorithm}_input_results.csv')) # Model scoring results
        clashing_results = pd.read_csv(os.path.join(self.model_path, 'output', f'{self.run_name}_{self.scoring_algorithm}_input_clashing.csv')) # Clashing results

        docking_col = 'vina_score' if self.docking_software == 'vina' else 'diffdock_confidence' # Determine docking software specific column for dropping

        # Merge the results
        pose_results = pd.merge(docking_results, scoring_results, on='pose_ID', how='left')
        pose_results = pd.merge(pose_results, clashing_results, on='pose_ID', how='left')
        pose_results['Molecule'] = pose_results['molblock'].apply(Chem.MolFromMolBlock)
        pose_results['Kinase'] = pose_results['klifs_ID'].apply(lambda x: self.kin_data[self.kin_data['klifs_ID'] == x]['kinase'].values[0]) # Retrieves relevant kinase information based on Klifs_ID
        pose_results['accession'] = pose_results['klifs_ID'].apply(lambda x: self.kin_data[self.kin_data['klifs_ID'] == x]['accession'].values[0]) # Retrieves relevant kinase information based on Klifs_ID
        pose_results.drop(columns=['pose_ID', 'molblock', docking_col], inplace=True) # Drop unnecessary columns

        # Rename some columns
        if self.docking_software == 'diffdock':
            pose_results.rename(columns={'smiles_input': 'SMILES'}, inplace=True)
            pose_results.drop(columns=['smiles_output'], inplace=True)
        else:
            pose_results.rename(columns={'smiles': 'SMILES'}, inplace=True)

        pose_results = pose_results.reindex(columns=['SMILES', 'Kinase', 'accession', 'klifs_ID', 'pIC50', 'clash_score', 'Molecule']) # Change Dataframe column order
        pose_results['pIC50'] = pose_results['pIC50'].apply(lambda x: round(x, 2)) # Round pIC50 to 2 decimals

        # Save pose results to .SDF
        PandasTools.WriteSDF(pose_results, os.path.join(self.results_path, f'{self.run_name}_{self.docking_software}_results.sdf'), molColName='Molecule', idName='SMILES', properties=list(pose_results.columns))

        # Aggregate pose results
        agg_results = pose_results.groupby(['klifs_ID', 'SMILES']).agg({'Kinase': 'first', 'accession': 'first', 'pIC50': 'mean', 'clash_score': 'max'}).reset_index()

        agg_results.rename(columns={'pIC50': 'avg_score', 'clash_score': 'clash_score_max'}, inplace=True)
        agg_results = agg_results.reindex(columns=['SMILES', 'Kinase', 'accession', 'klifs_ID', 'avg_score', 'clash_score_max']) # Change Dataframe column order
        agg_results.to_csv(os.path.join(self.results_path, f'{self.run_name}_{self.docking_software}_results.csv'), index=False) # Save aggregated results to .csv

    def preprocessing_vina(self):
        '''
        Preprocess the structures and smiles to an input file for vina.

        The input file should be a csv with the following columns:
        - klifs_ID
        - smiles
        - box_center_x
        - box_center_y
        - box_center_z
        - box_size_x
        - box_size_y
        - box_size_z
        '''
        structure_smiles = itertools.product(self.structures, self.smiles_list) # Create all combinations of structures and smiles
        df = pd.DataFrame(structure_smiles, columns=['klifs_ID', 'smiles'])
        df.to_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_input.csv'), index=False)

    def preprocessing_diffdock(self):
        '''
        Preprocess the structures and smiles to an input file for diffdock. 
        Also save input file per KLIFS to enable a loop in the pipeline.

        The input file should be a csv with the following columns:
        - complex_name (smiles, since it will dock per klifs)
        - protein_path (path to pdb file: input/pdb/{klifs_ID}.pdb)
        - ligand_description (smiles)
        - protein_sequence (empty)
        '''
        structure_smiles = itertools.product(self.structures, self.smiles_list) # Create all combinations of structures and smiles
        df = pd.DataFrame(structure_smiles, columns=['klifs_ID', 'ligand_description'])
        df['complex_name'] = df['ligand_description']
        df['protein_path'] = df['klifs_ID'].apply(lambda x: f'input/pdb/{x}.pdb') # Create the path to the pdb file
        df['protein_sequence'] = '' # Empty protein sequence, but the input file needs it
        df = df.reindex(columns=['complex_name', 'protein_path', 'ligand_description', 'protein_sequence'])

        # Split the input file into chunks per klifs_ID
        for prot_path in df['protein_path'].unique():
            klifs_ID = prot_path.split('/')[-1].split('.')[0]
            df[df['protein_path'] == prot_path].to_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_input_{klifs_ID}.csv'), index=False)

        df.to_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_input.csv'), index=False)

    def preprocessing_dnn(self):
        '''
        Preprocess the structures and smiles to an input file for the DNN.
        Reads docking results, and extracts relevant information.
        '''
        docking_data = pd.read_csv(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_docking_results.csv'))

        dnn_input = docking_data.reindex(columns=['pose_ID', 'klifs_ID', 'molblock'])
        dnn_input['klifs_ID'] = dnn_input['klifs_ID'].astype(int)
        dnn_input.to_csv(os.path.join(self.model_path, f'{self.run_name}_DNN_input.csv'), index=False)

    def get_structures(self):
        '''
        Get all KLIFS structures for the given kinase families and individual accessions.
        '''
        structures = []

        for family in self.kinase_families:
            structures.extend(self.kin_data[self.kin_data['kinasegroup'] == family]['klifs_ID'].tolist())

        if self.accessions:
            structures.extend(self.kin_data[self.kin_data['accession'].isin(self.accessions)]['klifs_ID'].tolist())

        structures = list(set(structures)) # Remove potential duplicates from single accessions being in the same group as was in selection

        # Save the pdb files to the output_path/pdb folder
        for pdb in structures:
            compressed_pdb = self.kin_data[self.kin_data['klifs_ID'] == pdb]['pdb_compressed'].values[0]
            pdb_string = self.decompress(compressed_pdb)

            with open(os.path.join(self.pdb_path, f'{pdb}.pdb'), 'w') as f:
                f.write(pdb_string)

        return structures

    def setup_folders(self):
        '''
        Setup the folders for the pipeline.
        '''
        # Create output folder
        os.makedirs(self.output_path, exist_ok=True)

        # Create PDB folder
        os.makedirs(self.pdb_path, exist_ok=True)

        # Create Docking software folders
        os.makedirs(self.docking_path, exist_ok=True)
        os.makedirs(os.path.join(self.docking_path, 'output'), exist_ok=True)

        # Create scoring algorithm folders
        if self.scoring_algorithm == 'DNN':
            os.makedirs(self.model_path, exist_ok=True)
            os.makedirs(os.path.join(self.model_path, 'output'), exist_ok=True)

        # Create results folder
        os.makedirs(self.results_path, exist_ok=True)

    def cleanup(self):
        '''
        Cleanup the output folders.
        '''
        if self.docking_software == 'vina':
            shutil.rmtree(os.path.join(self.docking_path, 'output'))
        elif self.docking_software == 'diffdock':
            shutil.rmtree(os.path.join(self.docking_path, 'output'))

            # Remove the separate input files for diffdock
            for klifs_ID in self.structures:
                os.remove(os.path.join(self.docking_path, f'{self.run_name}_{self.docking_software}_input_{klifs_ID}.csv'))

        if self.scoring_algorithm == 'DNN':
            shutil.rmtree(os.path.join(self.model_path, 'output'))
            
    def decompress(self, compressed):
        '''
        Decompress a string.
        '''
        return zlib.decompress(bytes.fromhex(compressed)).decode('utf-8')
    
    def get_results_filepath(self):
        '''
        Get the filepath to the results file. Returns .sdf and .csv filepaths.
        '''
        return os.path.join(self.results_path, f'{self.run_name}_{self.docking_software}_results.sdf'), os.path.join(self.results_path, f'{self.run_name}_{self.docking_software}_results.csv')

    def get_current_runtime(self):
        '''
        Get the current runtime of the pipeline.
        '''
        return str(round(time.time() - self.start_time)) + ' s'

'''
VinaGPU
'''

class VinaGPU():
    """
    Class methods for running Vina-GPU docker container
    Also contains methods for preparing the ligand and target:
        - Ligand preparation via rdkit and meeko
        - Target preparation via ADFR Suite and pdb_tools
    """
    def __init__(self, docker_image_name='apajanssen/kinasedocker2', devices=['0'], out_path=None):
        self.device = 'gpu'
        self.device_id = devices

        self.out_path = os.path.join(out_path, 'output') if out_path is not None else os.path.join(os.getcwd(), 'output')

        # Setup ADFR suite for target preparation
        self.adfr_suite_docker_path = '/htd/ADFRsuite-1.0'

        # Setup meeko for ligand preparation
        self.molecule_preparation = MoleculePreparation(rigid_macrocycles=True)

        # Setup VinaGPU docker paths
        self.vina_dir = '/vina-gpu-dockerized/vina'
        self.docking_dir = self.vina_dir + '/docking'

        ## Configuration for running the Vina-GPU docker container 
        # (requires nvidia-docker runtime)
        dev_req = docker.types.DeviceRequest  # type: ignore
        self.container = None
        self.client = docker.from_env()

        self.docker_kwargs = dict(
            image=docker_image_name,
            volumes = [f'{self.out_path}:{self.docking_dir}'],
            device_requests=[dev_req(device_ids=devices, capabilities=[['gpu']])])
        
    def dock(self, target_pdb_path, smiles=[], output_subfolder='', 
             box_center=(0,0,0), box_size=(20,20,20), search_depth=3,
             threads=256, threads_per_call=256, num_modes=3, clean=True, verbose=False, # num_modes determines number of poses
             write_log=True, **kwargs):
        """
        Use Vina-GPU docker image to dock ligands (list of SMILES) to the target. 
        Produces a .pdbqt file for each ligand (with multiple docked orientations). 

        Parameters:
            target_pdb_path (str)                   : path to target pdb file
            smiles: (list(str))                     : list of smiles strings    
            output_subfolder (str), opt             : subfolder to save output files
            box_center (tuple(float)), opt          : coordinates of the active site of the target (x,y,z)=(0,0,0)
            box_size (tuple(float)), opt            : size of the bounding box around the active site (x,y,z)=(20,20,20)
            threads (int), opt                      : number of threads to use for docking
            thread_per_call (int), opt              : number of threads to use for each call to Vina
            num_modes (int), opt                    : number of poses to generate for each ligand
            clean (bool), opt                       : remove ligand .pdbqt files after docking
            verbose (bool), opt                     : print docking progress, scores, etc.
            write_log (bool), opt                   : write log file with docking results

        Returns:
            all_scores (list(list((float)))         : list of docking scores for each ligand
        """
        assert (len(smiles) > 0), "A list of smiles strings must be provided"

        results_path = os.path.join(self.out_path, output_subfolder)
        os.makedirs(results_path, exist_ok=True)

        # Prepare target .pdbqt file
        target_pdbqt_path = self.prepare_target(target_pdb_path, output_path=results_path)

        # Ensure no ligands from prior docking run linger (caused issues in loop)
        ligand_pdbqt_paths = []
        
        # Prepare ligand .pdbqt files
        print('Processing ligands...') if verbose else None
        for i, mol in enumerate(smiles): 
            ligand_pdbqt_path = os.path.join(results_path, f'ligand_{i}.pdbqt')
            out_path = self.prepare_ligand(mol, out_path=ligand_pdbqt_path)

            if out_path is not None:
                ligand_pdbqt_paths.append(ligand_pdbqt_path)

        basenames = [os.path.basename(p) for p in ligand_pdbqt_paths] # Ligand basenames (format 'ligand_0.pdbqt')
        basenames_docked = [lig.replace('.pdbqt', '_docked.pdbqt') for lig in basenames] # Docked ligand basenames (format 'ligand_0_docked.pdbqt')
        ligand_paths_docked = [os.path.join(results_path, p) for p in basenames_docked]
        
        ### Start Vina-GPU docker container and dock everything
        self.container = self.start_docker_container()

        try:
            timing, dates = [], []
            all_scores = [[0] for i in range(len(smiles))]
            target = os.path.basename(target_pdb_path).strip('.pdbqt')

            for i, ligand_file in enumerate(basenames):
                t0 = time.time()

                docking_args = dict(
                    receptor = f'docking/{output_subfolder}/{os.path.basename(target_pdbqt_path)}',
                    ligand   = f'docking/{output_subfolder}/{ligand_file}',
                    out      = f'docking/{output_subfolder}/{basenames_docked[i]}',
                    center_x = box_center[0],
                    center_y = box_center[1],
                    center_z = box_center[2],
                    size_x   = box_size[0],
                    size_y   = box_size[1],
                    size_z   = box_size[2],
                    thread   = threads,
                    search_depth = search_depth,
                    thread_per_call = threads_per_call,
                    num_modes = num_modes)

                cmd = './Vina-GPU ' + ' '.join([f'--{k} {v}' for k, v in docking_args.items()])

                try:
                    _, (stdout, stderr) = self.container.exec_run(
                        cmd=cmd,
                        workdir=self.vina_dir,
                        demux=True)

                    scores = process_stdout(stdout)

                    if len(scores) > 0 and scores != [None]:
                        all_scores[i] = scores

                    timing += [round(time.time() - t0, 2)]
                    dates += [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

                    if verbose:
                        print(f'- {self.device}:{self.device_id} | [{dates[-1]} | t={timing[-1]}s] Docked ligand {i+1}/{len(basenames)} | Affinity values: {all_scores[i]}...')
                    
                    if write_log:
                        log_path = os.path.join(results_path, 'log.tsv')
                        write_to_log(log_path, smiles[i], target, all_scores[i], ligand_paths_docked[i])

                    if clean: # Remove intermediate files (undocked ligand .pdbqt files)
                        os.remove(ligand_pdbqt_paths[i])
                        os.remove(ligand_paths_docked[i])
                except Exception as d:
                    print(d)
                    
        except Exception as e:
            print(f'Error has occurred while docking ligand {i}: {e, stderr}')
            raise e
        except KeyboardInterrupt:
            print('Docking interrupted by user')
        finally:
            self.remove_docker_container()
    
        return all_scores

    def start_docker_container(self):
        """ 
        Start Vina-GPU docker container (runs until it is killed)
        Returns:
            docker container object
        """
        container = self.client.containers.run(
            command='sleep infinity', # Keeps the container running until it is killed
            detach=True,              # Run container in background
            **self.docker_kwargs)
        
        return container
 
    def remove_docker_container(self):
        """
        Stop Vina-GPU docker container
        """
        self.container.remove(force=True) 
        self.container = None
        
    def prepare_ligand(self, smiles, out_path=None):
        """
        Prepare ligand for docking, return ligand .pdbqt file path

        Arguments:
            smiles (str)     : smiles string
            out_path (str)   : path to save the .pdbqt file (default: ./drugex/utils/docking/output)
        Returns:
            path to the ligand .pdbqt file
        """
        try:
            # Ligand preparation via rdkit and meeko
            mol = Chem.MolFromSmiles(smiles)             # type: ignore
            protonated_ligand = Chem.AddHs(mol)          # type: ignore
            AllChem.EmbedMolecule(protonated_ligand)     # type: ignore
            self.molecule_preparation.prepare(protonated_ligand)

            # Write to .pdbqt file required by Vina
            if out_path is None:
                out_path = self.out_path

            self.molecule_preparation.write_pdbqt_file(out_path)
        except Exception as e:
            print(f'Error while preparing ligand: {e}')
            out_path = None
        return out_path

    def prepare_target(self, pdb_path, output_path=None, chain='A'):
        """ 
        Prepare target for docking, return target pdbqt path
        Arguments:
            pdb_path (str)   : path to target .pdb file
            out_path (str)   : path to save the .pdbqt file
            chain (str)      : chain to use for docking (if target is a multi-chain protein)
        Returns:
            path to the processed target .pdbqt file
        """
        ## Output filenames
        
        # Prepare target
        if pdb_path.endswith('.pdb'): # If target is a .pdb file, convert to .pdbqt
            target_pdbqt_path = os.path.join(output_path, os.path.basename(pdb_path).replace('.pdb', '.pdbqt'))

            if not os.path.isfile(target_pdbqt_path):
                if output_path is None:
                    output_path = self.out_path

                basename = os.path.basename(pdb_path)
                out_file_path = os.path.join(output_path, basename)              # This is where the target .pdb file will be saved
                shutil.copyfile(pdb_path, out_file_path)                         # Copy target .pdb file to output folder   
                chain_basename = basename.replace('.pdb', f'_chain_{chain}.pdb') # Name of the .pdb file with only the selected chain
                chain_pdb_path = os.path.join(output_path, chain_basename)       # Full path to the .pdb file with only the selected chain
                pdbqt_basename = basename.replace('.pdb', '.pdbqt')              # Name of the .pdbqt file
                target_pdbqt_path = os.path.join(output_path, pdbqt_basename)    # Full path to the .pdbqt file
                
                # Processing within the docker container
                # Select a single chain in case the target is a multimer
                if self.container is None:
                    self.container = self.start_docker_container()
                try:
                    workdir = self.docking_dir + '/' + os.path.basename(output_path)
                    cmd = f"bash -c 'pdb_selchain -{chain} {basename} | pdb_delhetatm | \
                            pdb_tidy > {chain_basename}'"
                    self.container.exec_run(
                        cmd=cmd,
                        workdir=workdir,
                        demux=True)

                    ## Prepare the target for docking using ADFR Suite 'prepare_receptor' binary
                    adfr_binary = os.path.join(self.adfr_suite_docker_path, 'bin', 'prepare_receptor')
                    cmd = f'{adfr_binary} -r {chain_basename} -o {pdbqt_basename} -A checkhydrogens'
                    self.container.exec_run(
                        cmd=cmd,
                        workdir=workdir,
                        demux=True)
                except Exception as e:
                    print(f'Error while preparing target: {e}')
                except KeyboardInterrupt:
                    print('KeyboardInterrupt')
                finally:
                    self.remove_docker_container()
        else:
            target_pdbqt_path = None
            raise ValueError(f'Invalid file type: {pdb_path}')

        return target_pdbqt_path
        
'''
VinaGPU utils
'''

def run_executable(cmd, shell=True, **kwargs):
    """ Run executable command and return output from stdout and stderr """
    proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, shell=shell, **kwargs)
    stdout, stderr = proc.communicate()
    return (stdout, stderr)

def process_stdout(stdout):
    """ Processes the stdout of Vina, returns the affinity of each docking orientation. """
    affinities = []
    is_int = re.compile(r'^\s*\d+\s*$')

    for line in stdout.splitlines():
        if bool(is_int.match(line.decode('utf-8')[:4])):
            orientation_id, affinity, dist1, dist2  = line.split()
            affinities += [float(affinity)]
            
    return affinities

def compress_string(string):
    """
    Compresses a string
    Arguments:

        string (str)              : string to compress  
    Returns:
        compressed (str)          : compressed string
    """ 
    return zlib.compress(string.encode('utf-8')).hex()

def decompress_string(compressed):
    """
    Decompresses a compressed string
    Arguments:
        compressed (str)          : compressed string
    Returns:
        string (str)              : decompressed string
    """
    return zlib.decompress(bytes.fromhex(compressed)).decode('utf-8')

def write_to_log(log_path, smiles, target, scores, pdbqt_path=None):
    """
    Writes a log file
    Arguments:
        log_path (str)            : path to log file
        smiles (str)              : SMILES of ligand
        target (str)              : target name
        scores (list)             : list of scores
        pdbqt_path (str)          : path to pdbqt file
    """
    # If no log file exists, create one with a header
    if not os.path.isfile(log_path):
        with open(os.path.join(log_path), 'w') as f:
            header = '\t'.join(['smiles', 'target', 'score', 'molfile'])
            f.write(header + '\n')

    if pdbqt_path is not None: # If a pdbqt file is provided, read it in as PDBQT molecule
        with open(pdbqt_path, 'r') as f:
            pdbqt = f.read()

        pdbqt_mol = PDBQTMolecule(pdbqt, skip_typing=True)
    else:
        pdbqt_mol = None
    
    # If scores is not a list, make it a list
    if not isinstance(scores, list):
        scores = [scores]
    
    z = [str(score) for score in  scores] # Convert scores to strings

    # Write to log file
    with open(log_path, 'a') as f:
        for i, score in enumerate(z):
            if pdbqt_mol:
                rdkit_mol = pdbqt_mol[i].export_rdkit_mol()
                pose_block = Chem.MolToMolBlock(rdkit_mol)

                # Replace header with Smiles_target_VINA_poserank
                index = pose_block.find('3D') + 2
                title = smiles + f'_{target}_VINA_{i + 1}\n'
                pose_block = title + pose_block[index:]
                pose_block = compress_string(pose_block)
            else:
                pose_block = ''

            f.write('\t'.join([smiles, target, score, pose_block])+'\n')

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Pipeline')
    parser.add_argument('--run_name', type=str, help='Name of the run', required=True)
    parser.add_argument('--smi_file', type=str, help='Path to the SMILES file', required=True)
    parser.add_argument('--kinase_families', type=str, nargs='+', choices=['AGC', 'CAMK', 'CK1', 'CMGC', 'Other', 'STE', 'TK', 'TKL'], help='Kinase families to include', required=False)
    parser.add_argument('--accessions', type=str, nargs='+', help='Kinase accessions to include', required=False)
    parser.add_argument('--docking_engine', type=str, choices=['vina', 'diffdock'], help='Docking engine to use', required=True)
    parser.add_argument('--scoring_function', type=str, choices=['DNN'], help='Scoring function to use', default='DNN', required=False)
    parser.add_argument('--output_path', type=str, help='Path to the output folder', required=True)
    args = parser.parse_args()

    # Load SMILES
    if not os.path.isfile(args.smi_file):
        print('SMILES file not found!')
        exit()

    with open(args.smi_file, 'r') as f:
        smiles = [line.strip() for line in f if line.strip()] # strip whitespace and remove empty lines

    # Check if SMILES are valid (RDKit)
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)

        if not mol:
            print(f'Invalid SMILES: {smile}')
            exit()
    
    # Check if kinase families or accessions are selected
    if not args.kinase_families and not args.accessions:
        print('No kinase families or accessions selected!')
        exit()

    if not args.kinase_families:
        args.kinase_families = []

    # Load kinase data
    kin_data = pd.read_csv('kinase_data.csv')

    # Check if accessions are valid
    if args.accessions:
        for accession in args.accessions:
            if accession not in kin_data['accession'].unique():
                print('-'*50)
                print(f'Invalid accession: {accession}')
                print('-'*50)
                print('Valid accessions:')

                for acc in kin_data['accession'].unique():
                    print(f'- {acc}')

                exit()

    # Check if output path exists
    if not os.path.isdir(args.output_path):
        print(f'{args.output_path} not found!')
        exit()

    print('Start pipeline...')
    pipeline = Pipeline(args.run_name, smiles, args.kinase_families, args.accessions, args.docking_engine, args.scoring_function, args.output_path)
    pipeline.run()
