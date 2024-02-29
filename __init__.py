'''
First two functions are pymol's plugin setup
'''

def __init_plugin__(app=None):
    '''
    Add an entry to the PyMOL "Plugin" menu
    '''
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('KinaseDocker\u00b2', run_plugin_gui)

# global reference to avoid garbage collection of the dialog
dialog = None

def run_plugin_gui():
    '''
    Open custom dialog
    '''
    global dialog

    if dialog is None:
        dialog = make_dialog()

    dialog.show()

def make_dialog():
    '''
    This function creates the plugin dialog in which the entire plugin is situated
    '''
    # Relevant imports inside function to not delay pymol startup
    import os

    import pandas as pd
    from pymol import cmd
    from pymol.Qt.utils import loadUi
    from pymol.Qt import QtWidgets, QtGui, QtCore
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from .pipeline import Pipeline

    class MainWindow(QtWidgets.QDialog):
        '''
        Main window of the plugin
        '''
        def __init__(self):
            super().__init__()
            loadUi(os.path.join(os.path.dirname(__file__), 'docker_tool.ui'), self) # load .ui file
            self.setFixedSize(self.size()) # prevent resizing

            # Hook signals and slots (roughly in order)
            self.smiles_input.textChanged.connect(self.generate_mol)
            self.smi_file = None
            self.browse_smi.clicked.connect(self.browse_smi_files)

            self.select_all.clicked.connect(lambda: [child.setChecked(True) for child in self.findChildren(QtWidgets.QCheckBox)]) # Selects all checkboxes in the window!

            self.accessions = None
            self.browse_kinase.clicked.connect(self.browse_kinase_table)
            
            self.output_folder = None
            self.browse_output.clicked.connect(self.browse_output_folder)

            self.load_prev_results.clicked.connect(self.view_results)
            self.run.clicked.connect(self.start_process)

        def start_process(self):
            '''
            Start the pipeline
            '''
            # Get all user input
            run_name = self.run_name.text()
            smiles = [self.smiles_input.text()]

            if not smiles[0] and self.smi_file:
                with open(self.smi_file, 'r') as f:
                    smiles = [line.strip() for line in f if line.strip()] # strip whitespace and remove empty lines

            accessions = self.accessions
            kinase_families = sorted([child.objectName() for child in self.findChildren(QtWidgets.QCheckBox) if child.isChecked()])
            docking_engine = self.docking_engine.currentText().lower()
            output_folder = self.output_folder
            scoring_algorithm = self.scoring_algorithm.currentText()

            # Check if all input is valid:
            if not run_name:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No run name specified')
                return

            if not smiles[0]:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No SMILES selected')
                return

            if not self.check_smiles(smiles):
                return

            if not len(kinase_families) and not accessions:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No kinase (families) selected')
                return

            if not docking_engine:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No docking engine selected')
                return

            if not output_folder or not os.path.exists(output_folder):
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No valid output folder selected')
                return
            
            if not scoring_algorithm:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No docking score selected')
                return

            print('Start pipeline...')
            pipeline = Pipeline(run_name, smiles, kinase_families, accessions, docking_engine, scoring_algorithm, output_folder)
            pipeline.run()
            results_sdf_filepath, _ = pipeline.get_results_filepath() # get results .sdf filepath and ignore the .csv filepath

            print('View results...')
            self.view_results(results_sdf_filepath)
            
        def view_results(self, results_path=None):
            '''
            View results in pymol

            This function creates the ResultViewer class and loads the associated Qdialog
            '''
            cmd.reinitialize()
            self.results_view = ResultViewer()

            # This check only loads data if called directly from pipeline otherwise loads the empty dialog
            if results_path:
                self.results_view.load_data(results_path)
            
            self.results_view.show()

        def generate_mol(self):
            '''
            This function dynamically generates a molecule in pymol from the SMILES input
            '''
            cmd.delete('mol') # delete previous molecule
            smiles = self.smiles_input.text()
            
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                
                if mol:
                    # Add H's and generate 3D coordinates
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    AllChem.UFFOptimizeMolecule(mol)

                    cmd.read_molstr(Chem.MolToMolBlock(mol), 'mol')
                
        def generate_mols(self):
            '''
            This function generates molecules in pymol from the .smi file input
            '''
            cmd.delete('mols') # delete previous molecules
            smi_file = self.smi_filename.text()

            if smi_file:
                with open(smi_file, 'r') as f:
                    for line in f:
                        line = line.strip()

                        # Skip empty lines
                        if not line:
                            continue

                        mol = Chem.MolFromSmiles(line)

                        if mol:
                            # Add H's and generate 3D coordinates
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol)
                            AllChem.UFFOptimizeMolecule(mol)

                            cmd.read_molstr(Chem.MolToMolBlock(mol), 'mols', state=0)
                        else:
                            # Immediately warn user if one of their SMILES is invalid
                            QtWidgets.QMessageBox.warning(self, 'Warning', f'Could not parse {line} in RDKit')
                            return
                        
        def browse_smi_files(self):
            '''
            Browse for .smi file
            '''
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .smi file', QtCore.QDir.rootPath() , '*.smi')
            self.smi_file = filename

            if filename:
                self.smi_filename.setText(filename)
                self.smiles_input.setText('') # clear smiles input, because the smiles_input has a higher priority in the pipeline
                self.generate_mols()

        def browse_output_folder(self):
            '''
            Browse for output folder
            '''
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select output folder', QtCore.QDir.rootPath())
            self.output_folder = folder

            if folder:
                self.output_foldername.setText(folder)

        def browse_kinase_table(self):
            '''
            This function creates the KinaseSelector class and loads the associated Qdialog
            '''
            dialog = KinaseSelector()

            if dialog.exec_() == QtWidgets.QDialog.Accepted: # The exec_() function forces a modal that prevents interaction with other windows
                kinase_data = dialog.get_kinases() # get all selected kinases

                # Retrieve accessions and update label
                kinases, self.accessions = zip(*kinase_data)
                label_text = '\n'.join([f'{kinase} ({accession})' for kinase, accession in kinase_data[:3]])
                label_text += '\n...' if len(kinase_data) > 3 else ''
                self.kinase_label.setText(label_text)

                # Add tooltip with all kinases
                self.kinase_label.setToolTip('\n'.join([f'{kinase} ({accession})' for kinase, accession in kinase_data]))

        def check_smiles(self, smiles):
            '''
            Check if all SMILES are valid (according to RDKit)
            '''
            for smile in smiles:
                mol = Chem.MolFromSmiles(smile)

                if not mol:
                    QtWidgets.QMessageBox.warning(self, 'Warning', f'Could not parse {smile} in RDKit')
                    return False

            return True

    class ResultViewer(QtWidgets.QDialog):
        '''
        This class creates the results viewer dialog
        '''
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Results')
            self.setMinimumSize(self.size())

            # Setup loading results
            self.load_button = QtWidgets.QToolButton()
            self.load_button.setText('Load .sdf results...')
            self.load_button.clicked.connect(self.browse_results)

            self.results_label = QtWidgets.QLabel('No results loaded')
            
            # Setup table
            self.table = QtWidgets.QTableWidget()
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.table.setSortingEnabled(False)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

            # Fire function when cell is clicked
            self.table.cellClicked.connect(self.cell_clicked)

            # Setup buttons
            self.exit_button = QtWidgets.QPushButton('Exit')
            self.exit_button.clicked.connect(self.accept)

            # Create layout
            layout = QtWidgets.QVBoxLayout()

            layout.addWidget(self.load_button)
            layout.addWidget(self.results_label)
            layout.addWidget(self.table)
            
            button_layout = QtWidgets.QHBoxLayout()

            spacer = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            button_layout.addItem(spacer)
            button_layout.addWidget(self.exit_button)

            layout.addLayout(button_layout)

            self.setLayout(layout)

        def load_data(self, results_path):
            '''
            Load results
            '''
            # Check if results file exists
            if not os.path.exists(results_path):
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No valid results file (.sdf) selected')
                return
            
            # Check if pdb folder exists for interactive rendering
            if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(results_path)), 'pdb')):
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No valid PDB folder found\n \
                                                                Did you adhere to the correct folder structure?\n \
                                                                Results\n\t|_ results.sdf\npdb\n\t|_ pdb files\n')
                return
            
            cmd.reinitialize() # clear pymol, to prevent clashing object names

            self.results_path = results_path
            self.results_label.setText(f'Showing results from {results_path}')

            # Load SDF file into RDKit
            molecules = Chem.SDMolSupplier(results_path)
            property_names = list(molecules[0].GetPropNames())
            all_props = []

            # Retrieve properties
            for i, mol in enumerate(molecules):
                properties = [mol, mol.GetProp('_Name')] + [mol.GetProp(prop) for prop in property_names]
                all_props.append(properties)

            # Create dataframe
            self.pose_results = pd.DataFrame(all_props, columns=['Molecule', 'SMILES'] + property_names)
            self.pose_results['clash_score'] = self.pose_results['clash_score'].astype(float)
            self.pose_results['pIC50'] = self.pose_results['pIC50'].astype(float)

            # Get averaged results
            self.agg_results = self.pose_results.groupby(['klifs_ID', 'SMILES'], sort=False).agg({'Kinase': 'first', 'accession': 'first', 'pIC50': 'mean', 'clash_score': 'max'}).reset_index()
            self.agg_results.rename(columns={'pIC50': 'avg_score', 'clash_score': 'clash_score_max'}, inplace=True)
            self.agg_results['avg_score'] = self.agg_results['avg_score'].round(2)
            
            # Create a complex id that goes up in number like complex_0 complex_1 etc.
            self.agg_results['complex_ID'] = self.agg_results.index.values
            self.agg_results['complex_ID'] = self.agg_results['complex_ID'].apply(lambda x: f'complex_{x}')

            self.agg_results = self.agg_results.reindex(columns=['complex_ID', 'SMILES', 'Kinase', 'accession', 'klifs_ID', 'avg_score', 'clash_score_max'])

            # Populate table
            n_rows = len(self.agg_results)
            n_cols = len(self.agg_results.columns)

            self.table.setRowCount(n_rows)
            self.table.setColumnCount(n_cols)

            self.table.setHorizontalHeaderLabels(self.agg_results.columns)
            self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

            for i, row in self.agg_results.iterrows():
                for j, col in enumerate(self.agg_results.columns):
                    item = QtWidgets.QTableWidgetItem(str(row[col]))
                    self.table.setItem(i, j, item)
                    
                    if col == 'clash_score_max':
                        if row[col] > 10:
                            item.setForeground(QtGui.QColor(255, 0, 0))
                    
            self.table.setSortingEnabled(True)
            
            # Show first complex in pymol
            self.cell_clicked(0, 0)
            self.table.selectRow(0)

        def cell_clicked(self, row, col):
            '''
            This function is called when a cell is clicked in the table.
            It loads the corresponding complex in pymol as a separate KLIFS object and a separate complex_{x} object with the poses as states:
                - If the complex is already loaded, it will be enabled
                - If the complex is not loaded, it will be loaded
            '''
            cmd.disable('all') # Disable all possible previous complexes

            # Get all values from row and put in a dict with corresponding column name
            row_values = {self.table.horizontalHeaderItem(i).text(): self.table.item(row, i).text() for i in range(self.table.columnCount())}
            klifs = row_values['klifs_ID']
            smiles = row_values['SMILES']

            existing_objects = cmd.get_names('objects')

            # If complex is already loaded, enable it and return
            if row_values['complex_ID'] in existing_objects:
                cmd.enable(f'{klifs}')
                cmd.enable(row_values['complex_ID'])
                return

            # Get all poses with the same klifs_ID and SMILES
            poses = self.pose_results[(self.pose_results['klifs_ID'] == klifs) & (self.pose_results['SMILES'] == smiles)]

            # Load PDB as object if not already loaded, otherwise enable it
            if klifs not in existing_objects:
                pdb_path = os.path.join(os.path.dirname(os.path.dirname(self.results_path)), 'pdb', f'{klifs}.pdb')
                cmd.load(pdb_path, object=f'{klifs}')
            else:
                cmd.enable(f'{klifs}')
                
            # Load poses as states in a separate complex_{x} object
            for pose in poses['Molecule']:
                cmd.read_molstr(Chem.MolToMolBlock(pose), row_values['complex_ID'], state=0)

        def browse_results(self):
            '''
            Browse for .sdf file
            '''
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .sdf file', QtCore.QDir.rootPath() , '*.sdf')

            if filename:
                self.load_data(filename)

    class KinaseSelector(QtWidgets.QDialog):
        '''
        This class creates the kinase selector dialog
        '''
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Select kinase(s)')
            self.setMinimumSize(self.size())

            # Setup search bar
            self.query = QtWidgets.QLineEdit()
            self.query.setPlaceholderText("Search...")
            self.query.textChanged.connect(self.search)

            # Load kinase data
            kinases = pd.read_csv(os.path.join(os.path.dirname(__file__), 'kinase_data.csv'), usecols=['kinasegroup', 'kinase', 'accession'])
            kinases = kinases.drop_duplicates(keep='first').sort_values(by=['kinasegroup']).reset_index(drop=True) # Sort table by kinasegroup

            # Setup table
            n_rows = len(kinases)
            n_cols = 3

            self.table = QtWidgets.QTableWidget()
            self.table.setRowCount(n_rows)
            self.table.setColumnCount(n_cols)
            self.table.setSortingEnabled(True)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

            self.table.setHorizontalHeaderLabels(['Group', 'Kinase', 'Accession'])
            self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

            # Populate table
            for i, row in kinases.iterrows():
                accession_checkbox = CheckboxTableWidgetItem(row['kinasegroup'])

                self.table.setItem(i, 0, accession_checkbox)
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(row['kinase']))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(row['accession']))

            # If row is clicked, toggle the corresponding checkbox (Apparently clicking the checkbox directly can only enable and not disable)
            self.table.cellClicked.connect(self.cell_clicked)

            # Setup buttons
            self.ok = QtWidgets.QPushButton('OK')
            self.ok.clicked.connect(self.accept)

            self.cancel = QtWidgets.QPushButton('Cancel')
            self.cancel.clicked.connect(self.reject)

            # Setup labels
            self.kinase_counter = QtWidgets.QLabel(f'0 kinases selected')

            # Create layout
            layout = QtWidgets.QVBoxLayout()

            layout.addWidget(self.query)
            layout.addWidget(self.table)
            layout.addWidget(self.kinase_counter)

            button_layout = QtWidgets.QHBoxLayout()
            button_layout.addWidget(self.cancel)
            button_layout.addWidget(self.ok)

            layout.addLayout(button_layout)

            self.setLayout(layout)

        def cell_clicked(self, row, col):
            '''
            This function is called when a cell is clicked in the table.
            It toggles the corresponding checkbox
            '''
            # Toggle checkbox
            self.table.item(row, 0).setCheckState(QtCore.Qt.Checked if self.table.item(row, 0).checkState() == QtCore.Qt.Unchecked else QtCore.Qt.Unchecked)

            # Count number of checkboxes in the first column that are checked
            num_checked = len(self.get_kinases())
            self.kinase_counter.setText(f'{num_checked} kinases selected')

        def get_kinases(self):
            '''
            Get all checked kinases
            '''
            rows = [i for i in range(self.table.rowCount()) if self.table.item(i, 0).checkState() == QtCore.Qt.Checked]
            checked_items = [(self.table.item(row, 1).text(), self.table.item(row, 2).text()) for row in rows] # Extract row values

            return checked_items

        def accept(self):
            '''
            Accept function for the dialog, activates when the OK button is pressed
            '''
            checked_items = self.get_kinases()

            # Check if any kinases are selected
            if len(checked_items):
                super().accept()
            else:
                QtWidgets.QMessageBox.warning(self, 'Warning', 'No kinase selected')

        def reject(self):
            '''
            Reject function for the dialog, activates when the Cancel button is pressed
            '''
            super().reject()

        def search(self, s):
            '''
            Dynamic earch function for the dialog, activates when the search bar is used
            '''
            # Clear current selection.
            self.table.setCurrentItem(None)

            if not s:
                # Empty string, don't search.
                return

            matching_items = self.table.findItems(s, QtCore.Qt.MatchContains) # Find all items that contain the search string.

            if matching_items:
                item = matching_items[0]
                self.table.setCurrentItem(item) # Select the first matching item.
                
    class CheckboxTableWidgetItem(QtWidgets.QTableWidgetItem):
        '''
        This class creates a custom QTableWidgetItem with a checkbox
        '''
        def __init__(self, text):
            super().__init__(text, QtWidgets.QTableWidgetItem.UserType)
            # These settings are required to make the checkbox work
            self.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            self.setCheckState(QtCore.Qt.Unchecked) # Default state is unchecked

        def __lt__(self, other):
            '''
            This function overrides the __lt__ function which is used when sorting the table.
            It ensures that the checkboxes are sorted correctly.
            '''
            if self.checkState() == other.checkState():
                return self.text() < other.text() # If the checkboxes are the same, sort alphabetically
            elif self.checkState() == QtCore.Qt.Unchecked:
                return False # A checked state is sorted lower than an unchecked state

            return True

    dialog = MainWindow()

    return dialog




