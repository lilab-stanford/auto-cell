import kaplanmeier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from core import Core
from lifelines import CoxPHFitter
from natsort import natsorted
from sklearn import tree
from sklearn import cluster
from sklearn import metrics
from tqdm import tqdm

class Dataset:
    """
    Class used to handle multiple cores at once, calculate their biomarkers
    and perform statistical analysis on them.
    
    """

    def __init__(self, directory=None):
        """
        Constructor for the Dataset class.

        Args:
            directory: String, directory where csv files containing cell data 
                        for all cores are stored.

        Returns:
            Dataset: Initialized dataset object.
        """

        # store biomarker value for every core in a list
        # with biomarker name as key
        self.biomarkers = {}

        # store biomarker mean value
        self.biomarkers_mean = {}

        # store best cutoff for lowest log-rank p-value 
        # for every biomarker
        self.biomarker_best_cutoff = {}

        # store log-rank p-value for every biomarker for
        # median, mean and optimal as cutoffs
        self.log_rank_p = {}

        # store cox p-value for every biomarker 
        # using a univariate cox model
        self.cox_p = {}

        # store hazard ratio for every biomarker
        # using a univariate cox model
        self.hazard_ratios = {}

        # store patient overall survival in months
        self.patient_months = []

        # store patient status (dead or alive)
        self.patient_status = []

        # store cores in a list where each core corresponds
        # to a single patient
        self.cores = []

        # store cores name in a list
        self.cores_name = []
        if directory is not None:
            self.load_from_directory(directory)

    def sync_cell_types(self):
        """
        This function is used to snychronize the cell types of all cores, 
        such that each core contains the same cell types in the same order.

        """
        print("Synchronizing cell types over all cores in dataset...")

        # Find all cell types in all cores
        cell_types = set()
        for core in self.cores:
            for cell_type in core.cell_types_set:
                cell_types.add(cell_type)
        
        # Add missing cell types to all cores
        for core in self.cores:
            core.cell_types_set = sorted(list(cell_types))
            for cell_type in core.cell_types_set:
                if not cell_type in core.cell_types_dic.keys():
                    core.cell_types_dic[cell_type] = len(core.cell_types_dic)
                    core.cell_types_number[cell_type] = 0

    def load_from_directory(self, directory):
        """
        This function is used to load all cores from a given directory.

        Args:
            directory: String, directory where csv files containing cell data 
                        for all cores are stored.
        """

        print("Loading cores from directory " + directory + " ...")
        for filename in natsorted(os.listdir(directory)):
            if filename.endswith(".csv"):
                core = Core(os.path.join(directory, filename))
                self.cores.append(core)
                self.cores_name.append(core.name)
        self.sync_cell_types()

    def load_patient_from_csv(self, csv_file):
        """
        This function is used to load all patients from a given csv file.

        Args:
            csv_file: String, file containing overall survival and status
                        for each core in the dataset.
        """

        df = pd.read_csv(csv_file)

        # Check if core is contained in csv file
        for core in self.cores:
            try:
                core.load_patient_from_csv_(csv_file)
                self.patient_months.append(core.patient_months)
                self.patient_status.append(core.patient_status)

            except ValueError:
                print("Core " + core.name + " not found in patient information csv file")

        self.patient_months = np.array(self.patient_months)
        self.patient_status = np.array(self.patient_status)

    def calculate_biomarker_mean(self):
        """
        This function calculates the mean value of every biomarker
        throughout the entire dataset.

        """

        for biomarker in self.biomarkers.keys():
            self.biomarkers_mean[biomarker] = np.nanmean(self.biomarkers[biomarker])

    def calculate_biomarker(self, *biomarkers):
        """
        This function calculates the value of every biomarker whose
        name is given as a string argument. If no argument is given,
        all biomarkers are calculated.

        Args:
            *biomarkers: Strings, names of biomarkers to be calculated.
        """

        # if no biomarker is given, calculate all biomarkers
        if not biomarkers:
            biomarkers = []
            for method in dir(Core):
                if not method.endswith("_"):
                    biomarkers.append(method)

        # Calculate affiliations of every cell in the entire dataset
        # to a celullar neighbourhood before any biomarkers based
        # on cellular neighbourhoods are calculated
        cn = False
        for biomarker in biomarkers:
            if "cellular_neighbourhoods" in biomarker and not hasattr(self.cores[0], "neighbourhoods_number"):
                cn = True
                break

        if cn:
            print("Calculating cellular neighbourhoods...")
            for i in tqdm(range(len(self.cores))):
                self.cores[i].calculate_neighbourhoods_()
                if i == 0:
                    neighbourhoods = self.cores[i].neighbourhoods
                else:
                    neighbourhoods = np.concatenate((neighbourhoods, 
                                                    self.cores[i].neighbourhoods), axis=0)
            
            # Amount of cellular neighbourhoods
            k = 8
            clusterer = cluster.MiniBatchKMeans(n_clusters=k, random_state=0).fit(neighbourhoods)

            # Assign cellular neighbourhoods to each cell in each core
            for i in range(len(self.cores)):
                self.cores[i].neighbourhoods_labels = clusterer.predict(self.cores[i].neighbourhoods)
                self.cores[i].neighbourhoods_number = k

        # Calculate every biomarkers on every core
        for biomarker in biomarkers:
            print("Calculating biomarker " + biomarker)
            try:
                for i in tqdm(range(len(self.cores))):
                    biomarker_dic = getattr(self.cores[i], biomarker)()
                    for marker in biomarker_dic.keys():
                        if i == 0:
                            self.biomarkers[marker] = [biomarker_dic[marker]]
                        else:
                            self.biomarkers[marker].append(biomarker_dic[marker])

            # Complain when the given biomarker name is not defined
            except AttributeError:
                print("Can't calculate biomarker " + biomarker + " as it is not defined!")

    def kaplan_meier(self, *biomarkers):
        """
        This function is used to plot a Kaplan-Meier curve for a given biomarker.
        If no argument is given, all biomarkers are used.

        Args:
            *biomarkers: Strings, names of biomarkers for which Kaplan-Meier curves
                        are to be plotted.
        """

        for biomarker in biomarkers:
            try:
                biomarker_values = np.array(self.biomarkers[biomarker])

            except KeyError:
                print("Biomarker " + biomarker + " not defined or not calculated yet!")
                continue
            
            biomarker_median = np.nanmedian(biomarker_values)
            group0 = biomarker_values > biomarker_median
            results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            kaplanmeier.plot(results, savepath="./kaplan_meier/"+biomarker+"_kaplan_meier.png", 
                             title="Kaplan-Meier curve for {} \n Log-rank p-value = {:.4f}"
                             .format(biomarker, results['logrank_P']))

    def log_rank_test(self, *biomarkers):
        """
        This function is used to calculate the log-rank p-value as well as 
        the optimal cutoff threshold for a given biomarker.
        If no argument is given, p-values for all biomarkers are computed.

        Args:
            *biomarkers: Strings, names of biomarkers for which log-rank p-values
                        and cutoffs are to be calculated.
        """

        # if no biomarker is given, calculate all biomarkers
        if not biomarkers:
            biomarkers = self.biomarkers.keys()
        
        for biomarker in biomarkers:
            try:
                biomarker_values = np.array(self.biomarkers[biomarker])

            except KeyError:
                print("Biomarker " + biomarker + " not defined or not calculated yet!")
                continue

            if biomarker_values.size != len(self.patient_months):
                print("Biomarker " + biomarker + " has not the same number of patients as the patient information!")
                continue
            
            # Calculate log-rank p-value for the median as threshold
            biomarker_median = np.nanmedian(biomarker_values)
            group0 = biomarker_values > biomarker_median
            results_median = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            self.log_rank_p[biomarker] = [results_median['logrank_P']]

            # Calculate log-rank p-value for the mean as threshold
            biomarker_mean = np.nanmean(biomarker_values)
            group0 = biomarker_values > biomarker_mean
            results_median = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
            self.log_rank_p[biomarker].append(results_median['logrank_P'])

            # Calculate log-rank p-value for the optimal threshold
            cutoffs = np.linspace(np.nanmin(biomarker_values), np.nanmax(biomarker_values), 20)
            cutoffs = cutoffs[1:-1]
            log_rank_p = []
            for cutoff in cutoffs:
                group0 = biomarker_values > cutoff
                results = kaplanmeier.fit(self.patient_months, self.patient_status, group0)
                log_rank_p.append(results['logrank_P'])

            log_rank_p = np.array(log_rank_p)
            if not np.isnan(log_rank_p).all():
                self.log_rank_p[biomarker].append(np.nanmin(log_rank_p))
                self.biomarker_best_cutoff[biomarker] = cutoffs[np.nanargmin(log_rank_p)]

    def univariate_cox_model(self, *biomarkers):
        """
        This function is used to calculate the cox p-value and hazard ratio based
        on a univariate cox model for the given biomarkers. If no argument is 
        given, p-values for all biomarkers are computed.

        Args:
            *biomarkers: Strings, names of biomarkers for which p-value and hazard ratio
                         are to be calculated.
        """

        # if no biomarker is given, calculate all biomarkers
        if not biomarkers:
            biomarkers = self.biomarkers.keys()
        
        for biomarker in biomarkers:
            try:
                biomarker_values = np.array(self.biomarkers[biomarker])

            except KeyError:
                print("Biomarker " + biomarker + " not defined or not calculated yet!")
                continue
            
            if biomarker_values.size != len(self.patient_months):
                print("Biomarker " + biomarker + " has not the same number of patients as the patient information!")
                continue
            
            # Calculate cox p-value
            df = pd.DataFrame(biomarker_values, columns=["biomarker"])
            df["months"] = self.patient_months
            df["status"] = self.patient_status
            cox = CoxPHFitter()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna()
            if len(df) == 0:
                continue
            cox.fit(df[["months", "status", "biomarker"]], duration_col="months", event_col="status")
            self.hazard_ratios[biomarker] = cox.hazard_ratios_.values[0]
            self.cox_p[biomarker] = cox.summary[["p"]].values[0][0]

    def construct_decision_tree(self, *biomarkers, max_depth=10, survival_cutoff=24):
        """
        This function is used to build a decision tree based on the given biomarkers,
        with a maximum depth of max_depth. The decision tree is then used to predict
        the response label for each patient, which is based on the survival cutoff.

        Args:
            *biomarkers: Strings, names of biomarkers which should be considered for
                            the decision tree.
            max_depth: Integer, maximum depth of the decision tree.
            survival_cutoff: Integer, survival cutoff in months.
        """
        
        # if no biomarker is given, calculate all biomarkers
        if not biomarkers:
            biomarkers = self.biomarkers.keys()
        
        # Create decision tree
        self.decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=max_depth)
        features, labels,_ = self.get_features(*biomarkers, survival_cutoff=survival_cutoff)
        self.decision_tree = self.decision_tree.fit(features, labels)

    def get_features(self, *biomarkers, survival_cutoff=24):
        """
        This function is used to get the features and labels for the decision tree.
        Nan and inf values are removed from the features.

        Args:
            *biomarkers: Strings, names of biomarkers which should be considered for
                            the decision tree.
            survival_cutoff: Integer, survival cutoff in months.

        Returns:
            features: Numpy array, features for the decision tree.
            labels: Numpy array, labels for the decision tree (0, 1 for binary classification).
            features_names: List of strings, names of the features.
        """

        # if no biomarker is given, calculate all biomarkers
        if not biomarkers:
            biomarkers = self.biomarkers.keys()

        features = []
        features_names = []
        labels = []

        # Turn into binary classification problem according to survival cutoff
        labels = np.array(self.patient_months) > survival_cutoff

        for biomarker in biomarkers:
            if not "Others" in biomarker:
                if len(self.biomarkers[biomarker]) == len(self.patient_months):
                    features.append(self.biomarkers[biomarker])
                    features_names.append(biomarker)
        features = np.array(features)

        # Remove nan and inf values
        mask = ~np.logical_or(np.isnan(features).any(axis=0), np.isinf(features).any(axis=0))
        features = np.transpose(features[:, mask])
        labels = labels[mask]
        return features, labels, features_names

    def save(self, filename=""):
        """
        This function is used to save the dataset to a pickle file. All the attributes
        in the list "data" defined below are saved.

        Args:
            filename: String, name of the pickle file.
        """

        # default filename
        if filename == "":
            filename = "dataset.pkl"

        # Information to be stored
        data = [self.biomarkers, self.biomarkers_mean, self.log_rank_p, self.cores_name, 
                self.patient_months, self.patient_status, self.biomarker_best_cutoff,
                self.cox_p, self.hazard_ratios]
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
    def load(self, filename=""):
        """
        This function is used to load the dataset from a pickle file.

        Args:
            filename: String, name of the pickle file.
        """

        # default filename
        if filename == "":
            filename = "dataset.pkl"

        with open(filename, "rb") as f:
            data = pickle.load(f)
        
        loaded_biomarkers = data[0]
        loaded_biomarkers_mean = data[1]
        loaded_log_rank_p = data[2]
        loaded_cores_name = data[3]
        loaded_patient_months = data[4]
        loaded_patient_status = data[5]
        loaded_biomarker_best_cutoff = data[6]
        loaded_cox_p = data[7]
        #loaded_hr = data[8]

        for key, value in loaded_biomarkers.items():
            self.biomarkers[key] = value
        for key, value in loaded_biomarkers_mean.items():
            self.biomarkers_mean[key] = value
        for key, value in loaded_log_rank_p.items():
            self.log_rank_p[key] = value
        for key, value in loaded_cox_p.items():
            self.cox_p[key] = value
        #for key, value in loaded_hr.items():
        #    self.hazard_ratios[key] = value
        if self.cores_name == []:
            self.cores_name = loaded_cores_name
        if self.patient_months == []:
            self.patient_months = loaded_patient_months
        if self.patient_status == []:
            self.patient_status = loaded_patient_status
        for key, value in loaded_biomarker_best_cutoff.items():
            self.biomarker_best_cutoff[key] = value
