import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class Core:
    """
    Class used to handle the single cell map of an H&E core and calculate biomarker
    expressions of the core.
    
    """

    def __init__(self, csv_file=None):
        """
        Constructor for the Core class.

        Args:
            csv_file: String, path to the csv file containing the 
            single cell map data. Defaults to None.

        Returns:
            Core: Core object with the data from the csv file.
        """

        if csv_file is not None:
            self.load_from_csv_(csv_file)

    def load_from_csv_(self, csv_file):
        """
        This function loads the single cell map data from a csv file 
        into the Core object.

        Args:
            csv_file: String, path to the csv file containing 
                    the single cell map data.
        """

        self.df = pd.read_csv(csv_file)

        # Name of core and cell coordinates
        self.name = os.path.basename(csv_file).split(".")[0]
        self.cell_coordinates = self.df.loc[:, ["Centroid X px", "Centroid Y px"]].values / 0.497
        self.cell_areas = self.df.loc[:, ["Area px^2"]].values
        
        # Area of the core
        self.area = np.pi * np.prod(np.max(self.cell_coordinates, axis=0) - np.min(self.cell_coordinates, axis=0))
    
        # Cell types and their integer encoding
        self.cell_types = self.df.loc[:, "Class Type"].values.tolist()
        self.cell_types_set = sorted(list(set(self.cell_types)))
        self.cell_types_dic = dict(zip(self.cell_types_set, np.arange(len(self.cell_types_set))))
        self.cell_types = np.array([self.cell_types_dic[cell] for cell in self.cell_types])

        # Number of cells of each type
        self.cell_types_number = dict(zip(self.cell_types_set, [np.sum(self.cell_types == i) for i in np.arange(len(self.cell_types_set))]))

        # Number of cells in the core
        self.cell_number = self.cell_types.shape[0]

        # dictionary for all biomarkers that are implemented
        self.biomarkers = {}

    def load_patient_from_csv_(self, csv_file):
        """
        This function loads the patient overall survival, status and ID from a csv file.

        Args:
            csv_file: String, path to the csv file containing all the patient information
                    of the entire dataset.
        """
        df = pd.read_csv(csv_file)

        # Check if core is contained in csv file
        if df.loc[df["Core ID"] == self.name].empty:
            raise ValueError("Patient information not found for core " + self.name) 
        
        # Set patient information
        self.patient_id = df.loc[df["Core ID"] == self.name]["Patient ID"].values[0]
        self.patient_months = int(df.loc[df["Core ID"] == self.name]["OS(Months)"].values[0])
        self.patient_status = bool(df.loc[df["Core ID"] == self.name]["Status"].values[0])

    def calculate_cell_distances_(self):
        """
        This function calculates the distance matrix between all cells in the core.
        Distances between a cell with itself are set to infinity.

        """

        self.cell_distances = distance_matrix(self.cell_coordinates, self.cell_coordinates)
        self.cell_distances[self.cell_distances==0] = np.inf

    def calculate_neighbourhoods_(self, k=50):
        """
        This function finds the k nearest neighbours of each cell in the core and
        calculates the distribution of the neighbouring cell types. The cellular neighbourhood
        clustering is based on this distribution.

        Args:
            k: Integer, number of nearest neighbours to consider. Defaults to 10.
        """

        # Calculate distance matrix if not already done
        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        # Find k nearest neighbours for each cell and save the distribution of cell types
        # in self.neighbourhoods
        self.neighbourhoods = np.zeros((self.cell_number, len(self.cell_types_set)))
        nearest_neighbours_idx = np.argpartition(self.cell_distances, k, axis=1)[:, :k]
        for i in range(self.cell_number):
            nearest_neighbours = self.cell_types[nearest_neighbours_idx[i,:]]
            self.neighbourhoods[i,:] = np.bincount(nearest_neighbours, minlength=len(self.cell_types_set))/(k+1)
            self.neighbourhoods[i,self.cell_types[i]] += 1 / (k+1)


    # +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= #
    #                            BIOMARKERS                            #
    # +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= #

    def fraction_cell_type(self):
        """
        This function calculates the fraction of cells of each type in the core.
        
        Returns:
            returns: Dictionary, keys are the biomarker names
                    and values are the fraction of cells.
        """

        returns = {}
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_fraction"] = self.cell_types_number[cell_type] / self.cell_number
            returns[cell_type + "_fraction"] = self.biomarkers[cell_type + "_fraction"]
        return returns

    def density_cell_type(self):
        """
        This function calculates the areal density of each cell type in the core.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values are the densities.
        """

        returns = {}
        self.biomarkers["cell_density_mu^2"] = self.cell_number / self.area
        returns["cell_density_mu^2"] = self.biomarkers["cell_density_mu^2"]
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_density_mu^2"] = self.cell_types_number[cell_type] / self.area
            returns[cell_type + "_density_mu^2"] = self.biomarkers[cell_type + "_density_mu^2"]
        return returns
    
    def area_cell_type(self):
        """
        This function calculates the average area of the cells of each cell type.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values are the 
                    average cell areas.
        """

        returns = {}
        self.biomarkers["average_area_px^2"] = np.mean(self.cell_areas)
        returns["average_area_px^2"] = self.biomarkers["average_area_px^2"]
        for cell_type in self.cell_types_set:
            self.biomarkers[cell_type + "_average_area_px^2"] = np.mean(self.cell_areas[self.cell_types == self.cell_types_dic[cell_type]])
            returns[cell_type + "_average_area_px^2"] = self.biomarkers[cell_type + "_average_area_px^2"]
        return returns
    
    def neighbouring_cell_type_distance_cutoff(self, radius=50):
        """
        This function finds the average amount of cells of type2 within a radius around
        cells of type1. The average is calculated for each possible combination of cell types.
        Moreover, the average amount of cells regardless of their type 
        within a radius around cells of type1 is also calculated.

        Args:
            radius: Float, radius in micrometers. Defaults to 50.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the average cell counts within radius.
        """

        returns = {}

        # Calculate distance matrix if not already done
        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        for cell_type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[cell_type]
            distances = self.cell_distances[mask, :]
            close_cells_idx = distances < radius
            counts = np.zeros(len(self.cell_types_set))

            for i in range(close_cells_idx.shape[0]):
                close_cells = self.cell_types[close_cells_idx[i,:]]
                counts += np.histogram(close_cells, bins=np.arange(len(self.cell_types_set)+1))[0]

            self.biomarkers["Average amount of cells within " + str(radius) + "mu around " + cell_type] = np.sum(counts) / np.sum(mask)
            returns["Average amount of cells within " + str(radius) + "mu around " + cell_type] = np.sum(counts) / np.sum(mask)

            for i in range(counts.shape[0]):
                self.biomarkers["Average amount of " + self.cell_types_set[i] + " cells within " + str(radius) + "mu of " + cell_type] = counts[i] / np.sum(mask)
                returns["Average amount of " + self.cell_types_set[i] + " cells within " + str(radius) + "mu of " + cell_type] = counts[i] / np.sum(mask)
        return returns

    def neighbouring_cell_type_amount_cutoff(self, k=50):
        """
        This function finds the average fraction of cells of type2 around
        among the k-nearest cells around cells of type1. Thus, here, instead of cells 
        within radius, the k-nearest cells are considered. The fraction is 
        calculated for each possible combination of cell types.

        Args:
            k: Integer, number of nearest neighbours to consider. Defaults to 50.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            is the fraction of cells of type2 among the k-nearest cells around cells of type1.
        """

        returns = {}

        # Calculate distance matrix if not already done
        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()
        
        for cell_type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[cell_type]
            distances = self.cell_distances[mask, :]
            close_cells_idx = np.argpartition(distances, k, axis=1)[:, :k]
            counts = np.zeros(len(self.cell_types_set))

            for i in range(close_cells_idx.shape[0]):
                close_cells = self.cell_types[close_cells_idx[i,:]]
                counts += np.histogram(close_cells, bins=np.arange(len(self.cell_types_set)+1))[0]

            for i in range(counts.shape[0]):
                self.biomarkers["Fraction of " + self.cell_types_set[i] + " among " + str(k) + " closest cells next to " + cell_type] = counts[i] / (np.sum(mask)*k)
                returns["Fraction of " + self.cell_types_set[i] + " among " + str(k) + " closest cells next to " + cell_type] = counts[i] / (np.sum(mask)*k)
        return returns

    def smallest_distance_cell_type(self):
        """
        This function finds the smallest distance between each cell of type1 to
        cells of type2. The average of these distances is then calculated for each 
        possible combination of cell types.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the average smallest distances from type1 to type2.
        """

        returns = {}

        # Calculate distance matrix if not already done
        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()
        
        for type1 in self.cell_types_set:
            for type2 in self.cell_types_set:
                mask1 = self.cell_types == self.cell_types_dic[type1]
                mask2 = self.cell_types == self.cell_types_dic[type2]
                distances = self.cell_distances[mask1, :]
                distances = distances[:, mask2]
                if distances.size == 0:
                    dist = np.nan
                else:
                    dist = np.mean(np.min(distances, axis=1))
                self.biomarkers["Average smallest distance from " + type1 + " to " + type2] = dist
                returns["Average smallest distance from " + type1 + " to " + type2] = dist
        return returns

    def g_function(self, plot=True):
        """
        This function calculates the G-function for each possible combination of cell types
        (i.e. the cumulative sum of the smallest distances between cells of type1 and type2 that are
        less than a radius r). Then, the integral of the difference between the 
        empirical G-function and the theoretical G-function based on complete spatial randomness
        is calculated and used as a biomarker.

        Args:
            plot: Boolean, whether to plot the G-function or not. Defaults to False.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the integral differences.
        """

        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        if not "Lymphocyte_density_mu^2" in self.biomarkers.keys():
            self.density_cell_type()

        for type1 in self.cell_types_set:
            mask1 = self.cell_types == self.cell_types_dic[type1]
            distances = self.cell_distances[mask1, :]

            if distances.size == 0:
                self.biomarkers["G-function L1 for " + type1] = np.nan
                returns["G-function L1 for " + type1] = np.nan
            else:
                distances = np.min(distances, axis=1)
                distances = np.sort(distances)        
                radius = np.linspace(0, 200, 100).reshape(-1, 1)
                g_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                g_function_theo = 1 - np.exp(-np.pi * radius**2 * (self.cell_number / self.area))                
                g_diff = np.sum(g_function_theo - g_function_emp) / radius.shape[0]
                self.biomarkers["G-function L1 for " + type1] = g_diff
                returns["G-function L1 for " + type1] = g_diff
            
            for type2 in self.cell_types_set:
                distances = self.cell_distances[mask1, :]
                mask2 = self.cell_types == self.cell_types_dic[type2]
                distances = distances[:, mask2]

                if distances.size == 0:
                    self.biomarkers["G-function L1 for " + type1 + " to " + type2] = np.nan
                    returns["G-function L1 for " + type1 + " to " + type2] = np.nan
                    continue

                distances = np.min(distances, axis=1)
                distances = np.sort(distances)        
                radius = np.linspace(0, 200, 100).reshape(-1, 1)
                g_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                g_function_theo = 1 - np.exp(-np.pi * radius**2 * (self.biomarkers[type2 + "_density_mu^2"]))
                g_diff = np.sum(g_function_theo - g_function_emp) / radius.shape[0]
                self.biomarkers["G-function L1 for " + type1 + " to " + type2] = g_diff
                returns["G-function L1 for " + type1 + " to " + type2] = g_diff

                if plot:
                    if os.path.exists("./g_function") == False:
                        os.mkdir("./g_function")
                    plt.figure()
                    plt.plot(radius, g_function_emp, label="Empirical")
                    plt.plot(radius, g_function_theo, label="Theoretical")
                    plt.xlabel("Radius (mu)")
                    plt.ylabel("G-function")
                    plt.title("G-function for " + type1 + " to " + type2)
                    plt.legend()
                    plt.savefig("./g_function/G_function_" + self.name + "_" + type1 + "_to_" + type2 + ".png")

        return returns
    
    def k_function(self, plot=False):
        """
        This function calculates the K-function for each possible combination of cell types
        (i.e. the cumulative distribution of the all the distances between cells of type1 and type2 that are
        less than a radius r). Then, the integral of the difference between the 
        empirical K-function and the theoretical K-function based on complete spatial randomness
        is calculated and used as a biomarker.

        Args:
            plot: Boolean, whether to plot the K-function or not. Defaults to False.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the integral differences.
        """

        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        if not "Lymphocyte_density_mu^2" in self.biomarkers.keys():
            self.density_cell_type()

        for type1 in self.cell_types_set:
            mask1 = self.cell_types == self.cell_types_dic[type1]
            distances = self.cell_distances[mask1, :]

            if distances.size == 0:
                self.biomarkers["K-function L1 for " + type1] = np.nan
                returns["K-function L1 for " + type1] = np.nan
            else:
                distances = distances.reshape(1, -1)
                radius = np.linspace(0, 6000, 100).reshape(-1, 1)
                k_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                k_function_theo = np.pi * radius**2 * (self.cell_number - self.cell_types_number[type1]) / self.area
                k_diff = np.sum(k_function_theo - k_function_emp) / radius.shape[0]
                self.biomarkers["K-function L1 for " + type1] = k_diff
                returns["K-function L1 for " + type1] = k_diff
            
            for type2 in self.cell_types_set:
                distances = self.cell_distances[mask1, :]
                mask2 = self.cell_types == self.cell_types_dic[type2]
                distances = distances[:, mask2]

                if distances.size == 0:
                    self.biomarkers["K-function L1 for " + type1 + " to " + type2] = np.nan
                    returns["K-function L1 for " + type1 + " to " + type2] = np.nan
                    continue
                
                distances = distances.reshape(1, -1)
                radius = np.linspace(0, 6000, 100).reshape(-1, 1)
                k_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                k_function_theo = np.pi * radius**2 * self.biomarkers[type2 + "_density_mu^2"]
                k_diff = np.sum(k_function_theo - k_function_emp) / radius.shape[0]
                self.biomarkers["K-function L1 for " + type1 + " to " + type2] = k_diff
                returns["K-function L1 for " + type1 + " to " + type2] = k_diff

                if plot:
                    if os.path.exists("./k_function") == False:
                        os.mkdir("./k_function")
                    plt.figure()
                    plt.plot(radius, k_function_emp, label="Empirical")
                    plt.plot(radius, k_function_theo, label="Theoretical")
                    plt.xlabel("Radius (mu)")
                    plt.ylabel("K-function")
                    plt.title("K-function for " + type1 + " to " + type2)
                    plt.legend()
                    plt.show()
                    plt.savefig("./k_function/K_function_" + self.name + "_" + type1 + "_to_" + type2 + ".png")

        return returns
    
    def fraction_cellular_neighbourhoods(self):
        """
        This function calculate the fraction of cells that belong to a cellular neighbourhood.
        Moreover, it calculates this fraction for each cell type leading to
        #(cellular neighbourhood) times #(cell types) biomarkers.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the cell fractions.
        """

        returns = {}

        if hasattr(self, "neighbourhoods_labels") == False:
            return returns

        distribution = np.bincount(self.neighbourhoods_labels, minlength=self.neighbourhoods_number) / self.cell_number
        
        for i in range(self.neighbourhoods_number):
            self.biomarkers["Fraction of cellular neighbourhood " + str(i) + " / " + str(self.neighbourhoods_number)] = distribution[i]
            returns["Fraction of cellular neighbourhoods " + str(i) + " / " + str(self.neighbourhoods_number)] = distribution[i]

        for type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[type]

            distribution = np.bincount(np.array(self.neighbourhoods_labels[mask], dtype=int), minlength=self.neighbourhoods_number) / np.sum(mask)
            for i in range(self.neighbourhoods_number):
                self.biomarkers["Fraction of " + type + " that belong to neighbourhood " + str(i) + " / " + str(self.neighbourhoods_number)] = distribution[i]
                returns["Fraction of " + type + " that belong to neighbourhood " + str(i) + " / " + str(self.neighbourhoods_number)] = distribution[i]

        return returns
    
    def entropy_cellular_neighbourhoods(self):
        """
        This function calculates the entropy of the cellular neighbourhood distribution 
        of all cell types and of each individual cell type.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the entropies.
        """

        returns = {}

        if hasattr(self, "neighbourhoods_labels") == False:
            return returns

        distribution = np.bincount(self.neighbourhoods_labels) / self.cell_number

        self.biomarkers["Entropy of cellular neighbourhood distribution for " + str(self.neighbourhoods_number) + " neighbourhoods"] = entropy(distribution, base=2)
        returns["Entropy of cellullar neighbourhood distribution for " + str(self.neighbourhoods_number) + " neighbourhoods"] = entropy(distribution, base=2)

        for type in self.cell_types_set:
            mask = self.cell_types == self.cell_types_dic[type]
            distribution = np.bincount(self.neighbourhoods_labels[mask]) / np.sum(mask)
            self.biomarkers["Entropy of cellular neighbourhood distribution of " + type + " for " + str(self.neighbourhoods_number) + " neighbourhoods"] = entropy(distribution, base=2)
            returns["Entropy of cellular neighbourhood distribution of " + type + " for " + str(self.neighbourhoods_number) + " neighbourhoods"] = entropy(distribution, base=2)

        return returns

    def g_function_cellular_neighbourhoods(self):
        """
        This function calculates the G-function for each possible combination of neighbourhood types
        (i.e. the cumulative sum of the smallest distances between cells of neighbourhood1 
        and neighbourhood that are less than a radius r). Then, the integral of the difference between the 
        empirical G-function and the theoretical G-function based on complete spatial randomness
        is calculated and used as a biomarker.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the integral differences.
        """

        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        if hasattr(self, "neighbourhoods_labels") == False:
            return returns

        for cluster1 in range(self.neighbourhoods_number):
            mask1 = self.neighbourhoods_labels == cluster1
            distances = self.cell_distances[mask1, :]

            if distances.size == 0:
                self.biomarkers["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                returns["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = np.nan
            else:
                distances = np.min(distances, axis=1)
                distances = np.sort(distances)        
                radius = np.linspace(0, 200, 100).reshape(-1, 1)
                g_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                g_function_theo = 1 - np.exp(-np.pi * radius**2 * (self.cell_number / self.area))                
                g_diff = np.sum(g_function_theo - g_function_emp) / radius.shape[0]
                self.biomarkers["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = g_diff
                returns["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = g_diff
            
            for cluster2 in range(self.neighbourhoods_number):
                distances = self.cell_distances[mask1, :]
                mask2 = self.neighbourhoods_labels == cluster2
                distances = distances[:, mask2]

                if distances.size == 0:
                    self.biomarkers["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                    returns["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                    continue

                distances = np.min(distances, axis=1)
                distances = np.sort(distances)        
                radius = np.linspace(0, 200, 100).reshape(-1, 1)
                g_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                g_function_theo = 1 - np.exp(-np.pi * radius**2 * (np.sum(mask2) / self.area))
                g_diff = np.sum(g_function_theo - g_function_emp) / radius.shape[0]
                self.biomarkers["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = g_diff
                returns["G-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = g_diff

        return returns
    
    def k_function_cellular_neighbourhoods(self):
        """
        This function calculates the K-function for each possible combination of neighbourhood types
        (i.e. the cumulative distribution of the all the distances between cells of neighbourhood1 
        and neighbourhood2 that are less than a radius r). Then, the integral of the difference between the 
        empirical K-function and the theoretical K-function based on complete spatial randomness
        is calculated and used as a biomarker.
        
        Returns:
            returns: Dictionary, keys are the biomarker names and values 
            are the integral differences.
        """

        returns = {}

        if hasattr(self, "cell_distances") == False:
            self.calculate_cell_distances_()

        if hasattr(self, "neighbourhoods_labels") == False:
            return returns

        for cluster1 in range(self.neighbourhoods_number):
            mask1 = self.neighbourhoods_labels == cluster1
            distances = self.cell_distances[mask1, :]

            if distances.size == 0:
                self.biomarkers["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                returns["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = np.nan
            else:
                distances = distances.reshape(1, -1)
                radius = np.linspace(0, 6000, 100).reshape(-1, 1)
                k_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                k_function_theo = np.pi * radius**2 * (self.cell_number - np.sum(mask1)) / self.area
                k_diff = np.sum(k_function_theo - k_function_emp) / radius.shape[0]
                self.biomarkers["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = k_diff
                returns["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number)] = k_diff
            
            for cluster2 in range(self.neighbourhoods_number):
                distances = self.cell_distances[mask1, :]
                mask2 = self.neighbourhoods_labels == cluster2
                distances = distances[:, mask2]

                if distances.size == 0:
                    self.biomarkers["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                    returns["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = np.nan
                    continue
                
                distances = distances.reshape(1, -1)
                radius = np.linspace(0, 6000, 100).reshape(-1, 1)
                k_function_emp = np.sum(distances < radius + 12.5, axis=1) / np.sum(mask1)
                k_function_theo = np.pi * radius**2 * np.sum(mask2) / self.area
                k_diff = np.sum(k_function_theo - k_function_emp) / radius.shape[0]
                self.biomarkers["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = k_diff
                returns["K-function L1 for neighbourhood " + str(cluster1+1) + " / " + str(self.neighbourhoods_number) + " to neighbourhood " + str(cluster2+1) + " / " + str(self.neighbourhoods_number)] = k_diff
    
        return returns
